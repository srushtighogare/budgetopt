"""Optimization engine API (report architecture item 4)."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_accessible_team, get_current_user, require_admin
from app.db.models import AllocationPolicy, AuditEvent, OptimizationRun, Role, Team, User
from app.db.session import get_db
from app.services.optimizer.service import RunParams, apply_solution, run_optimization

router = APIRouter(prefix="/optimization", tags=["optimization"])


class RunIn(BaseModel):
    team_id: int | None = None  # None = organisation-wide traffic
    method: Literal["nsga2", "ilp"] = "nsga2"
    monthly_requests: int | None = Field(default=None, gt=0, le=1_000_000_000)
    budget: float | None = Field(default=None, gt=0)
    q_min: dict[Literal["low", "medium", "high"], float] | None = None
    latency_max_ms: float | None = Field(default=None, gt=0)
    pop_size: int = Field(default=100, ge=20, le=400)
    generations: int = Field(default=150, ge=10, le=1000)
    ilp_objective: Literal["max_quality", "min_cost"] = "max_quality"
    quality_floor: float | None = Field(default=None, ge=0, le=10)

    @model_validator(mode="after")
    def _q_range(self):
        for v in (self.q_min or {}).values():
            if not 0 <= v <= 10:
                raise ValueError("q_min values must be between 0 and 10")
        return self


class ApplyIn(BaseModel):
    team_id: int
    preset: str | None = None  # cost_saver | balanced | quality_first | optimal
    index: int | None = Field(default=None, ge=0)
    slot: Literal["primary", "savings"] = "primary"

    @model_validator(mode="after")
    def _one_of(self):
        if (self.preset is None) == (self.index is None):
            raise ValueError("give exactly one of preset or index")
        return self


def _summary(run: OptimizationRun) -> dict:
    presets = run.inputs.get("presets", {})
    front = run.pareto_front or []
    return {
        "id": run.id, "team_id": run.team_id, "method": run.method,
        "created_at": run.created_at, "points": len(front), "selected_index": run.selected_index,
        "presets": {k: {"index": i, **{m: front[i][m] for m in
                                       ("monthly_cost", "quality", "latency_ms")},
                        "feasible": front[i].get("feasible", True)}
                    for k, i in presets.items() if i < len(front)},
        "baselines": run.inputs.get("baselines", []),
        "metrics": run.inputs.get("metrics", {}),
        "notes": run.inputs.get("notes", []),
        "traffic_source": run.inputs.get("traffic_source"),
    }


async def _run_for_user(run_id: int, user: User, db: AsyncSession) -> OptimizationRun:
    run = await db.get(OptimizationRun, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Run not found")
    if run.team_id is not None:
        await get_accessible_team(run.team_id, user, db, action="view_optimization")
    elif user.role != Role.ORG_ADMIN:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Organisation-wide runs are admin-only")
    return run


@router.post("/runs", status_code=status.HTTP_201_CREATED)
async def create_run(
    body: RunIn, user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> dict:
    team = await get_accessible_team(body.team_id, user, db) if body.team_id else None
    try:
        run = await run_optimization(db, team, RunParams(
            **body.model_dump(exclude={"team_id"})), user.id)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    db.add(AuditEvent(user_id=user.id, action="run_optimization", resource="optimization_run",
                      resource_id=str(run.id), details={"method": body.method}))
    await db.commit()
    return _summary(run)


@router.get("/runs")
async def list_runs(
    team_id: int | None = None, user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    q = select(OptimizationRun).order_by(OptimizationRun.id.desc()).limit(50)
    if team_id is not None:
        await get_accessible_team(team_id, user, db, action="list_optimization")
        q = q.where(OptimizationRun.team_id == team_id)
    elif user.role != Role.ORG_ADMIN:
        q = q.where(OptimizationRun.team_id == user.team_id)
    return [_summary(r) for r in await db.scalars(q)]


@router.get("/runs/{run_id}")
async def get_run(
    run_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    run = await _run_for_user(run_id, user, db)
    return {**_summary(run), "front": [
        {"index": i, **{k: p[k] for k in ("monthly_cost", "quality", "latency_ms")}}
        for i, p in enumerate(run.pareto_front or [])], "inputs": run.inputs}


@router.get("/runs/{run_id}/points/{index}")
async def get_point(
    run_id: int, index: int, user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    run = await _run_for_user(run_id, user, db)
    if not 0 <= index < len(run.pareto_front or []):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such point")
    return {"index": index, **run.pareto_front[index]}


@router.post("/runs/{run_id}/apply")
async def apply(
    run_id: int, body: ApplyIn, user: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> dict:
    run = await _run_for_user(run_id, user, db)
    team = await get_accessible_team(body.team_id, user, db, action="apply_policy")
    index = body.index
    if body.preset is not None:
        index = run.inputs.get("presets", {}).get(body.preset)
        if index is None:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                f"run has no preset '{body.preset}'")
    try:
        policy = await apply_solution(db, run, team, index=index, slot=body.slot,
                                      user_id=user.id, label=body.preset)
    except IndexError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {"policy_id": policy.id, "name": policy.name, "slot": body.slot,
            "expected_metrics": policy.expected_metrics, "rules": policy.rules}


@router.delete("/teams/{team_id}/policy", status_code=status.HTTP_204_NO_CONTENT)
async def clear_policy(
    team_id: int, slot: Literal["primary", "savings"] = "primary",
    user: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> None:
    team = await get_accessible_team(team_id, user, db, action="clear_policy")
    if slot == "savings":
        team.savings_policy_id = None
    else:
        team.policy_id = None
    db.add(AuditEvent(user_id=user.id, action="clear_policy", resource="team",
                      resource_id=str(team_id), details={"slot": slot}))
    await db.commit()


@router.get("/policies/{policy_id}")
async def get_policy(
    policy_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    policy = await db.get(AllocationPolicy, policy_id)
    if policy is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Policy not found")
    if user.role != Role.ORG_ADMIN:
        team = await db.get(Team, user.team_id) if user.team_id else None
        if team is None or policy_id not in (team.policy_id, team.savings_policy_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Policy not found")
    return {"id": policy.id, "name": policy.name, "run_id": policy.optimization_run_id,
            "rules": policy.rules, "expected_metrics": policy.expected_metrics,
            "created_at": policy.created_at}
