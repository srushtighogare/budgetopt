"""What-if simulation and savings reports."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_accessible_team, get_current_user, require_manager
from app.db.models import AuditEvent, Role, User
from app.db.session import get_db
from app.services.savings import savings_report
from app.services.whatif import Scenario, simulate

router = APIRouter(tags=["analysis"])


class ScenarioIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    allocation: Literal["actual", "current_policy", "cheapest", "model", "policy",
                        "run_point"] = "actual"
    model: str | None = None
    policy_id: int | None = None
    run_id: int | None = None
    point_index: int | None = Field(default=None, ge=0)
    price_multipliers: dict[str, float] = Field(default_factory=dict)
    volume_growth_pct: float = Field(default=0.0, ge=-100, le=10_000)


class WhatIfIn(BaseModel):
    team_id: int
    days: int = Field(default=30, ge=1, le=365)
    scenarios: list[ScenarioIn] = Field(min_length=1, max_length=10)


@router.post("/whatif")
async def whatif(
    body: WhatIfIn, user: User = Depends(require_manager), db: AsyncSession = Depends(get_db)
) -> dict:
    """Replay the team's recent requests under alternative allocations, prices and volumes."""
    team = await get_accessible_team(body.team_id, user, db, action="whatif")
    for sc in body.scenarios:
        if any(m <= 0 for m in sc.price_multipliers.values()):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                                "price multipliers must be positive")
    try:
        result = await simulate(db, team, [Scenario(**sc.model_dump()) for sc in body.scenarios],
                                days=body.days)
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e)) from e
    db.add(AuditEvent(user_id=user.id, action="whatif", resource="team",
                      resource_id=str(team.id), details={"scenarios": len(body.scenarios)}))
    await db.commit()
    return result


@router.get("/reports/savings")
async def savings(
    team_id: int | None = None, days: int = Query(default=30, ge=1, le=365),
    platform_cost_per_month: float | None = Query(default=None, ge=0),
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
) -> dict:
    """Realised savings vs the priciest-model baseline, with optional feasibility check."""
    if team_id is not None:
        await get_accessible_team(team_id, user, db, action="view_savings")
    elif user.role != Role.ORG_ADMIN:
        if user.team_id is None:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Organisation report is admin-only")
        team_id = user.team_id
    report = await savings_report(db, user.organization_id, team_id=team_id, days=days,
                                  platform_cost_per_month=platform_cost_per_month)
    db.add(AuditEvent(user_id=user.id, action="view_savings", resource="report",
                      resource_id=str(team_id) if team_id else "org"))
    await db.commit()
    return report
