"""Run, store and apply optimizations."""

import asyncio
from dataclasses import dataclass

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AllocationPolicy, AuditEvent, OptimizationRun, Team
from app.services.optimizer import ilp, nsga2
from app.services.optimizer.policy import allocation_to_rules
from app.services.optimizer.problem import OptimizationInputs, build_inputs

PRESETS = ("cost_saver", "balanced", "quality_first")


@dataclass
class RunParams:
    method: str = "nsga2"  # nsga2 | ilp
    monthly_requests: int | None = None
    budget: float | None = None
    q_min: dict[str, float] | None = None
    latency_max_ms: float | None = None
    pop_size: int = 100
    generations: int = 150
    ilp_objective: str = "max_quality"
    quality_floor: float | None = None


def _point(inp: OptimizationInputs, X: np.ndarray, cost: float, quality: float,
           latency: float, **extra) -> dict:
    return {"monthly_cost": round(cost, 6), "quality": round(quality, 4),
            "latency_ms": round(latency, 1), "rules": allocation_to_rules(inp, X), **extra}


def _baselines(inp: OptimizationInputs) -> list[dict]:
    """What the same traffic costs without optimisation."""
    out = []
    for j, m in enumerate(inp.models):
        X = inp.single_model_allocation(j)
        if X is not None:
            c, q, lat = inp.evaluate(X)
            out.append({"name": f"all traffic to {m.display_name}", "monthly_cost": round(c, 6),
                        "quality": round(q, 4), "latency_ms": round(lat, 1)})
    # The pre-optimizer router: cheapest eligible model for every class
    X = np.zeros(inp.eligible.shape)
    for i in range(X.shape[0]):
        if inp.eligible[i].any():
            X[i, np.argmin(np.where(inp.eligible[i], inp.cost[i], np.inf))] = 1.0
    c, q, lat = inp.evaluate(X)
    out.append({"name": "cheapest eligible per class", "monthly_cost": round(c, 6),
                "quality": round(q, 4), "latency_ms": round(lat, 1)})
    return sorted(out, key=lambda b: b["monthly_cost"])


def _validate_against_lp(inp: OptimizationInputs, front: nsga2.FrontResult) -> list[dict]:
    """Gap between NSGA-II and the exact LP optimum at a few budget levels."""
    lo, hi = front.solutions[0].monthly_cost, front.solutions[-1].monthly_cost
    checks = []
    for frac in (0.1, 0.5, 1.0):
        budget = lo + frac * (hi - lo)
        lp = ilp.solve(inp, "max_quality", budget=budget, relax=True)
        best = max((s.quality for s in front.solutions if s.monthly_cost <= budget + 1e-9),
                   default=None)
        if lp.quality and best is not None:
            checks.append({"budget": round(budget, 6), "lp_optimum_quality": round(lp.quality, 4),
                           "nsga2_quality": round(best, 4),
                           "gap_pct": round(100 * (lp.quality - best) / lp.quality, 3)})
    return checks


def _compute(inp: OptimizationInputs, p: RunParams) -> dict:
    """CPU-bound part, run in a worker thread."""
    if p.method == "nsga2":
        front = nsga2.optimize(inp, pop_size=p.pop_size, generations=p.generations)
        if inp.budget and not front.solutions[0].feasible:
            inp.notes.append(
                f"Budget ${inp.budget:,.2f}/month cannot cover {inp.monthly_requests:,} requests "
                f"within the quality floors; the cheapest compliant allocation costs "
                f"${front.solutions[0].monthly_cost:,.2f}/month (shown, marked infeasible).")
        points = [_point(inp, s.allocation, s.monthly_cost, s.quality, s.latency_ms,
                         feasible=s.feasible) for s in front.solutions]
        return {"points": points, "presets": front.presets,
                "metrics": {"hypervolume": round(front.hypervolume, 4),
                            "evaluations": front.evaluations, "generations": front.generations,
                            "lp_validation": _validate_against_lp(inp, front)}}
    if p.method == "ilp":
        r = ilp.solve(inp, p.ilp_objective, budget=p.budget, quality_floor=p.quality_floor,
                      mean_latency_max_ms=p.latency_max_ms)
        if r.allocation is None:
            return {"points": [], "presets": {}, "metrics": {"status": r.status,
                                                              "solve_ms": round(r.wall_ms, 1)}}
        point = _point(inp, r.allocation, r.monthly_cost, r.quality, r.latency_ms)
        return {"points": [point], "presets": {"optimal": 0},
                "metrics": {"status": r.status, "solve_ms": round(r.wall_ms, 1),
                            "objective": p.ilp_objective}}
    raise ValueError("method must be 'nsga2' or 'ilp'")


async def run_optimization(db: AsyncSession, team: Team | None, p: RunParams,
                           user_id: int | None) -> OptimizationRun:
    inp = await build_inputs(db, team=team, monthly_requests=p.monthly_requests,
                             budget=p.budget, q_min=p.q_min, latency_max_ms=p.latency_max_ms)
    if not inp.models:
        raise ValueError("no active models with pricing")
    result = await asyncio.to_thread(_compute, inp, p)

    estimated = {m.name: sorted(m.quality_estimated) for m in inp.models if m.quality_estimated}
    run = OptimizationRun(
        team_id=team.id if team else None, method=p.method, created_by=user_id,
        inputs={
            "params": {k: v for k, v in p.__dict__.items()},
            "monthly_requests": inp.monthly_requests, "budget": inp.budget,
            "q_min": inp.q_min, "traffic_source": inp.traffic_source,
            "classes": [{"key": c.key, "share": round(c.share, 4), "in_tokens": c.in_tokens,
                         "out_tokens": c.out_tokens} for c in inp.classes],
            "models": [{"model_id": m.model_id, "model": m.name,
                        "input_per_1m": m.input_per_1m, "output_per_1m": m.output_per_1m,
                        "quality": {k: round(v, 2) for k, v in m.quality.items()}}
                       for m in inp.models],
            "quality_estimated_from_model_mean": estimated,
            "eligible_pairs": int(inp.eligible.sum()), "notes": inp.notes,
            "baselines": _baselines(inp), "presets": result["presets"],
            "metrics": result["metrics"],
        },
        pareto_front=result["points"],
    )
    db.add(run)
    await db.commit()
    return run


async def apply_solution(db: AsyncSession, run: OptimizationRun, team: Team, *,
                         index: int, slot: str, user_id: int,
                         label: str | None = None) -> AllocationPolicy:
    if not run.pareto_front or not 0 <= index < len(run.pareto_front):
        raise IndexError("no such solution in this run")
    point = run.pareto_front[index]
    preset = label or next((k for k, v in run.inputs.get("presets", {}).items() if v == index),
                           None)
    policy = AllocationPolicy(
        name=f"{preset or f'point {index}'} (run #{run.id})", optimization_run_id=run.id,
        rules=point["rules"],
        expected_metrics={k: point[k] for k in ("monthly_cost", "quality", "latency_ms")},
    )
    db.add(policy)
    await db.flush()
    if slot == "savings":
        team.savings_policy_id = policy.id
    else:
        team.policy_id = policy.id
    run.selected_index = index
    db.add(AuditEvent(user_id=user_id, action="apply_policy", resource="team",
                      resource_id=str(team.id),
                      details={"policy_id": policy.id, "run_id": run.id, "index": index,
                               "slot": slot}))
    await db.commit()
    return policy
