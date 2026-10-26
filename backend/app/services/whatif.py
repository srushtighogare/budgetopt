"""What-if analysis (report objective O6, and Economic Feasibility: "compare alternative
model allocations" and "quantify potential savings before changing production").

Replays a team's logged requests from the last N days under one or more scenarios
and reports what they would have cost, plus expected quality and latency, without
calling any LLM. A scenario combines:

  allocation   "actual" (what really happened) | "current_policy" | "cheapest" (cheapest
               model meeting the quality floor) | "model" (all traffic to one model) |
               "policy" (a saved allocation policy) | "run_point" (any point of an
               optimization run's Pareto front)
  prices       per-model price multipliers (e.g. a provider raising prices 20%)
  volume       growth in request volume (%)

Each replayed request keeps its real input tokens. Output tokens are scaled by how
verbose the replacement model is compared with the original one for that task type
(median output tokens in the profiling results), since models answer at different
lengths. Allocations are applied in expectation (shares, not random draws), so
results are deterministic.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    AllocationPolicy,
    OptimizationRun,
    ProfilingResult,
    Request,
    RequestStatus,
    Team,
)
from app.services.optimizer.problem import DEFAULT_Q_MIN, ModelProfile, _model_profiles

ALLOCATIONS = ("actual", "current_policy", "cheapest", "model", "policy", "run_point")
REPLAYED = (RequestStatus.SUCCESS, RequestStatus.FALLBACK_SUCCESS)


@dataclass
class Scenario:
    name: str
    allocation: str = "actual"
    model: str | None = None
    policy_id: int | None = None
    run_id: int | None = None
    point_index: int | None = None
    price_multipliers: dict[str, float] = field(default_factory=dict)
    volume_growth_pct: float = 0.0


@dataclass
class ReplayRow:
    task: str
    band: str
    model_id: int
    input_tokens: int
    output_tokens: int


async def _load_rows(db: AsyncSession, team: Team, since: datetime) -> tuple[list[ReplayRow],
                                                                              int]:
    rows = (await db.execute(
        select(Request.task_type, Request.complexity_band, Request.model_id,
               Request.input_tokens, Request.output_tokens)
        .where(Request.team_id == team.id, Request.created_at >= since,
               Request.status.in_(REPLAYED), Request.model_id.is_not(None),
               Request.task_type.is_not(None), Request.input_tokens.is_not(None))
    )).all()
    cache_hits = await db.scalar(select(func.count()).where(
        Request.team_id == team.id, Request.created_at >= since,
        Request.status == RequestStatus.CACHE_HIT))
    return [ReplayRow(t.value, (b.value if b else "medium"), m, i, o or 0)
            for t, b, m, i, o in rows], int(cache_hits or 0)


async def _verbosity(db: AsyncSession) -> dict[tuple[int, str], float]:
    """Median output tokens per (model, task) from profiling (same prompts for all models)."""
    rows = (await db.execute(
        select(ProfilingResult.model_id, ProfilingResult.task_type,
               func.percentile_cont(0.5).within_group(ProfilingResult.output_tokens))
        .where(ProfilingResult.output_tokens.is_not(None))
        .group_by(ProfilingResult.model_id, ProfilingResult.task_type))).all()
    return {(m, t.value): float(v) for m, t, v in rows if v}


def _cheapest_eligible(models: list[ModelProfile], task: str, band: str,
                       in_tok: float, out_tok: float) -> int:
    floor = DEFAULT_Q_MIN[band]
    ok = [m for m in models if m.quality.get(task, 0) >= floor]
    if not ok:  # same relaxation as the optimizer: best available quality
        ok = [max(models, key=lambda m: m.quality.get(task, 0))]
    return min(ok, key=lambda m: in_tok * m.input_per_1m + out_tok * m.output_per_1m).model_id


async def _rules_for(db: AsyncSession, team: Team, sc: Scenario) -> dict[str, list[dict]] | None:
    if sc.allocation == "current_policy":
        if not team.policy_id:
            raise ValueError(f"scenario '{sc.name}': team has no allocation policy")
        return (await db.get(AllocationPolicy, team.policy_id)).rules
    if sc.allocation == "policy":
        p = await db.get(AllocationPolicy, sc.policy_id) if sc.policy_id else None
        if p is None:
            raise ValueError(f"scenario '{sc.name}': policy not found")
        return p.rules
    if sc.allocation == "run_point":
        run = await db.get(OptimizationRun, sc.run_id) if sc.run_id else None
        if run is None or sc.point_index is None or not (
                0 <= sc.point_index < len(run.pareto_front or [])):
            raise ValueError(f"scenario '{sc.name}': run point not found")
        return run.pareto_front[sc.point_index]["rules"]
    return None


async def simulate(db: AsyncSession, team: Team, scenarios: list[Scenario],
                   days: int = 30) -> dict:
    since = datetime.now(UTC) - timedelta(days=days)
    rows, cache_hits = await _load_rows(db, team, since)
    models = await _model_profiles(db)
    by_id = {m.model_id: m for m in models}
    by_name = {m.name: m for m in models}
    verbose = await _verbosity(db)
    if not rows:
        return {"team_id": team.id, "days": days, "requests_replayed": 0,
                "cache_hits": cache_hits, "scenarios": [],
                "note": "no successful requests in this window to replay"}

    # Baseline for 'savings vs premium': everything on the priciest active model
    priciest = max(models, key=lambda m: m.input_per_1m + m.output_per_1m)
    results = []
    for sc in [Scenario("baseline: all to " + priciest.display_name, "model",
                        model=priciest.name), *scenarios]:
        if sc.allocation not in ALLOCATIONS:
            raise ValueError(f"unknown allocation '{sc.allocation}'")
        if sc.allocation == "model" and sc.model not in by_name:
            raise ValueError(f"scenario '{sc.name}': unknown or inactive model '{sc.model}'")
        unknown = set(sc.price_multipliers) - set(by_name)
        if unknown:
            raise ValueError(f"scenario '{sc.name}': unknown models in prices: {sorted(unknown)}")
        rules = await _rules_for(db, team, sc)
        mult = {by_name[n].model_id: f for n, f in sc.price_multipliers.items()}

        cost = quality = quality_weight = latency = 0.0
        by_model: dict[str, float] = defaultdict(float)
        by_task: dict[str, float] = defaultdict(float)
        for r in rows:
            if sc.allocation == "actual":
                split = {r.model_id: 1.0}
            elif sc.allocation == "model":
                split = {by_name[sc.model].model_id: 1.0}
            elif sc.allocation == "cheapest" or rules is None:
                split = {_cheapest_eligible(models, r.task, r.band, r.input_tokens,
                                            r.output_tokens): 1.0}
            else:
                rule = [x for x in rules.get(f"{r.task}:{r.band}", [])
                        if x["model_id"] in by_id]
                total = sum(x["share"] for x in rule)
                split = ({x["model_id"]: x["share"] / total for x in rule} if total
                         else {_cheapest_eligible(models, r.task, r.band, r.input_tokens,
                                                  r.output_tokens): 1.0})
            for mid, share in split.items():
                m = by_id.get(mid)
                if m is None:  # original model since deactivated: price it as cheapest
                    m = by_id[_cheapest_eligible(models, r.task, r.band, r.input_tokens,
                                                 r.output_tokens)]
                orig_v, new_v = verbose.get((r.model_id, r.task)), verbose.get((m.model_id,
                                                                                 r.task))
                out = r.output_tokens * (new_v / orig_v if orig_v and new_v else 1.0)
                c = share * mult.get(m.model_id, 1.0) * (
                    r.input_tokens * m.input_per_1m + out * m.output_per_1m) / 1e6
                cost += c
                if r.task in m.quality:  # unprofiled models don't count towards quality
                    quality += share * m.quality[r.task]
                    quality_weight += share
                latency += share * m.latency_p95.get(r.task, 0.0)
                by_model[m.display_name] += c
                by_task[r.task] += c

        growth = 1 + sc.volume_growth_pct / 100
        cost *= growth
        monthly = cost * 30 / days
        n = len(rows)
        results.append({
            "name": sc.name, "allocation": sc.allocation,
            "volume_growth_pct": sc.volume_growth_pct, "price_multipliers": sc.price_multipliers,
            "cost": round(cost, 6), "monthly_projection": round(monthly, 6),
            "fits_budget": monthly <= float(team.monthly_budget),
            "mean_quality": round(quality / quality_weight, 3) if quality_weight else None,
            "quality_coverage": round(quality_weight / n, 3),
            "mean_p95_latency_ms": round(latency / n, 1),
            "cost_by_model": {k: round(v * growth, 6) for k, v in
                              sorted(by_model.items(), key=lambda kv: -kv[1])},
            "cost_by_task": {k: round(v * growth, 6) for k, v in
                             sorted(by_task.items(), key=lambda kv: -kv[1])},
        })

    actual = next((x for x in results if x["allocation"] == "actual"), None)
    base = results[0]
    for x in results:
        if actual:
            x["savings_vs_actual"] = round(actual["cost"] - x["cost"], 6)
        x["savings_vs_baseline"] = round(base["cost"] - x["cost"], 6)
        x["savings_vs_baseline_pct"] = (round(100 * (base["cost"] - x["cost"]) / base["cost"], 2)
                                        if base["cost"] else None)
    return {"team_id": team.id, "days": days, "requests_replayed": len(rows),
            "cache_hits": cache_hits, "monthly_budget": float(team.monthly_budget),
            "scenarios": results}
