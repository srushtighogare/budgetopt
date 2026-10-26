"""Savings report and economic feasibility (report: "Cost and Economic Feasibility").

Every routed request already stores what it cost and what the un-optimised
baseline (the priciest model) would have cost for the same tokens. This module
aggregates those per period: realised savings split into routing savings and
cache savings, spend by model / task / team, a daily series, and, given the
platform's own running cost, the net monthly benefit and ROI. The report defines
BudgetOpt as economically feasible when savings exceed operating cost.
"""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

from sqlalchemy import Date, case, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LLMModel, Request, RequestStatus, RoutingDecision, Team

ZERO = Decimal(0)


def _f(x) -> float:
    return round(float(x or 0), 6)


async def savings_report(db: AsyncSession, organization_id: int, *, team_id: int | None = None,
                         days: int = 30, platform_cost_per_month: float | None = None,
                         now: datetime | None = None) -> dict:
    now = now or datetime.now(UTC)
    since = now - timedelta(days=days)
    base = (select(Request).join(Team, Team.id == Request.team_id)
            .outerjoin(RoutingDecision, RoutingDecision.request_id == Request.id)
            .where(Team.organization_id == organization_id, Request.created_at >= since))
    if team_id is not None:
        base = base.where(Request.team_id == team_id)
    sub = base.with_only_columns(
        Request.id, Request.team_id, Request.model_id, Request.task_type, Request.status,
        Request.cost, Request.latency_ms, Request.created_at,
        RoutingDecision.baseline_cost, RoutingDecision.savings,
        RoutingDecision.fallback_used).subquery()

    is_cache = sub.c.status == RequestStatus.CACHE_HIT
    totals = (await db.execute(select(
        func.count(), func.coalesce(func.sum(sub.c.cost), 0),
        func.coalesce(func.sum(sub.c.baseline_cost), 0),
        func.coalesce(func.sum(case((is_cache, 0), else_=sub.c.savings)), 0),
        func.coalesce(func.sum(case((is_cache, sub.c.savings), else_=0)), 0),
        func.count().filter(is_cache),
        func.count().filter(sub.c.status == RequestStatus.BLOCKED_BUDGET),
        func.count().filter(sub.c.status == RequestStatus.FAILED),
        func.count().filter(sub.c.fallback_used.is_(True)),
        func.percentile_cont(0.5).within_group(sub.c.latency_ms),
    ))).one()
    (n, spend, baseline, routing_sav, cache_sav, cache_hits, blocked, failed, fallbacks,
     p50) = totals
    savings = Decimal(routing_sav) + Decimal(cache_sav)

    async def breakdown(col, label_expr=None) -> list[dict]:
        key = label_expr if label_expr is not None else col
        q = select(key, func.count(), func.coalesce(func.sum(sub.c.cost), 0),
                   func.coalesce(func.sum(sub.c.savings), 0)).select_from(sub)
        if label_expr is not None and col is sub.c.model_id:
            q = q.outerjoin(LLMModel, LLMModel.id == sub.c.model_id)
        if label_expr is not None and col is sub.c.team_id:
            q = q.join(Team, Team.id == sub.c.team_id)
        rows = (await db.execute(q.group_by(key).order_by(func.sum(sub.c.cost).desc()))).all()
        return [{"name": (k.value if hasattr(k, "value") else k) or "—", "requests": c,
                 "spend": _f(s), "savings": _f(v)} for k, c, s, v in rows]

    day = cast(func.timezone("UTC", sub.c.created_at), Date)
    daily_rows = (await db.execute(select(
        day, func.coalesce(func.sum(sub.c.cost), 0),
        func.coalesce(func.sum(sub.c.baseline_cost), 0)).group_by(day).order_by(day))).all()

    monthly_factor = Decimal(30) / Decimal(days)
    monthly_savings = savings * monthly_factor
    feasibility = None
    if platform_cost_per_month is not None:
        cost = Decimal(str(platform_cost_per_month))
        net = monthly_savings - cost
        feasibility = {
            "platform_cost_per_month": _f(cost),
            "llm_savings_per_month": _f(monthly_savings),
            "net_benefit_per_month": _f(net),
            "roi_pct": round(float(100 * net / cost), 1) if cost > 0 else None,
            "economically_feasible": bool(net > 0),
        }

    return {
        "period": {"from": since.isoformat(), "to": now.isoformat(), "days": days},
        "team_id": team_id,
        "requests": n, "cache_hits": cache_hits, "blocked_by_budget": blocked,
        "failed": failed, "fallbacks": fallbacks,
        "spend": _f(spend), "baseline_spend": _f(baseline), "savings": _f(savings),
        "routing_savings": _f(routing_sav), "cache_savings": _f(cache_sav),
        "savings_pct": round(float(100 * savings / Decimal(baseline)), 2)
        if Decimal(baseline) > 0 else None,
        "monthly_projection": {"spend": _f(Decimal(spend) * monthly_factor),
                               "savings": _f(monthly_savings)},
        "median_latency_ms": round(float(p50), 1) if p50 is not None else None,
        "by_model": await breakdown(sub.c.model_id, LLMModel.display_name),
        "by_task": await breakdown(sub.c.task_type),
        "by_team": await breakdown(sub.c.team_id, Team.name) if team_id is None else None,
        "daily": [{"date": d.isoformat() if isinstance(d, date) else str(d),
                   "spend": _f(s), "baseline": _f(b)} for d, s, b in daily_rows],
        "economic_feasibility": feasibility,
    }
