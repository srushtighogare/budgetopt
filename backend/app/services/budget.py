"""Team budget state for the current calendar month.

Spend is derived from `requests` (the source of truth) rather than a running
counter, so it can never drift from the logged costs.
"""

import enum
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Request, Team


class BudgetState(enum.StrEnum):
    OK = "ok"
    SOFT_ALERT = "soft_alert"   # routing switches to the team's cost-saver policy
    HARD_BLOCK = "hard_block"   # request rejected before reaching any LLM


@dataclass(frozen=True)
class BudgetStatus:
    period_start: datetime
    monthly_budget: Decimal
    spend: Decimal
    usage_ratio: float
    state: BudgetState


def month_start(now: datetime | None = None) -> datetime:
    now = now or datetime.now(UTC)
    return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def classify(spend: Decimal, budget: Decimal, threshold: float) -> tuple[float, BudgetState]:
    ratio = float(spend / budget) if budget > 0 else 1.0
    if spend >= budget:
        return ratio, BudgetState.HARD_BLOCK
    if ratio >= threshold:
        return ratio, BudgetState.SOFT_ALERT
    return ratio, BudgetState.OK


async def team_spend(db: AsyncSession, team_id: int, since: datetime) -> Decimal:
    total = await db.scalar(
        select(func.coalesce(func.sum(Request.cost), 0)).where(
            Request.team_id == team_id, Request.created_at >= since
        )
    )
    return Decimal(total)


async def budget_status(db: AsyncSession, team: Team, now: datetime | None = None) -> BudgetStatus:
    start = month_start(now)
    spend = await team_spend(db, team.id, start)
    ratio, state = classify(spend, team.monthly_budget, team.alert_threshold)
    return BudgetStatus(start, team.monthly_budget, spend, round(ratio, 4), state)
