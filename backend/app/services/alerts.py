"""Alert creation with de-duplication.

Budget alerts fire once per team per month and per kind (soft threshold, hard
block, forecast overrun). Anomaly alerts are keyed by subject and hour, so one
spike produces one alert however many times the detector runs.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Alert, AlertKind, Team
from app.services.budget import BudgetState, BudgetStatus, month_start


async def ensure_alert(db: AsyncSession, team: Team, kind: AlertKind, message: str, *,
                       key: str, since: datetime, details: dict | None = None,
                       user_id: int | None = None) -> Alert | None:
    """Create the alert unless one with the same team/kind/key exists since `since`."""
    existing = await db.scalar(
        select(Alert.id).where(Alert.team_id == team.id, Alert.kind == kind,
                               Alert.created_at >= since,
                               Alert.details["key"].as_string() == key))
    if existing:
        return None
    alert = Alert(team_id=team.id, user_id=user_id, kind=kind, message=message,
                  details={"key": key, **(details or {})})
    db.add(alert)
    await db.commit()
    return alert


async def budget_alerts(db: AsyncSession, team: Team, status: BudgetStatus) -> Alert | None:
    period = month_start()
    key = period.date().isoformat()
    if status.state == BudgetState.HARD_BLOCK:
        return await ensure_alert(
            db, team, AlertKind.BUDGET_HARD,
            f"{team.name} reached its monthly budget (${status.spend:.4f} of "
            f"${status.monthly_budget:.2f}); further requests are blocked.",
            key=key, since=period, details={"usage_ratio": status.usage_ratio})
    if status.state == BudgetState.SOFT_ALERT:
        return await ensure_alert(
            db, team, AlertKind.BUDGET_SOFT,
            f"{team.name} has used {status.usage_ratio:.0%} of its monthly budget; routing "
            f"now uses the team's cost-saver policy if one is set.",
            key=key, since=period, details={"usage_ratio": status.usage_ratio})
    return None


async def forecast_alert(db: AsyncSession, team: Team, fc) -> Alert | None:
    """Warn when the forecast says the budget will probably be exceeded this month."""
    if fc.overrun_probability < 0.5 or fc.spent_to_date >= fc.budget:
        return None  # unlikely, or already over (the hard-block alert covers that)
    period = month_start()
    when = f" around {fc.overrun_date:%d %b}" if fc.overrun_date else ""
    return await ensure_alert(
        db, team, AlertKind.FORECAST_OVERRUN,
        f"{team.name} is forecast to exceed its ${fc.budget:.2f} budget{when}: projected "
        f"${fc.projected:.4f} (90% interval ${fc.lower:.4f}–${fc.upper:.4f}), "
        f"{fc.overrun_probability:.0%} probability.",
        key=period.date().isoformat(), since=period,
        details={"projected": round(fc.projected, 6), "overrun_probability":
                 round(fc.overrun_probability, 3),
                 "overrun_date": fc.overrun_date.isoformat() if fc.overrun_date else None})
