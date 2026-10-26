"""Budget monitoring: forecasts, alerts and anomaly scans (report objective O6)."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_accessible_team, get_current_user, require_admin
from app.db.models import Alert, AlertKind, AuditEvent, Forecast, Role, Team, User
from app.db.session import get_db
from app.services.alerts import forecast_alert
from app.services.anomaly import scan_and_alert
from app.services.forecasting import forecast_team

router = APIRouter(tags=["monitoring"])


def _forecast_out(f: Forecast) -> dict:
    return {
        "id": f.id, "team_id": f.team_id, "method": f.method, "created_at": f.created_at,
        "period_start": f.period_start, "period_end": f.period_end,
        "projected_spend": f.projected_spend, "lower": f.lower, "upper": f.upper,
        "overrun_date": f.overrun_date, "overrun_probability": f.overrun_probability,
        "daily": f.daily, "backtest": f.backtest,
    }


@router.get("/teams/{team_id}/forecast")
async def team_forecast(
    team_id: int, refresh: bool = False,
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
) -> dict:
    """Latest month-end forecast; `refresh=true` recomputes it now."""
    team = await get_accessible_team(team_id, user, db, action="view_forecast")
    latest = await db.scalar(select(Forecast).where(Forecast.team_id == team_id)
                             .order_by(Forecast.id.desc()).limit(1))
    if refresh or latest is None:
        fc = await forecast_team(db, team)
        await forecast_alert(db, team, fc)
        latest = await db.scalar(select(Forecast).where(Forecast.team_id == team_id)
                                 .order_by(Forecast.id.desc()).limit(1))
    return {**_forecast_out(latest), "monthly_budget": team.monthly_budget}


@router.get("/alerts")
async def list_alerts(
    team_id: int | None = None, kind: AlertKind | None = None,
    acknowledged: bool | None = None, limit: int = Query(default=100, ge=1, le=500),
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db),
) -> list[dict]:
    q = (select(Alert).join(Team, Team.id == Alert.team_id)
         .where(Team.organization_id == user.organization_id)
         .order_by(Alert.id.desc()).limit(limit))
    if team_id is not None:
        await get_accessible_team(team_id, user, db, action="list_alerts")
        q = q.where(Alert.team_id == team_id)
    elif user.role != Role.ORG_ADMIN:
        q = q.where(Alert.team_id == user.team_id)
    if user.role == Role.MEMBER:  # members don't see other users' anomaly alerts
        q = q.where((Alert.kind != AlertKind.ANOMALY) | (Alert.user_id == user.id))
    if kind is not None:
        q = q.where(Alert.kind == kind)
    if acknowledged is not None:
        q = q.where(Alert.acknowledged.is_(acknowledged))
    return [{"id": a.id, "team_id": a.team_id, "user_id": a.user_id, "kind": a.kind.value,
             "message": a.message, "details": a.details, "acknowledged": a.acknowledged,
             "created_at": a.created_at} for a in await db.scalars(q)]


@router.post("/alerts/{alert_id}/ack")
async def acknowledge(
    alert_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    alert = await db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Alert not found")
    await get_accessible_team(alert.team_id, user, db, action="ack_alert")
    if user.role == Role.MEMBER:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Only managers and admins acknowledge")
    alert.acknowledged = True
    db.add(AuditEvent(user_id=user.id, action="ack_alert", resource="alert",
                      resource_id=str(alert_id)))
    await db.commit()
    return {"id": alert_id, "acknowledged": True}


@router.post("/monitoring/forecasts/refresh")
async def refresh_forecasts(
    user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> dict:
    teams = list(await db.scalars(select(Team).where(
        Team.organization_id == user.organization_id)))
    out = []
    for team in teams:
        fc = await forecast_team(db, team)
        await forecast_alert(db, team, fc)
        out.append({"team_id": team.id, "projected": round(fc.projected, 6),
                    "overrun_probability": round(fc.overrun_probability, 3), "method": fc.method})
    return {"forecasts": out}


@router.post("/monitoring/anomalies/scan")
async def scan(user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)) -> dict:
    found = await scan_and_alert(db)
    return {"anomalies": [{"team_id": a.team_id, "subject": a.subject, "metric": a.metric,
                           "value": a.value, "median": a.median, "z": a.z,
                           "hour": a.hour.isoformat()} for a in found]}
