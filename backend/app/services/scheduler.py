"""In-process periodic jobs: spend forecasts and anomaly scans.

A deliberately small scheduler (asyncio tasks started in the API lifespan)
instead of a separate worker service: the jobs are light, and every job is also
available as a CLI command and an admin endpoint. With several API replicas,
disable it on all but one (ENABLE_SCHEDULER=false).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import Team
from app.db.session import SessionLocal
from app.services.alerts import forecast_alert
from app.services.anomaly import scan_and_alert
from app.services.forecasting import forecast_team

log = logging.getLogger(__name__)


async def refresh_all_forecasts() -> int:
    async with SessionLocal() as db:
        teams = list(await db.scalars(select(Team)))
        for team in teams:
            fc = await forecast_team(db, team)
            await forecast_alert(db, team, fc)
    return len(teams)


async def scan_anomalies() -> int:
    async with SessionLocal() as db:
        return len(await scan_and_alert(db))


async def _every(seconds: int, job: Callable[[], Awaitable[int]]) -> None:
    while True:
        try:
            n = await job()
            log.info("%s: %s", job.__name__, n)
        except Exception:
            log.exception("scheduled job %s failed", job.__name__)
        await asyncio.sleep(seconds)


def start() -> list[asyncio.Task]:
    s = get_settings()
    if not s.enable_scheduler:
        return []
    return [asyncio.create_task(_every(s.forecast_interval_s, refresh_all_forecasts)),
            asyncio.create_task(_every(s.anomaly_interval_s, scan_anomalies))]
