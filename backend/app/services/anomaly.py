"""Abnormal-usage detection (report, Safety: "monitoring for abnormal or excessive AI usage").

For every caller (user or API key) the last complete hour of usage is compared
with that caller's own history at the same time of day (the same hour ±1 on each
of the previous 7 days, 21 samples), so normal working-hours peaks aren't flagged,
using a robust z-score:  z = (x - median) / (1.4826 * MAD). Median/MAD are not
dragged up by earlier spikes the way mean/std are. Hours with no activity count as
zero, so a normally idle key that suddenly bursts is caught.

To avoid alerting on noise, an hour is flagged only if z >= Z_THRESHOLD *and* it
also clears an absolute floor (MIN_REQUESTS requests or MIN_COST spend). When a
caller's history has no spread at all (MAD = 0), the scale falls back to
max(mean, floor) so a constant pattern doesn't make every small change "infinite".
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AlertKind, Request, Team
from app.services.alerts import ensure_alert

Z_THRESHOLD = 3.5
MIN_REQUESTS = 20
MIN_COST = 0.01
LOOKBACK_HOURS = 7 * 24


@dataclass(frozen=True)
class Anomaly:
    team_id: int
    subject: str  # "user:12" | "key:5"
    user_id: int | None
    hour: datetime
    metric: str  # requests | cost
    value: float
    median: float
    z: float


def robust_z(history: np.ndarray, x: float, floor: float) -> tuple[float, float]:
    med = float(np.median(history)) if history.size else 0.0
    mad = float(np.median(np.abs(history - med))) if history.size else 0.0
    scale = 1.4826 * mad
    if scale <= 0:
        scale = max(float(history.mean()) if history.size else 0.0, floor)
    return (x - med) / scale, med


async def detect(db: AsyncSession, now: datetime | None = None) -> list[Anomaly]:
    now = now or datetime.now(UTC)
    current = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    since = current - timedelta(hours=LOOKBACK_HOURS)
    hour = func.date_trunc("hour", func.timezone("UTC", Request.created_at))
    rows = (await db.execute(
        select(Request.team_id, Request.api_key_id, Request.user_id, hour, func.count(),
               func.sum(Request.cost))
        .where(Request.created_at >= since, Request.created_at < current + timedelta(hours=1),
               (Request.api_key_id.is_not(None)) | (Request.user_id.is_not(None)))
        .group_by(Request.team_id, Request.api_key_id, Request.user_id, hour)
    )).all()

    n_hours = LOOKBACK_HOURS + 1
    series: dict[tuple[int, str, int | None], np.ndarray] = {}
    for team_id, key_id, user_id, h, cnt, cost in rows:
        label = f"key:{key_id}" if key_id is not None else f"user:{user_id}"
        k = (team_id, label, user_id if key_id is None else None)
        arr = series.setdefault(k, np.zeros((n_hours, 2)))
        idx = int(h.replace(tzinfo=UTC).timestamp() - since.timestamp()) // 3600
        if 0 <= idx < n_hours:
            arr[idx] += (cnt, float(cost or 0))

    # Same hour of day (±1) on each previous day: offsets 24k-1, 24k, 24k+1 hours back
    same_time = sorted({LOOKBACK_HOURS - 24 * d + o for d in range(1, 8) for o in (-1, 0, 1)
                        if 0 <= LOOKBACK_HOURS - 24 * d + o < LOOKBACK_HOURS})
    found = []
    for (team_id, label, user_id), arr in series.items():
        hist, last = arr[same_time], arr[-1]
        for col, metric, floor in ((0, "requests", MIN_REQUESTS), (1, "cost", MIN_COST)):
            z, med = robust_z(hist[:, col], float(last[col]), floor)
            if z >= Z_THRESHOLD and last[col] >= floor:
                found.append(Anomaly(team_id, label, user_id, current, metric,
                                     float(last[col]), med, round(float(z), 2)))
    return found


async def scan_and_alert(db: AsyncSession, now: datetime | None = None) -> list[Anomaly]:
    anomalies = await detect(db, now)
    for a in anomalies:
        team = await db.get(Team, a.team_id)
        if a.metric == "requests":
            value, typical = f"{a.value:.0f} requests", f"{a.median:.0f}"
        else:
            value, typical = f"${a.value:.4f} spend", f"${a.median:.4f}"
        await ensure_alert(
            db, team, AlertKind.ANOMALY,
            f"Unusual usage by {a.subject} in {team.name}: {value} in the hour from "
            f"{a.hour:%d %b %H:00} UTC vs a typical {typical} per hour (robust z = {a.z}).",
            key=f"{a.subject}:{a.metric}:{a.hour.isoformat()}", since=a.hour,
            user_id=a.user_id,
            details={"subject": a.subject, "metric": a.metric, "value": a.value,
                     "median": a.median, "z": a.z, "hour": a.hour.isoformat()})
    return anomalies
