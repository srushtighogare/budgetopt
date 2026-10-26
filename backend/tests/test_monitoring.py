"""Forecasting, budget/forecast alerts, anomaly detection and the monitoring API."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from app.db.models import Alert, AlertKind, ApiKey, Request, RequestStatus, Role, Team, User
from app.services.alerts import budget_alerts, forecast_alert
from app.services.anomaly import detect, robust_z
from app.services.budget import budget_status
from app.services.forecasting import forecast_from_history, forecast_team
from tests.conftest import login, make_user

TODAY = date(2026, 10, 20)


def history(days: int, base=0.2, slope=0.0, weekly=0.0, noise=0.01, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = [TODAY - timedelta(days=days - i) for i in range(days)]
    vals = [base + slope * i + (weekly if d.weekday() < 5 else 0) + rng.normal(0, noise)
            for i, d in enumerate(idx)]
    return pd.Series(np.clip(vals, 0, None), index=idx)


# --- forecaster ---

@pytest.mark.parametrize(("days", "method_prefix"), [
    (45, "ets_"), (12, "ets_"), (3, "run_rate"), (0, "run_rate")])
def test_method_depends_on_history(days, method_prefix):
    fc = forecast_from_history(history(days), today=TODAY, spent_today=0, spent_to_date=1.0,
                               budget=100)
    assert fc.method.startswith(method_prefix)
    assert fc.lower <= fc.projected <= fc.upper
    if days >= 21:
        assert fc.method.endswith("_weekly")


def test_projection_tracks_known_run_rate():
    # ~0.5/day for 12 remaining days (20..31 Oct) on top of 10 already spent
    fc = forecast_from_history(history(45, base=0.5, noise=0.02), today=TODAY, spent_today=0,
                               spent_to_date=10.0, budget=100)
    assert fc.projected == pytest.approx(10 + 12 * 0.5, rel=0.05)
    assert fc.overrun_probability == 0 and fc.overrun_date is None


def test_overrun_date_and_probability():
    fc = forecast_from_history(history(45, base=1.0, noise=0.02), today=TODAY, spent_today=0,
                               spent_to_date=10.0, budget=15.0)
    # 10 + 1/day crosses 15 on the 5th forecast day (20 Oct is day 1)
    assert fc.overrun_date == TODAY + timedelta(days=4)
    assert fc.overrun_probability > 0.95
    already = forecast_from_history(history(45), today=TODAY, spent_today=0,
                                    spent_to_date=20.0, budget=15.0)
    assert already.overrun_date == TODAY and already.overrun_probability == 1.0


def test_partial_today_is_not_double_counted():
    h = history(45, base=1.0, noise=0.0001)
    none = forecast_from_history(h, today=TODAY, spent_today=0.0, spent_to_date=5.0,
                                 budget=1000)
    half = forecast_from_history(h, today=TODAY, spent_today=0.5, spent_to_date=5.5,
                                 budget=1000)
    assert half.projected == pytest.approx(none.projected, abs=0.02)


def test_backtest_reported_with_enough_history():
    fc = forecast_from_history(history(45, weekly=0.3), today=TODAY, spent_today=0,
                               spent_to_date=1, budget=100)
    assert fc.backtest and {"forecast_total_error_pct", "naive_total_error_pct"} <= set(
        fc.backtest)
    assert forecast_from_history(history(10), today=TODAY, spent_today=0, spent_to_date=1,
                                 budget=100).backtest is None


async def _add_spend(db, team, when: datetime, cost: str, **kw):
    db.add(Request(team_id=team.id, prompt_hash="h" * 64, cost=Decimal(cost),
                   status=RequestStatus.SUCCESS, created_at=when, **kw))


async def test_forecast_team_reads_db_and_stores(db):
    team = await db.scalar(select(Team).where(Team.name == "Support"))  # budget 2.00
    now = datetime.now(UTC)
    for d in range(1, 30):
        await _add_spend(db, team, now - timedelta(days=d), "0.05")
    await db.commit()
    fc = await forecast_team(db, team, now=now)
    assert fc.history_days == 29 and fc.method.startswith("ets")
    stored = await db.scalar(select(func.count()).select_from(
        select(Request.id).where(Request.team_id == team.id).subquery()))
    assert stored == 29
    from app.db.models import Forecast
    row = await db.scalar(select(Forecast).where(Forecast.team_id == team.id))
    assert row is not None and row.daily and row.overrun_probability is not None


# --- alerts ---

async def test_budget_alerts_fire_once_per_month(db):
    team = await db.scalar(select(Team).where(Team.name == "Marketing"))  # budget 0.50
    await _add_spend(db, team, datetime.now(UTC), "0.42")
    await db.commit()
    s = await budget_status(db, team)
    assert (await budget_alerts(db, team, s)).kind == AlertKind.BUDGET_SOFT
    assert await budget_alerts(db, team, s) is None  # de-duplicated
    await _add_spend(db, team, datetime.now(UTC), "0.10")
    await db.commit()
    hard = await budget_alerts(db, team, await budget_status(db, team))
    assert hard.kind == AlertKind.BUDGET_HARD


async def test_forecast_alert_only_when_overrun_likely(db):
    team = await db.scalar(select(Team).where(Team.name == "Support"))
    likely = forecast_from_history(history(45, base=1.0), today=TODAY, spent_today=0,
                                   spent_to_date=1.0, budget=2.0)
    assert (await forecast_alert(db, team, likely)).kind == AlertKind.FORECAST_OVERRUN
    assert await forecast_alert(db, team, likely) is None
    other = await db.scalar(select(Team).where(Team.name == "Engineering"))
    safe = forecast_from_history(history(45, base=0.01), today=TODAY, spent_today=0,
                                 spent_to_date=0.1, budget=5.0)
    assert await forecast_alert(db, other, safe) is None


# --- anomalies ---

def test_robust_z():
    z, med = robust_z(np.array([10, 11, 9, 10, 12, 10, 9.0]), 40, floor=1)
    assert med == 10 and z > 10
    z, _ = robust_z(np.zeros(21), 5, floor=20)  # flat history: scale falls back to floor
    assert z == pytest.approx(0.25)


async def _key(db, team) -> ApiKey:
    user = await db.scalar(select(User).where(User.role == Role.ORG_ADMIN))
    k = ApiKey(team_id=team.id, created_by=user.id, name="k", prefix="bo_x",
               key_hash=f"{team.id}{'0' * 63}"[:64])
    db.add(k)
    await db.flush()
    return k


async def test_spike_is_detected_but_daily_pattern_is_not(db):
    team = await db.scalar(select(Team).where(Team.name == "Engineering"))
    steady, spiky = await _key(db, team), None
    k2 = ApiKey(team_id=team.id, created_by=steady.created_by, name="k2", prefix="bo_y",
                key_hash="f" * 64)
    db.add(k2)
    await db.flush()
    spiky = k2
    now = datetime(2026, 10, 20, 15, 10, tzinfo=UTC)
    last = now.replace(minute=0) - timedelta(hours=1)  # 14:00, the hour being checked
    for d in range(0, 8):
        for h in range(24):
            ts = last - timedelta(days=d) + timedelta(hours=h - 14)
            if ts > last + timedelta(minutes=59):
                continue
            busy = 9 <= ts.hour < 18
            # steady key: 30 req/h in working hours, 1 at night, every day incl. today
            for _ in range(30 if busy else 1):
                await _add_spend(db, team, ts + timedelta(minutes=5), "0.0001",
                                 api_key_id=steady.id)
            if ts < last:  # spiky key: 2 req/h normally ...
                for _ in range(2):
                    await _add_spend(db, team, ts + timedelta(minutes=5), "0.0001",
                                     api_key_id=spiky.id)
    for _ in range(200):  # ... then 200 in the checked hour
        await _add_spend(db, team, last + timedelta(minutes=10), "0.0001", api_key_id=spiky.id)
    await db.commit()

    found = await detect(db, now)
    subjects = {(a.subject, a.metric) for a in found}
    assert (f"key:{spiky.id}", "requests") in subjects
    assert not any(a.subject == f"key:{steady.id}" for a in found)


# --- API ---

async def test_forecast_and_alert_endpoints(client, admin_headers, db):
    team = await db.scalar(select(Team).where(Team.name == "Marketing"))
    now = datetime.now(UTC)
    for d in range(1, 25):  # varying daily spend (a constant series uses the run-rate)
        await _add_spend(db, team, now - timedelta(days=d), f"{0.02 + 0.01 * (d % 3):.2f}")
    await db.commit()

    r = await client.get(f"/api/v1/teams/{team.id}/forecast?refresh=true",
                         headers=admin_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["method"].startswith("ets") and body["daily"]
    assert Decimal(body["lower"]) <= Decimal(body["projected_spend"]) <= Decimal(body["upper"])

    await make_user(db, "lead@demo.io", Role.TEAM_MANAGER, "Marketing")
    await make_user(db, "eng@demo.io", Role.MEMBER, "Engineering")
    lead, eng = await login(client, "lead@demo.io"), await login(client, "eng@demo.io")
    denied = await client.get(f"/api/v1/teams/{team.id}/forecast", headers=eng)
    assert denied.status_code == 403  # the report's cross-team example, again

    # Marketing is already over its 0.50 budget -> the hard-block alert (the forecast alert
    # deliberately stays quiet once a team is over budget)
    await budget_alerts(db, team, await budget_status(db, team))
    alerts = (await client.get("/api/v1/alerts", headers=lead)).json()
    assert alerts and all(a["team_id"] == team.id for a in alerts)
    assert (await client.get("/api/v1/alerts", headers=eng)).json() == []
    ack = await client.post(f"/api/v1/alerts/{alerts[0]['id']}/ack", headers=lead)
    assert ack.json()["acknowledged"] is True
    assert (await client.post(f"/api/v1/alerts/{alerts[0]['id']}/ack",
                              headers=eng)).status_code == 403

    scan = await client.post("/api/v1/monitoring/anomalies/scan", headers=admin_headers)
    assert scan.status_code == 200 and "anomalies" in scan.json()
    refresh = await client.post("/api/v1/monitoring/forecasts/refresh", headers=admin_headers)
    assert len(refresh.json()["forecasts"]) == 3
    assert await db.scalar(select(func.count()).select_from(Alert)) >= 1
