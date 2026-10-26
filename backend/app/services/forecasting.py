"""Month-end spend forecasting per team (report objective O6, Budget & Forecasting module).

Daily spend is modelled with exponential smoothing (statsmodels ETS):
  >= 21 days of history   additive error, weekly seasonality, with or without a damped
                          trend (whichever has the lower AICc)
  7–20 days               additive error, damped trend vs no trend (lower AICc)
  1–6 days                run-rate (mean daily spend), bootstrap interval
The month-end total and its 90% interval come from 500 simulated future paths
(not just the point forecast), which also gives the probability of overrunning
the budget. Overrun date = first day the mean cumulative path crosses the budget.

A 7-day backtest (refit without the last week) is reported alongside, against a
naive "average of the previous 7 days" baseline, so the forecast quality is
visible rather than assumed.
"""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import Date, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Forecast, Request, Team
from app.services.budget import month_start

log = logging.getLogger(__name__)

SIMULATIONS = 500
# Simulated paths understate uncertainty (they ignore parameter/model uncertainty), so
# intervals are widened around the mean by this factor. Measured by ml/eval_forecast.py
# on a calibration set of synthetic series as the widening needed for 90% coverage.
INTERVAL_SCALE = 1.63
HISTORY_DAYS = 90
SEASONAL_MIN_DAYS = 21
TREND_MIN_DAYS = 7
BACKTEST_DAYS = 7


@dataclass
class SpendForecast:
    method: str
    period_start: date
    period_end: date
    spent_to_date: float
    projected: float
    lower: float
    upper: float
    budget: float
    overrun_probability: float
    overrun_date: date | None
    history_days: int
    daily: list[dict] = field(default_factory=list)
    backtest: dict | None = None


def _month_end(d: date) -> date:
    nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return nxt - timedelta(days=1)


def _fit_paths(y: pd.Series, horizon: int, seed: int) -> tuple[str, np.ndarray]:
    """Simulated daily-spend paths, shape (horizon, SIMULATIONS), clipped at zero."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n:  # statsmodels needs a regular DatetimeIndex
        y = pd.Series(y.to_numpy(dtype=float),
                      index=pd.date_range(pd.Timestamp(y.index[0]), periods=n, freq="D"))
    if n >= TREND_MIN_DAYS and y.std() > 0:
        from statsmodels.tsa.exponential_smoothing.ets import ETSModel

        seasonal = n >= SEASONAL_MIN_DAYS
        fits = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for trend in (True, False):
                try:
                    fit = ETSModel(y, error="add", trend="add" if trend else None,
                                   damped_trend=trend, seasonal="add" if seasonal else None,
                                   seasonal_periods=7 if seasonal else None).fit(disp=False)
                    if np.isfinite(fit.aicc):
                        fits.append((fit.aicc, trend, fit))
                except Exception as e:  # numerical failures on tiny/odd series
                    log.info("ETS fit failed (trend=%s): %s", trend, e)
            if fits:
                _, trend, fit = min(fits, key=lambda f: f[0])
                # Resample the model's own residuals (not Gaussian noise): spend has spikes,
                # and bootstrapped errors keep those heavy tails in the interval
                sims = fit.simulate(horizon, repetitions=SIMULATIONS, anchor="end",
                                    random_errors="bootstrap",
                                    rng=np.random.default_rng(seed))
                paths = np.clip(np.asarray(sims, dtype=float).reshape(horizon, SIMULATIONS),
                                0, None)
                if np.isfinite(paths).all():
                    name = "ets" + ("_damped_trend" if trend else "_level")
                    return name + ("_weekly" if seasonal else ""), paths
    # Run-rate: resample observed days (bootstrap) for each future day
    values = y.to_numpy() if n else np.zeros(1)
    return "run_rate", rng.choice(values, size=(horizon, SIMULATIONS), replace=True)


def calibrate(values: np.ndarray) -> np.ndarray:
    """Widen a sample around its mean by INTERVAL_SCALE (keeps the mean unchanged)."""
    mean = values.mean()
    return np.clip(mean + INTERVAL_SCALE * (values - mean), 0, None)


def _backtest(y: pd.Series, seed: int) -> dict | None:
    if len(y) < 4 * BACKTEST_DAYS:
        return None
    train, test = y.iloc[:-BACKTEST_DAYS], y.iloc[-BACKTEST_DAYS:]
    method, paths = _fit_paths(train, BACKTEST_DAYS, seed)
    pred = paths.mean(axis=1)
    naive = np.full(BACKTEST_DAYS, train.iloc[-BACKTEST_DAYS:].mean())
    actual_total = float(test.sum())

    def pct(err: float) -> float | None:
        return round(float(100 * err / actual_total), 2) if actual_total > 0 else None

    return {
        "method": method, "days": BACKTEST_DAYS, "actual_total": round(actual_total, 6),
        "forecast_total": round(float(pred.sum()), 6),
        "forecast_total_error_pct": pct(abs(pred.sum() - actual_total)),
        "naive_total_error_pct": pct(abs(naive.sum() - actual_total)),
        "forecast_daily_mae": round(float(np.mean(np.abs(pred - test.to_numpy()))), 6),
        "naive_daily_mae": round(float(np.mean(np.abs(naive - test.to_numpy()))), 6),
        "actual_within_90pct_interval": bool(
            np.percentile(calibrate(paths.sum(axis=0)), 5) <= actual_total
            <= np.percentile(calibrate(paths.sum(axis=0)), 95)),
    }


def forecast_from_history(history: pd.Series, *, today: date, spent_today: float,
                          spent_to_date: float, budget: float, seed: int = 0) -> SpendForecast:
    """`history` = complete days before `today` (index: dates, values: USD)."""
    start, end = today.replace(day=1), _month_end(today)
    horizon = (end - today).days + 1  # today .. end of month
    method, paths = _fit_paths(history, horizon, seed)
    # Today is partly over: only the not-yet-spent part of today's forecast is future spend
    paths = paths.copy()
    paths[0] = np.clip(paths[0] - spent_today, 0, None)
    totals = spent_to_date + calibrate(paths.sum(axis=0))
    mean_cum = spent_to_date + np.cumsum(paths.mean(axis=1))

    overrun_date = None
    if spent_to_date >= budget:
        overrun_date = today
    else:
        crossed = np.nonzero(mean_cum >= budget)[0]
        if crossed.size:
            overrun_date = today + timedelta(days=int(crossed[0]))

    daily = [{"date": d.isoformat(), "actual": round(float(v), 6)}
             for d, v in history.items() if d >= start]
    for k in range(horizon):
        daily.append({
            "date": (today + timedelta(days=k)).isoformat(),
            "actual": round(spent_today, 6) if k == 0 else None,
            "mean": round(float(paths[k].mean()) + (spent_today if k == 0 else 0), 6),
            "lower": round(float(np.percentile(paths[k], 5)), 6),
            "upper": round(float(np.percentile(paths[k], 95)), 6),
            "cumulative_mean": round(float(mean_cum[k]), 6),
        })
    return SpendForecast(
        method=method, period_start=start, period_end=end, spent_to_date=spent_to_date,
        projected=float(totals.mean()), lower=float(np.percentile(totals, 5)),
        upper=float(np.percentile(totals, 95)), budget=budget,
        overrun_probability=float((totals >= budget).mean()), overrun_date=overrun_date,
        history_days=len(history), daily=daily, backtest=_backtest(history, seed),
    )


async def daily_spend(db: AsyncSession, team_id: int, since: date, until: date) -> pd.Series:
    """Spend per UTC day in [since, until), zero-filled."""
    day = cast(func.timezone("UTC", Request.created_at), Date)
    rows = (await db.execute(
        select(day, func.sum(Request.cost)).where(
            Request.team_id == team_id, day >= since, day < until).group_by(day)
    )).all()
    idx = pd.date_range(since, until - timedelta(days=1), freq="D").date
    s = pd.Series(0.0, index=idx)
    for d, v in rows:
        s[d] = float(v)
    return s


async def forecast_team(db: AsyncSession, team: Team, now: datetime | None = None,
                        store: bool = True) -> SpendForecast:
    now = now or datetime.now(UTC)
    today = now.date()
    full = await daily_spend(db, team.id, today - timedelta(days=HISTORY_DAYS),
                             today + timedelta(days=1))
    # Drop leading days before the team's first spend so ETS doesn't fit an empty prefix
    nz = np.nonzero(full.to_numpy())[0]
    first = full.index[nz[0]] if nz.size else today
    history = full[(full.index >= first) & (full.index < today)]
    spent_today = float(full.get(today, 0.0))
    period_start = month_start(now).date()
    spent_to_date = float(full[full.index >= period_start].sum())

    fc = forecast_from_history(history, today=today, spent_today=spent_today,
                               spent_to_date=spent_to_date, budget=float(team.monthly_budget),
                               seed=team.id)
    if store:
        db.add(Forecast(
            team_id=team.id, method=fc.method, period_start=fc.period_start,
            period_end=fc.period_end, projected_spend=round(fc.projected, 8),
            lower=round(fc.lower, 8), upper=round(fc.upper, 8),
            overrun_date=fc.overrun_date, overrun_probability=round(fc.overrun_probability, 4),
            daily=fc.daily, backtest=fc.backtest,
        ))
        await db.commit()
    return fc
