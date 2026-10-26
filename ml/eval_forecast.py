"""Benchmark the spend forecaster on many synthetic daily-spend series.

Each series has a random level, growth rate, weekly pattern strength and noise,
plus occasional spikes, 35–84 days long. For each, the last 7 days are held out and
predicted by:
  budgetopt   the backend forecaster (ETS with AICc model choice, see
              app/services/forecasting.py)
  naive_7     mean of the previous 7 days, repeated
  run_rate    mean of all history, repeated
Reports the median absolute % error of the 7-day total, daily MAE, and how often the
actual 7-day total falls inside BudgetOpt's 90% interval (calibration).

Usage: python ml/eval_forecast.py   -> ml/eval/forecast_report.md, forecast_metrics.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from app.services.forecasting import INTERVAL_SCALE, _fit_paths, calibrate  # noqa: E402

N_SERIES = 120
HORIZON = 7
SEED = 2026


def synthetic(rng: np.random.Generator) -> pd.Series:
    n = int(rng.integers(35, 85))
    t = np.arange(n + HORIZON)
    level = rng.uniform(0.2, 5.0)
    growth = rng.uniform(-0.01, 0.025)
    weekly = rng.uniform(0.0, 0.6)
    noise = rng.uniform(0.03, 0.25)
    dow = (t + int(rng.integers(7))) % 7
    pattern = np.where(dow < 5, 1.0, 1.0 - weekly)
    y = level * (1 + growth) ** t * pattern * (1 + rng.normal(0, noise, t.size))
    spikes = rng.random(t.size) < 0.02
    y[spikes] *= rng.uniform(2, 4, spikes.sum())
    return pd.Series(np.clip(y, 0, None),
                     index=pd.date_range("2026-01-01", periods=t.size, freq="D").date)


def required_scale(seed: int, n: int) -> float:
    """Interval widening needed for 90% coverage, measured on a separate calibration set."""
    rng = np.random.default_rng(seed)
    ratios = []
    for k in range(n):
        s = synthetic(rng)
        train, actual = s.iloc[:-HORIZON], s.iloc[-HORIZON:].sum()
        _, paths = _fit_paths(train, HORIZON, seed=k)
        totals = paths.sum(axis=0)
        mean = totals.mean()
        half = (np.percentile(totals, 95) - mean) if actual >= mean else (
            mean - np.percentile(totals, 5))
        ratios.append(abs(actual - mean) / half if half > 0 else 0.0)
    return float(np.percentile(ratios, 90))


def main() -> None:
    calib_scale = required_scale(SEED + 1, N_SERIES)
    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.perf_counter()
    for k in range(N_SERIES):
        s = synthetic(rng)
        train, test = s.iloc[:-HORIZON], s.iloc[-HORIZON:].to_numpy()
        method, paths = _fit_paths(train, HORIZON, seed=k)
        preds = {"budgetopt": paths.mean(axis=1),
                 "naive_7": np.full(HORIZON, train.iloc[-7:].mean()),
                 "run_rate": np.full(HORIZON, train.mean())}
        actual = test.sum()
        raw = paths.sum(axis=0)
        totals = calibrate(raw)
        row = {"method": method,
               "inside_90_raw": bool(np.percentile(raw, 5) <= actual <= np.percentile(raw, 95)),
               "inside_90": bool(np.percentile(totals, 5) <= actual
                                 <= np.percentile(totals, 95))}
        for name, p in preds.items():
            row[f"{name}_total_ape"] = abs(p.sum() - actual) / actual * 100
            row[f"{name}_daily_mae_rel"] = np.mean(np.abs(p - test)) / test.mean() * 100
        rows.append(row)
    secs = time.perf_counter() - t0
    df = pd.DataFrame(rows)

    summary = {name: {"median_total_ape_pct": round(float(df[f"{name}_total_ape"].median()), 2),
                      "mean_total_ape_pct": round(float(df[f"{name}_total_ape"].mean()), 2),
                      "median_daily_mae_pct_of_mean": round(
                          float(df[f"{name}_daily_mae_rel"].median()), 2)}
               for name in ("budgetopt", "naive_7", "run_rate")}
    wins = float((df["budgetopt_total_ape"] < df["naive_7_total_ape"]).mean())
    metrics = {"series": N_SERIES, "horizon_days": HORIZON, "methods_chosen":
               df["method"].value_counts().to_dict(), "summary": summary,
               "budgetopt_beats_naive_7_share": round(wins, 3),
               "interval_90_coverage_uncalibrated": round(float(df["inside_90_raw"].mean()), 3),
               "interval_90_coverage": round(float(df["inside_90"].mean()), 3),
               "interval_scale_in_use": INTERVAL_SCALE,
               "interval_scale_measured_on_calibration_set": round(calib_scale, 3),
               "seconds_per_series": round(secs / N_SERIES, 3)}
    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    (out / "forecast_metrics.json").write_text(json.dumps(metrics, indent=2))

    table = "\n".join(
        f"| {n} | {v['median_total_ape_pct']} | {v['mean_total_ape_pct']} | "
        f"{v['median_daily_mae_pct_of_mean']} |" for n, v in summary.items())
    (out / "forecast_report.md").write_text(f"""# Spend forecaster — evaluation

{N_SERIES} synthetic daily-spend series (random level, growth −1%…+2.5%/day, weekly pattern,
noise 3–25%, occasional 2–4× spikes; 35–84 days of history). The last {HORIZON} days are held out.

| Method | Median 7-day total error % | Mean 7-day total error % | Median daily MAE % of mean |
|---|---|---|---|
{table}

- BudgetOpt beats the naive 7-day average on **{wins:.0%}** of series.
- Interval calibration: the raw simulated 90% interval contained the actual 7-day total in
  only {metrics['interval_90_coverage_uncalibrated']:.0%} of series, because simulation ignores
  parameter uncertainty. Intervals are therefore widened around the mean by a factor of
  {INTERVAL_SCALE} (`INTERVAL_SCALE`, measured on a separate calibration set of {N_SERIES}
  series: {calib_scale:.2f}). On this test set the calibrated interval covers the actual
  total in **{metrics['interval_90_coverage']:.0%}** of series (target 90%).
- Models chosen by AICc: `{metrics['methods_chosen']}`
- Fit + 500-path simulation: {metrics['seconds_per_series']} s per series.
""")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
