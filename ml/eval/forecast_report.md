# Spend forecaster — evaluation

120 synthetic daily-spend series (random level, growth −1%…+2.5%/day, weekly pattern,
noise 3–25%, occasional 2–4× spikes; 35–84 days of history). The last 7 days are held out.

| Method | Median 7-day total error % | Mean 7-day total error % | Median daily MAE % of mean |
|---|---|---|---|
| budgetopt | 11.0 | 14.39 | 17.74 |
| naive_7 | 10.82 | 14.68 | 24.1 |
| run_rate | 26.57 | 28.96 | 33.73 |

- BudgetOpt beats the naive 7-day average on **59%** of series.
- Interval calibration: the raw simulated 90% interval contained the actual 7-day total in
  only 63% of series, because simulation ignores
  parameter uncertainty. Intervals are therefore widened around the mean by a factor of
  1.63 (`INTERVAL_SCALE`, measured on a separate calibration set of 120
  series: 1.63). On this test set the calibrated interval covers the actual
  total in **82%** of series (target 90%).
- Models chosen by AICc: `{'ets_level_weekly': 93, 'ets_damped_trend_weekly': 27}`
- Fit + 500-path simulation: 0.02 s per series.
