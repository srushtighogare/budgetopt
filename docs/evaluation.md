# BudgetOpt — evaluation

Generated 2026-10-26 by `ml/build_report.py` from the metric files in `ml/eval/`.
All LLM-based labelling, profiling and benchmarking used free-tier API quotas; every label
records which model produced it.

## 1. Workload analysis: task classifier

Task-type classifier: TF-IDF word and character n-grams with logistic regression
(selected by validation macro-F1 among logistic regression, calibrated linear SVM and
complement naive Bayes). Training data: 3275 prompts from six public datasets, all
labelled by one LLM annotator with one rubric.

| Evaluation | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Held-out split of the annotated set (n=412) | 0.777 | 0.775 | same annotator as training |
| Out-of-distribution prompts (n=320) | 0.622 | 0.596 | labelled by a different LLM; top-2 accuracy 0.797 |
| Agreement between two LLM annotators on the same OOD prompts | 0.719 | | practical ceiling for any classifier |

Inference: **0.46 ms per prompt** on one CPU core.

**Finding.** A first version trained on each dataset's own category labels scored 0.86 on an
in-distribution test but only 0.44 on independently labelled prompts: it had learned each
dataset's writing style (e.g. CodeAlpaca vs Alpaca) rather than the task. Relabelling a mixed
pool with a single annotator removed the shortcut (0.44 → 0.62 on the
OOD set, against an inter-annotator ceiling of 0.72).
Annotator agreement with the source datasets' own labels: `{'codealpaca': 0.846, 'dolly': 0.584, 'gsm8k': 1.0, 'math': 0.987, 'mbpp': 1.0}`.

Details: [`ml/eval/classifier_report.md`](../ml/eval/classifier_report.md).

## 2. Workload analysis: complexity and output tokens

Complexity model and output-token estimator: the labelling run that produces their training data (cheap-model answer + strong reference + judge per prompt) is still collecting on free-tier quotas; until then the router uses the transparent fallbacks (a feature-based complexity heuristic and per-task median output lengths), and each routing explanation says so.

_Not produced yet. Run `make train-workload` (complexity model and token estimator)._

## 3. Model profiling

Each model answers a fixed set of 64 prompts (8 per task type). Math answers are checked
against gold answers; other answers are graded 1–10 by a judge model that never grades its own
answers. Every graded answer is stored (`profiling_results`) and aggregated per model and task
(`model_performance`), which the optimizer and the dashboard's quality heatmap use. Latency
p50/p95 per model and task comes from live traffic plus profiling calls (synthetic demo traffic
is excluded). With 4–8 prompts per cell the per-task scores are indicative, not precise.

## 4. Optimizer

NSGA-II (pymoo) over monthly cost, quality and p95 latency, with traffic shares per
workload class as decision variables and hard per-band quality floors, compared with the exact
LP optimum of the same problem on 20 random instances (60 budget checks).

| Metric | Value |
|---|---|
| Quality gap to the exact optimum (median / p90 / worst) | 0.832% / 2.486% / 5.505% |
| NSGA-II runtime, median | 1.5 s for the whole front (~98 allocations) |
| Exact LP / ILP solve time, median | 1.0 ms / 3.0 ms |

The gap is largest at tight budgets (median 1.578% at 10% of the cost
range) and near zero at generous ones. Details: [`ml/eval/optimizer_report.md`](../ml/eval/optimizer_report.md).

## 5. End-to-end routing benchmark

_Not produced yet. Run `python -I ml/benchmark_collect.py && python -I ml/benchmark_eval.py` (routing benchmark on held-out prompts)._

## 6. Spend forecasting

Exponential smoothing (statsmodels ETS; weekly seasonality, damped trend vs level chosen by
AICc; run-rate for short histories), 500 simulated paths with bootstrapped residuals for the
month-end total, overrun probability and date. Benchmark: 120 synthetic daily-spend
series, last 7 days held out.

| Method | Median error, 7-day total | Median daily error (% of mean) |
|---|---|---|
| budgetopt | 11.0% | 17.74% |
| naive 7 | 10.82% | 24.1% |
| run rate | 26.57% | 33.73% |

- BudgetOpt beats the naive 7-day mean on 59% of series.
- 90% interval coverage: 63% raw → **82%**
  after widening by a factor (1.63) measured on a separate calibration set.
- 0.02 s per forecast.

Every production forecast also carries its own 7-day backtest. Details: [`ml/eval/forecast_report.md`](../ml/eval/forecast_report.md).

## 7. Semantic cache

Local embeddings (BAAI/bge-small-en-v1.5, ~2 ms) with a pgvector HNSW index, evaluated on 100
curated pairs: 50 rewordings that should hit the cache and 50 near-misses that must not
(different entity, opposite direction, other audience, related question).

Embeddings alone put opposites as close as rewordings ("is coffee good / bad for your health" =
0.947), so the only threshold with no wrong answers (0.95) served just 38% of rewordings.
The deployed design accepts at ≥ 0.97 and asks a cheap verifier model in the 0.85–0.97 band:

Accept at similarity >= 0.97; ask the verifier (`gemini-3.5-flash-lite`) when 0.85 <= similarity <
0.97; reject below 0.85.

- Pairs needing the verifier: 52 of 100
- Verifier said YES to 42/42 grey-zone rewordings
  and to 0/10 grey-zone near-misses
- **End-to-end: 48 correct hits, 0 wrong hits; precision 1.000,
  recall 96%** (threshold-only at 0.95: precision 1.000, recall
  38%)
- Wrong hits: none

Further guards: identical context and numbers, cache only task types where similar questions
share answers, off for confidential teams. Details: [`ml/eval/semantic_cache_report.md`](../ml/eval/semantic_cache_report.md).

## 8. Routing overhead

BudgetOpt's own latency per request, measured with the provider call replaced by an instant
answer (real PostgreSQL, Redis, classifier, routing, logging). Design target: p95 < 50 ms.

| Mode | Scenario | p50 ms | p95 ms | Throughput |
|---|---|---|---|---|
| without semantic cache | cache miss, sequential | 9.07 | 11.93 |  |
| without semantic cache | cache miss, 20 concurrent | 81.16 | 131.56 | 226.3 req/s |
| without semantic cache | exact cache hit, sequential | 6.13 | 8.82 |  |
| with semantic cache | cache miss, sequential | 11.69 | 24.54 |  |
| with semantic cache | cache miss, 20 concurrent | 124.07 | 181.56 | 149.3 req/s |
| with semantic cache | exact cache hit, sequential | 6.16 | 7.78 |  |

Concurrent latency includes queueing on one worker; throughput is the capacity figure (service
time ≈ 1000 / throughput ms). Details: [`ml/eval/overhead_report.md`](../ml/eval/overhead_report.md).

## 9. Automated tests

- Backend: pytest suite against real PostgreSQL and Redis covering auth and RBAC (including the
  report's "employee views another team's spend" case), the gateway's OpenAI compatibility,
  budget gate, caches, rate limits, fallback and circuit breaker, workload analysis, profiling,
  optimizer (non-dominance, exactness vs LP/ILP, budget feasibility), policies, forecasting,
  alerts, anomaly detection, what-if and the savings report.
- Frontend: Vitest tests for the API client's token refresh, formatting and components.
- CI (GitHub Actions) runs lint, migrations check, backend tests and the frontend type-check,
  tests and build on every push.

## Limitations

- **Sample sizes.** Free-tier quotas limited the labelling run, the profiling set (4–8 prompts
  per model and task) and the benchmark. Per-task numbers are indicative; the overall
  comparisons are the main result.
- **LLM-as-judge.** Non-math quality is graded by an LLM. Judges never grade their own answers
  and the benchmark uses judges outside the evaluated models, but judge bias remains possible;
  math is checked against gold answers.
- **Prices** for preview models are approximate list prices and are configurable
  (`POST /api/v1/models/{id}/pricing`).
- **Forecast intervals** are calibrated on synthetic spend series; real-traffic calibration
  needs months of history.
- **Synthetic demo traffic** (`app/demo_traffic.py`) is used only to demonstrate forecasting,
  anomaly detection and reports; it is kept in a separate team and excluded from latency
  profiles.

## Reproducing

```bash
make up                                   # PostgreSQL + Redis
make data && make dataset                 # public datasets and splits
make label                                # free-tier labelling (resumable)
make ml                                   # classifier, OOD check, complexity/token models
backend/.venv/bin/python -I ml/eval_optimizer.py
backend/.venv/bin/python -I ml/eval_forecast.py
backend/.venv/bin/python -I ml/eval_semantic_cache.py
backend/.venv/bin/python -I ml/benchmark_collect.py && backend/.venv/bin/python -I ml/benchmark_eval.py
(cd backend && .venv/bin/python -m perf.overhead)
backend/.venv/bin/python -I ml/build_report.py
```
