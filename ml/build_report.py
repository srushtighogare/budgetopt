"""Assemble docs/evaluation.md from the metric files every evaluation script writes.

Each section reads its metrics JSON from ml/eval/ and says so if it hasn't been
produced yet, so the document can be regenerated at any time with:

    backend/.venv/bin/python -I ml/build_report.py
"""

import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
EVAL = ROOT / "eval"
OUT = ROOT.parent / "docs" / "evaluation.md"


def load(name: str) -> dict | None:
    p = EVAL / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def pending(what: str, command: str) -> str:
    return f"_Not produced yet. Run `{command}` ({what})._\n"


def classifier() -> str:
    m = load("classifier_metrics")
    if not m:
        return pending("task classifier", "make train-classifier && make eval-ood")
    ood = m.get("ood", {})
    agree = m.get("annotator_vs_source_label_agreement", {})
    family = {"logreg": "logistic regression", "linear_svm": "a calibrated linear SVM",
              "complement_nb": "complement naive Bayes"}.get(m["selected"], m["selected"])
    return f"""Task-type classifier: TF-IDF word and character n-grams with {family}
(selected by validation macro-F1 among logistic regression, calibrated linear SVM and
complement naive Bayes). Training data: {m['n_train']} prompts from six public datasets, all
labelled by one LLM annotator with one rubric.

| Evaluation | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Held-out split of the annotated set (n={m['n_test']}) | {m['test_accuracy']:.3f} | {m['test_macro_f1']:.3f} | same annotator as training |
| Out-of-distribution prompts (n={ood.get('n', '—')}) | {ood.get('accuracy', 0):.3f} | {ood.get('macro_f1', 0):.3f} | labelled by a different LLM; top-2 accuracy {ood.get('top2_accuracy', 0):.3f} |
| Agreement between two LLM annotators on the same OOD prompts | {ood.get('inter_annotator_agreement') or 0:.3f} | | practical ceiling for any classifier |

Inference: **{m['latency_ms_per_prompt']} ms per prompt** on one CPU core.

**Finding.** A first version trained on each dataset's own category labels scored 0.86 on an
in-distribution test but only 0.44 on independently labelled prompts: it had learned each
dataset's writing style (e.g. CodeAlpaca vs Alpaca) rather than the task. Relabelling a mixed
pool with a single annotator removed the shortcut (0.44 → {ood.get('accuracy', 0):.2f} on the
OOD set, against an inter-annotator ceiling of {ood.get('inter_annotator_agreement') or 0:.2f}).
Annotator agreement with the source datasets' own labels: `{agree}`.

Details: [`ml/eval/classifier_report.md`](../ml/eval/classifier_report.md).
"""


def workload_models() -> str:
    m = load("workload_models_metrics")
    if not m:
        return ("Complexity model and output-token estimator: the labelling run that produces "
                "their training data (cheap-model answer + strong reference + judge per prompt) "
                "is still collecting on free-tier quotas; until then the router uses the "
                "transparent fallbacks (a feature-based complexity heuristic and per-task median "
                "output lengths), and each routing explanation says so.\n\n"
                + pending("complexity model and token estimator", "make train-workload"))
    cx, te = m["complexity"], m["token_estimator"]
    rows = "\n".join(f"| {k.replace('_', ' ')} | {v['roc_auc']} | {v['brier']} |"
                     for k, v in cx["cv"].items())
    trows = "\n".join(f"| {k.replace('_', ' ')} | {v['mae_tokens']} |"
                      for k, v in te["cv"].items())
    return f"""Trained on {m['n']} prompts from the labelling run (5-fold cross-validation).

Complexity = probability that the cheap model's answer is not good enough
(cheap-model failure rate in the data: {cx['positive_rate']}).

| Model | ROC-AUC | Brier |
|---|---|---|
{rows}

Output-token estimator (target: tokens billed by the cheap model):

| Model | MAE (tokens) |
|---|---|
{trows}
"""


def profiling_section() -> str:
    return """Each model answers a fixed set of 64 prompts (8 per task type). Math answers are checked
against gold answers; other answers are graded 1–10 by a judge model that never grades its own
answers. Every graded answer is stored (`profiling_results`) and aggregated per model and task
(`model_performance`), which the optimizer and the dashboard's quality heatmap use. Latency
p50/p95 per model and task comes from live traffic plus profiling calls (synthetic demo traffic
is excluded). With 4–8 prompts per cell the per-task scores are indicative, not precise.
"""


def optimizer() -> str:
    m = load("optimizer_metrics")
    if not m:
        return pending("optimizer vs exact optimum", "python -I ml/eval_optimizer.py")
    g = m["gap_pct"]
    return f"""NSGA-II (pymoo) over monthly cost, quality and p95 latency, with traffic shares per
workload class as decision variables and hard per-band quality floors, compared with the exact
LP optimum of the same problem on {m['instances']} random instances ({m['checks']} budget checks).

| Metric | Value |
|---|---|
| Quality gap to the exact optimum (median / p90 / worst) | {g['median']}% / {g['p90']}% / {g['max']}% |
| NSGA-II runtime, median | {m['nsga2_seconds']['median']} s for the whole front (~{m['front_size_median']} allocations) |
| Exact LP / ILP solve time, median | {m['lp_ms_median']} ms / {m['ilp_ms_median']} ms |

The gap is largest at tight budgets (median {m['gap_pct_by_budget']['0.1']}% at 10% of the cost
range) and near zero at generous ones. Details: [`ml/eval/optimizer_report.md`](../ml/eval/optimizer_report.md).
"""


def benchmark() -> str:
    m = load("benchmark_metrics")
    if not m:
        return pending("routing benchmark on held-out prompts",
                       "python -I ml/benchmark_collect.py && python -I ml/benchmark_eval.py")
    r = m["results"]
    keys = [k for k in r if not k.startswith("BudgetOpt")] + [k for k in r
                                                              if k.startswith("BudgetOpt")]
    rows = "\n".join(
        f"| {k} | {r[k]['quality']:.2f} | {r[k]['quality_retained_vs_premium_pct']:.0f}% | "
        f"${r[k]['monthly_cost']:,.2f} | {r[k]['cost_reduction_vs_premium_pct']:.0f}% | "
        f"{r[k]['quality_gain_vs_linear_mix']:+.2f} |" for k in keys)
    return f"""{m['prompts']} held-out prompts × {len(m['models'])} models (response matrix; every answer
scored once by gold answer or an external judge). Costs projected to
{m['monthly_requests']:,} requests a month at current list prices. BudgetOpt's policies were
built from profiling data only, never from benchmark answers.

| Strategy | Quality (0–10) | Kept vs premium | Monthly cost | Saved vs premium | vs random mix |
|---|---|---|---|---|---|
{rows}

![cost vs quality](../ml/eval/benchmark_cost_quality.png)

Details and per-task results: [`ml/eval/benchmark_report.md`](../ml/eval/benchmark_report.md).
"""


def forecasting() -> str:
    m = load("forecast_metrics")
    if not m:
        return pending("forecaster benchmark", "python -I ml/eval_forecast.py")
    s = m["summary"]
    rows = "\n".join(f"| {k.replace('_', ' ')} | {v['median_total_ape_pct']}% | "
                     f"{v['median_daily_mae_pct_of_mean']}% |" for k, v in s.items())
    return f"""Exponential smoothing (statsmodels ETS; weekly seasonality, damped trend vs level chosen by
AICc; run-rate for short histories), 500 simulated paths with bootstrapped residuals for the
month-end total, overrun probability and date. Benchmark: {m['series']} synthetic daily-spend
series, last {m['horizon_days']} days held out.

| Method | Median error, 7-day total | Median daily error (% of mean) |
|---|---|---|
{rows}

- BudgetOpt beats the naive 7-day mean on {m['budgetopt_beats_naive_7_share']:.0%} of series.
- 90% interval coverage: {m['interval_90_coverage_uncalibrated']:.0%} raw → **{m['interval_90_coverage']:.0%}**
  after widening by a factor ({m['interval_scale_in_use']}) measured on a separate calibration set.
- {m['seconds_per_series']} s per forecast.

Every production forecast also carries its own 7-day backtest. Details: [`ml/eval/forecast_report.md`](../ml/eval/forecast_report.md).
"""


def semantic_cache() -> str:
    p = EVAL / "semantic_cache_report.md"
    if not p.exists():
        return pending("semantic cache evaluation", "python -I ml/eval_semantic_cache.py")
    text = p.read_text()
    two_stage = text.split("## Two-stage design (deployed)")[-1].strip()
    return f"""Local embeddings (BAAI/bge-small-en-v1.5, ~2 ms) with a pgvector HNSW index, evaluated on 100
curated pairs: 50 rewordings that should hit the cache and 50 near-misses that must not
(different entity, opposite direction, other audience, related question).

Embeddings alone put opposites as close as rewordings ("is coffee good / bad for your health" =
0.947), so the only threshold with no wrong answers (0.95) served just 38% of rewordings.
The deployed design accepts at ≥ 0.97 and asks a cheap verifier model in the 0.85–0.97 band:

{two_stage}

Further guards: identical context and numbers, cache only task types where similar questions
share answers, off for confidential teams. Details: [`ml/eval/semantic_cache_report.md`](../ml/eval/semantic_cache_report.md).
"""


def overhead() -> str:
    m = load("overhead_metrics")
    if not m:
        return pending("routing overhead load test", "cd backend && python -m perf.overhead")
    rows = []
    for mode, res in m.items():
        for name, r in res.items():
            tp = f"{r['throughput_rps']} req/s" if "throughput_rps" in r else ""
            rows.append(f"| {mode} | {name} | {r['p50_ms']} | {r['p95_ms']} | {tp} |")
    return f"""BudgetOpt's own latency per request, measured with the provider call replaced by an instant
answer (real PostgreSQL, Redis, classifier, routing, logging). Design target: p95 < 50 ms.

| Mode | Scenario | p50 ms | p95 ms | Throughput |
|---|---|---|---|---|
{chr(10).join(rows)}

Concurrent latency includes queueing on one worker; throughput is the capacity figure (service
time ≈ 1000 / throughput ms). Details: [`ml/eval/overhead_report.md`](../ml/eval/overhead_report.md).
"""


def tests_section() -> str:
    return """- Backend: pytest suite against real PostgreSQL and Redis covering auth and RBAC (including the
  report's "employee views another team's spend" case), the gateway's OpenAI compatibility,
  budget gate, caches, rate limits, fallback and circuit breaker, workload analysis, profiling,
  optimizer (non-dominance, exactness vs LP/ILP, budget feasibility), policies, forecasting,
  alerts, anomaly detection, what-if and the savings report.
- Frontend: Vitest tests for the API client's token refresh, formatting and components.
- CI (GitHub Actions) runs lint, migrations check, backend tests and the frontend type-check,
  tests and build on every push.
"""


def main() -> None:
    sections = [
        ("1. Workload analysis: task classifier", classifier()),
        ("2. Workload analysis: complexity and output tokens", workload_models()),
        ("3. Model profiling", profiling_section()),
        ("4. Optimizer", optimizer()),
        ("5. End-to-end routing benchmark", benchmark()),
        ("6. Spend forecasting", forecasting()),
        ("7. Semantic cache", semantic_cache()),
        ("8. Routing overhead", overhead()),
        ("9. Automated tests", tests_section()),
    ]
    body = "\n".join(f"## {title}\n\n{text}" for title, text in sections)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(f"""# BudgetOpt — evaluation

Generated {date.today().isoformat()} by `ml/build_report.py` from the metric files in `ml/eval/`.
All LLM-based labelling, profiling and benchmarking used free-tier API quotas; every label
records which model produced it.

{body}
## Limitations

- **Sample sizes.** Free-tier quotas limited the labelling run, the profiling set (4–8 prompts
  per model and task) and the benchmark. Per-task numbers are indicative; the overall
  comparisons are the main result.
- **LLM-as-judge.** Non-math quality is graded by an LLM. Judges never grade their own answers
  and the benchmark uses judges outside the evaluated models, but judge bias remains possible;
  math is checked against gold answers.
- **Prices** for preview models are approximate list prices and are configurable
  (`POST /api/v1/models/{{id}}/pricing`).
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
""")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
