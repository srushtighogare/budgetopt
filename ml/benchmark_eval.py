"""Evaluate routing strategies on the benchmark response matrix.

Uses only prompts for which every benchmark model has a scored answer. For each
strategy, the chosen model's REAL answer for that prompt supplies the score (0-10),
billed tokens (priced at current list prices) and latency.

Strategies
  all-<model>       every prompt to one model (the un-optimised baselines)
  random            each model with equal probability (expected value)
  keyword router    the earlier prototype's heuristic: prompt length, reasoning
                    keywords, code indicators, multiple '?' -> premium if score >= 0.3
  BudgetOpt         the deployed pipeline run offline: the task classifier and
                    complexity scoring assign each prompt a workload class, and the
                    optimizer's allocation policies (built ONLY from model-profiling
                    data, never from benchmark answers) give each class's traffic split.
                    Every point of the NSGA-II Pareto front is evaluated, giving
                    BudgetOpt's cost-quality curve; presets are marked.
  cheapest-eligible the router's no-policy default: cheapest model whose profiled
                    quality meets the band's floor
  oracle            per prompt, the cheapest model with the best score (upper bound;
                    needs hindsight, not achievable)

Also reports the quality of a cost-matched random mix of the cheapest and the premium
model (the straight line between them), since a router is only useful if it beats it.

Usage: backend/.venv/bin/python -I ml/benchmark_eval.py
Outputs: ml/eval/benchmark_report.md, benchmark_metrics.json, benchmark_cost_quality.png
"""

import asyncio
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from app.db.session import SessionLocal  # noqa: E402
from app.seed import MODELS as SEED_MODELS  # noqa: E402
from app.services.optimizer import nsga2  # noqa: E402
from app.services.optimizer.policy import allocation_to_rules  # noqa: E402
from app.services.optimizer.problem import (  # noqa: E402
    DEFAULT_Q_MIN,
    OptimizationInputs,
    WorkloadClass,
    _model_profiles,
)
from app.services.providers.base import ChatMessage  # noqa: E402
from app.services.workload.analyzer import analyze  # noqa: E402

BENCH = ROOT / "datasets" / "benchmark"
EVAL = ROOT / "eval"
MODELS = ["openai/gpt-oss-20b", "gemini-3.5-flash-lite", "openai/gpt-oss-120b",
          "gemini-3.5-flash"]
PREMIUM = "gemini-3.5-flash"
PRICES = {name: (float(p_in), float(p_out)) for _, name, _, _, _, p_in, p_out in SEED_MODELS}
MONTHLY_REQUESTS = 100_000

# --- earlier prototype's keyword router (prudonomics/backend/complexity_scorer.py) ---
REASONING_KEYWORDS = ["analyze", "explain why", "compare", "step by step", "step-by-step",
                      "evaluate", "justify", "reasoning", "pros and cons", "trade-off",
                      "tradeoff", "architecture", "design", "optimize", "debug", "prove"]
CODE_INDICATORS = ["```", "def ", "function ", "class ", "import ", "SELECT ", "<html"]


def keyword_score(prompt: str) -> float:
    score, low = 0.0, prompt.lower()
    if len(prompt) > 400:
        score += 0.3
    elif len(prompt) > 150:
        score += 0.15
    if any(k in low for k in REASONING_KEYWORDS):
        score += 0.35
    if any(c.lower() in low for c in CODE_INDICATORS):
        score += 0.2
    if prompt.count("?") >= 2:
        score += 0.15
    return min(score, 1.0)


def load() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    def read(name):
        with open(BENCH / f"{name}.jsonl") as f:
            return [json.loads(line) for line in f]

    prompts = read("prompts")
    answers = {(a["bid"], a["model"]): a for a in read("answers")}
    scores = {(s["bid"], s["model"]): s for s in read("scores")}
    cell = {}
    for (bid, model), a in answers.items():
        if (bid, model) in scores:
            p_in, p_out = PRICES[model]
            cell[(bid, model)] = {
                "score": scores[(bid, model)]["score"], "judge": scores[(bid, model)]["judge"],
                "cost": (a["input_tokens"] * p_in + a["output_tokens"] * p_out) / 1e6,
                "latency": a["latency_ms"], "out_tokens": a["output_tokens"],
            }
    complete = [p for p in prompts if all((p["bid"], m) in cell for m in MODELS)]
    return complete, cell


def evaluate(prompts, cell, choose) -> dict:
    """`choose(prompt) -> {model: share}`; returns mean score, cost and latency."""
    score = cost = 0.0
    lat = []
    for p in prompts:
        for m, w in choose(p).items():
            c = cell[(p["bid"], m)]
            score += w * c["score"]
            cost += w * c["cost"]
            lat.append((c["latency"], w))
    n = len(prompts)
    lats = np.array([x for x, _ in lat])
    weights = np.array([w for _, w in lat])
    order = np.argsort(lats)
    cum = np.cumsum(weights[order]) / weights.sum()
    p95 = float(lats[order][np.searchsorted(cum, 0.95)])
    return {"quality": score / n, "cost_per_request": cost / n,
            "monthly_cost": cost / n * MONTHLY_REQUESTS, "p95_latency_ms": p95}


async def budgetopt_policies(prompts) -> tuple[dict, list[dict], dict, list]:
    """Classify each prompt, then optimise allocations from profiling data only."""
    profiles = {p["bid"]: analyze([ChatMessage("user", p["text"])]) for p in prompts}
    counts = Counter(f"{w.task_type.value}:{w.complexity_band.value}" for w in profiles.values())
    tokens: dict[str, list[tuple[int, int]]] = {}
    for p in prompts:
        w = profiles[p["bid"]]
        tokens.setdefault(f"{w.task_type.value}:{w.complexity_band.value}", []).append(
            (w.est_input_tokens, w.est_output_tokens))
    classes = [WorkloadClass(k.split(":")[0], k.split(":")[1], n / len(prompts),
                             float(np.median([t[0] for t in tokens[k]])),
                             float(np.median([t[1] for t in tokens[k]])))
               for k, n in sorted(counts.items())]
    async with SessionLocal() as db:
        models = [m for m in await _model_profiles(db) if m.name in MODELS]
    missing = sorted(set(MODELS) - {m.name for m in models if m.quality})
    inp = OptimizationInputs(classes=classes, models=models, monthly_requests=MONTHLY_REQUESTS)
    front = nsga2.optimize(inp, pop_size=120, generations=200)
    policies = [allocation_to_rules(inp, s.allocation) for s in front.solutions]
    # cheapest-eligible (the router's default without a policy)
    cheapest = {}
    for i, c in enumerate(inp.classes):
        js = np.nonzero(inp.eligible[i])[0]
        if js.size:
            cheapest[c.key] = inp.models[js[np.argmin(inp.cost[i, js])]].name
    return ({b: f"{w.task_type.value}:{w.complexity_band.value}" for b, w in profiles.items()},
            policies, {"presets": front.presets, "notes": inp.notes,
                       "unprofiled_models": missing, "cheapest": cheapest,
                       "class_counts": dict(counts)}, front.solutions)


def linear_mix_quality(cheap: dict, premium: dict, cost: float) -> float:
    """Quality of randomly mixing the cheapest and premium model at a given cost."""
    if premium["monthly_cost"] == cheap["monthly_cost"]:
        return cheap["quality"]
    t = (cost - cheap["monthly_cost"]) / (premium["monthly_cost"] - cheap["monthly_cost"])
    t = min(1.0, max(0.0, t))
    return cheap["quality"] + t * (premium["quality"] - cheap["quality"])


def main() -> None:
    prompts, cell = load()
    if len(prompts) < 20:
        print(f"only {len(prompts)} complete prompts so far; run benchmark_collect.py first")
        return
    results: dict[str, dict] = {}
    for m in MODELS:
        results[f"all → {m}"] = evaluate(prompts, cell, lambda p, m=m: {m: 1.0})
    results["random"] = evaluate(prompts, cell, lambda p: {m: 1 / len(MODELS) for m in MODELS})
    results["keyword router (old prototype)"] = evaluate(
        prompts, cell, lambda p: {PREMIUM if keyword_score(p["text"]) >= 0.3
                                  else "openai/gpt-oss-20b": 1.0})

    def oracle(p):
        best = max(cell[(p["bid"], m)]["score"] for m in MODELS)
        ok = [m for m in MODELS if cell[(p["bid"], m)]["score"] == best]
        return {min(ok, key=lambda m: cell[(p["bid"], m)]["cost"]): 1.0}
    results["oracle (hindsight)"] = evaluate(prompts, cell, oracle)

    klass, policies, meta, solutions = asyncio.run(budgetopt_policies(prompts))
    results["BudgetOpt cheapest-eligible (no policy)"] = evaluate(
        prompts, cell, lambda p: {meta["cheapest"].get(klass[p["bid"]], "openai/gpt-oss-20b"): 1.0})

    def by_policy(rules):
        def choose(p):
            split = rules.get(klass[p["bid"]])
            if not split:
                return {meta["cheapest"].get(klass[p["bid"]], "openai/gpt-oss-20b"): 1.0}
            total = sum(s["share"] for s in split)
            return {s["model"]: s["share"] / total for s in split}
        return choose

    curve = [evaluate(prompts, cell, by_policy(rules)) for rules in policies]
    for name, idx in meta["presets"].items():
        results[f"BudgetOpt {name}"] = curve[idx]

    premium = results[f"all → {PREMIUM}"]
    cheapest_single = min((results[f"all → {m}"] for m in MODELS),
                          key=lambda r: r["monthly_cost"])
    for r in results.values():
        r["cost_reduction_vs_premium_pct"] = 100 * (1 - r["monthly_cost"] / premium["monthly_cost"])
        r["quality_retained_vs_premium_pct"] = 100 * r["quality"] / premium["quality"]
        r["quality_gain_vs_linear_mix"] = r["quality"] - linear_mix_quality(
            cheapest_single, premium, r["monthly_cost"])

    # --- per-task view for the balanced preset
    per_task = {}
    bal = by_policy(policies[meta["presets"]["balanced"]])
    for task in sorted({p["task_type"] for p in prompts}):
        sub = [p for p in prompts if p["task_type"] == task]
        per_task[task] = {
            "n": len(sub),
            "budgetopt_balanced": round(evaluate(sub, cell, bal)["quality"], 2),
            "premium": round(evaluate(sub, cell, lambda p: {PREMIUM: 1.0})["quality"], 2),
            "cheapest": round(evaluate(sub, cell, lambda p: {"openai/gpt-oss-20b": 1.0})
                              ["quality"], 2)}

    judges = Counter(cell[(p["bid"], m)]["judge"] for p in prompts for m in MODELS)
    metrics = {"prompts": len(prompts), "models": MODELS, "monthly_requests": MONTHLY_REQUESTS,
               "judges": dict(judges), "results": results,
               "budgetopt_curve": curve, "per_task": per_task,
               "budgetopt_meta": {k: v for k, v in meta.items() if k != "cheapest"}}
    EVAL.mkdir(exist_ok=True)
    (EVAL / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    plot(results, curve, cheapest_single, premium)
    write_report(metrics, results, per_task)
    for k, r in results.items():
        print(f"{k:42} quality {r['quality']:.2f} ({r['quality_retained_vs_premium_pct']:.0f}%)"
              f"  ${r['monthly_cost']:8.2f}/mo (saves {r['cost_reduction_vs_premium_pct']:.0f}%)"
              f"  vs-mix {r['quality_gain_vs_linear_mix']:+.2f}")


def plot(results, curve, cheap, premium) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.set_facecolor("#fcfcfb")
    ax.grid(color="#e1e0d9", linewidth=0.8)
    # Each optimised policy, evaluated on the held-out prompts (not re-fitted to them)
    ax.scatter([p["monthly_cost"] for p in curve], [p["quality"] for p in curve], s=22,
               color="#2a78d6", edgecolor="white", linewidth=0.8, zorder=3,
               label="BudgetOpt policies (Pareto front, held-out evaluation)")
    # A cost-proportional mix is a curve on a log axis, so draw it from many points
    xs = np.geomspace(cheap["monthly_cost"], premium["monthly_cost"], 100)
    ax.plot(xs, [linear_mix_quality(cheap, premium, x) for x in xs], "--", color="#898781",
            linewidth=1.5, label="Random mix of cheapest and premium")
    for name, r in results.items():
        if name.startswith("all →"):
            ax.scatter(r["monthly_cost"], r["quality"], marker="D", s=60, color="#eb6834",
                       edgecolor="white", zorder=4)
            ax.annotate(name.replace("all → ", ""), (r["monthly_cost"], r["quality"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8, color="#52514e")
        elif name in ("keyword router (old prototype)", "random", "oracle (hindsight)"):
            ax.scatter(r["monthly_cost"], r["quality"], marker="s", s=50, color="#1baf7a",
                       edgecolor="white", zorder=4)
            ax.annotate(name, (r["monthly_cost"], r["quality"]), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(f"Monthly cost for {MONTHLY_REQUESTS:,} requests (USD, log scale)",
                  color="#52514e")
    ax.set_ylabel("Mean answer quality (0–10)", color="#52514e")
    ax.set_title("Cost vs quality on held-out benchmark prompts", color="#0b0b0b")
    ax.scatter([], [], marker="D", color="#eb6834", label="Single model (no routing)")
    ax.scatter([], [], marker="s", color="#1baf7a", label="Other strategies")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(EVAL / "benchmark_cost_quality.png", dpi=150)


def write_report(metrics, results, per_task) -> None:
    rows = "\n".join(
        f"| {k} | {r['quality']:.2f} | {r['quality_retained_vs_premium_pct']:.0f}% | "
        f"${r['monthly_cost']:,.2f} | {r['cost_reduction_vs_premium_pct']:.0f}% | "
        f"{r['quality_gain_vs_linear_mix']:+.2f} | {r['p95_latency_ms'] / 1000:.1f} s |"
        for k, r in results.items())
    tasks = "\n".join(f"| {re.sub('_', ' ', t)} | {v['n']} | {v['cheapest']} | "
                      f"{v['budgetopt_balanced']} | {v['premium']} |" for t, v in per_task.items())
    meta = metrics["budgetopt_meta"]
    (EVAL / "benchmark_report.md").write_text(f"""# Routing benchmark

{metrics['prompts']} held-out prompts (never used to train the classifier, not in the
profiling set) × {len(metrics['models'])} models: {', '.join(metrics['models'])}. Every model
answered every prompt once; math answers were checked against gold answers and all other
answers were graded 1–10 by a judge outside the benchmark models ({metrics['judges']}).
Strategies are scored by looking up the real answer of the model they pick (response-matrix
evaluation, as in RouterBench / RouteLLM). Costs use current list prices, projected to
{metrics['monthly_requests']:,} requests a month.

BudgetOpt's allocation policies were optimised from **profiling data only**; the benchmark
answers were never used to build them. Quality floors per complexity band: {DEFAULT_Q_MIN}.

| Strategy | Quality | Kept vs premium | Monthly cost | Saved vs premium | vs random mix | p95 |
|---|---|---|---|---|---|---|
{rows}

*vs random mix*: quality gained over randomly splitting traffic between the cheapest and the
premium model at the same cost (positive = the routing decisions add value).

![cost vs quality](benchmark_cost_quality.png)

## Quality by task type (balanced policy)

| Task type | Prompts | Cheapest model | BudgetOpt balanced | Premium model |
|---|---|---|---|---|
{tasks}

Workload classes seen: `{meta['class_counts']}`
Notes: {meta['notes'] or 'none'}
Models without a quality profile: {meta['unprofiled_models'] or 'none'}
""")


if __name__ == "__main__":
    main()
