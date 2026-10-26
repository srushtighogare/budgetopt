"""Benchmark the NSGA-II allocator against the exact optimum.

Generates random allocation problems (24 workload classes; 4–8 models with random
prices spanning two orders of magnitude, per-task quality and latency) and, for each,
compares the best NSGA-II front point within a budget to the exact LP optimum of the
same split-allocation problem at 10%, 50% and 90% of the front's cost range. Also
reports runtime, front size and the exact ILP (one model per class) for contrast.

Usage: backend/.venv/bin/python -I ml/eval_optimizer.py -> ml/eval/optimizer_report.md
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from app.services.optimizer import ilp, nsga2  # noqa: E402
from app.services.optimizer.problem import (  # noqa: E402
    BANDS,
    TASKS,
    ModelProfile,
    OptimizationInputs,
    WorkloadClass,
)

INSTANCES = 20
SEED = 7


def instance(rng: np.random.Generator) -> OptimizationInputs:
    n_models = int(rng.integers(4, 9))
    models = []
    for j in range(n_models):
        tier = rng.uniform(0, 1)  # 0 = cheapest, 1 = premium
        p_in = 0.05 * 30 ** tier
        quality = 6.0 + 3.0 * tier
        models.append(ModelProfile(
            j, f"m{j}", f"m{j}", p_in, p_in * rng.uniform(3, 6), 131_072, 0,
            {t: float(np.clip(quality + rng.normal(0, 0.8), 1, 10)) for t in TASKS},
            frozenset(), {t: float(800 + 3000 * tier * rng.uniform(0.6, 1.4)) for t in TASKS}))
    shares = rng.dirichlet(np.ones(len(TASKS) * len(BANDS)))
    classes = [WorkloadClass(t, b, float(s), float(rng.integers(80, 800)),
                             float(rng.integers(50, 900)))
               for (t, b), s in zip([(t, b) for t in TASKS for b in BANDS], shares,
                                    strict=True)]
    return OptimizationInputs(classes=classes, models=models, monthly_requests=100_000)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for k in range(INSTANCES):
        inp = instance(rng)
        t0 = time.perf_counter()
        front = nsga2.optimize(inp, pop_size=100, generations=150, seed=k)
        secs = time.perf_counter() - t0
        lo, hi = front.solutions[0].monthly_cost, front.solutions[-1].monthly_cost
        for frac in (0.1, 0.5, 0.9):
            budget = lo + frac * (hi - lo)
            lp = ilp.solve(inp, "max_quality", budget=budget, relax=True)
            ip = ilp.solve(inp, "max_quality", budget=budget)
            best = max(s.quality for s in front.solutions if s.monthly_cost <= budget + 1e-9)
            rows.append({"instance": k, "models": len(inp.models), "budget_frac": frac,
                         "lp_quality": lp.quality, "nsga2_quality": best,
                         "ilp_quality": ip.quality,
                         "gap_pct": 100 * (lp.quality - best) / lp.quality,
                         "nsga2_seconds": secs, "front_size": len(front.solutions),
                         "hypervolume": front.hypervolume, "lp_ms": lp.wall_ms,
                         "ilp_ms": ip.wall_ms})
    gaps = np.array([r["gap_pct"] for r in rows])
    by_frac = {f: np.array([r["gap_pct"] for r in rows if r["budget_frac"] == f])
               for f in (0.1, 0.5, 0.9)}
    secs = np.array([r["nsga2_seconds"] for r in rows[::3]])
    metrics = {
        "instances": INSTANCES, "checks": len(rows),
        "gap_pct": {"median": round(float(np.median(gaps)), 3),
                    "p90": round(float(np.percentile(gaps, 90)), 3),
                    "max": round(float(gaps.max()), 3)},
        "gap_pct_by_budget": {str(f): round(float(np.median(v)), 3) for f, v in by_frac.items()},
        "nsga2_seconds": {"median": round(float(np.median(secs)), 2),
                          "max": round(float(secs.max()), 2)},
        "front_size_median": int(np.median([r["front_size"] for r in rows[::3]])),
        "lp_ms_median": round(float(np.median([r["lp_ms"] for r in rows])), 1),
        "ilp_ms_median": round(float(np.median([r["ilp_ms"] for r in rows])), 1),
        "ilp_vs_lp_quality_gap_pct_median": round(float(np.median(
            [100 * (r["lp_quality"] - r["ilp_quality"]) / r["lp_quality"] for r in rows])), 3),
    }
    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    m = metrics
    by_budget = " / ".join(f"{v}%" for v in m["gap_pct_by_budget"].values())
    runtime = f"{m['nsga2_seconds']['median']} s / {m['nsga2_seconds']['max']} s"
    ilp_gap = m["ilp_vs_lp_quality_gap_pct_median"]
    (out / "optimizer_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "optimizer_report.md").write_text(f"""# Optimizer — NSGA-II vs the exact optimum

{INSTANCES} random allocation problems (24 workload classes, 4–8 models with prices spanning
~30×, per-task quality and latency). For each, the best NSGA-II solution within a budget is
compared with the exact LP optimum of the same split-allocation problem, at 10%, 50% and 90%
of the front's cost range ({len(rows)} checks).

| Metric | Value |
|---|---|
| Quality gap to the exact optimum, median | {metrics['gap_pct']['median']}% |
| Quality gap, 90th percentile | {metrics['gap_pct']['p90']}% |
| Quality gap, worst case | {metrics['gap_pct']['max']}% |
| Median gap at 10% / 50% / 90% of the cost range | {by_budget} |
| NSGA-II runtime (100 × 150 generations), median / max | {runtime} |
| Pareto front size, median | {metrics['front_size_median']} allocations |
| Exact LP solve time, median | {metrics['lp_ms_median']} ms |
| One-model-per-class ILP quality below the split optimum, median | {ilp_gap}% |

The LP finds one point per budget exactly. NSGA-II returns the whole cost–quality–latency
front in one run, which the dashboard shows as the choice of policies.
""")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
