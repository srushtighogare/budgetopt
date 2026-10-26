# Optimizer — NSGA-II vs the exact optimum

20 random allocation problems (24 workload classes, 4–8 models with prices spanning
~30×, per-task quality and latency). For each, the best NSGA-II solution within a budget is
compared with the exact LP optimum of the same split-allocation problem, at 10%, 50% and 90%
of the front's cost range (60 checks).

| Metric | Value |
|---|---|
| Quality gap to the exact optimum, median | 0.832% |
| Quality gap, 90th percentile | 2.486% |
| Quality gap, worst case | 5.505% |
| Median gap at 10% / 50% / 90% of the cost range | 1.578% / 1.052% / 0.125% |
| NSGA-II runtime (100 × 150 generations), median / max | 1.5 s / 1.62 s |
| Pareto front size, median | 98 allocations |
| Exact LP solve time, median | 1.0 ms |
| One-model-per-class ILP quality below the split optimum, median | 0.041% |

The LP finds one point per budget exactly. NSGA-II returns the whole cost–quality–latency
front in one run, which the dashboard shows as the choice of policies.
