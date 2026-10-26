"""Exact single-objective allocation with integer linear programming (OR-Tools / SCIP).

Each workload class is assigned to exactly one eligible model (binary y[c, m]).
Two formulations:

  max_quality   maximise  Σ share_c · q(c,m) · y        s.t.  monthly cost <= budget
  min_cost      minimise  N · Σ share_c · cost(c,m) · y  s.t.  mean quality >= quality_floor

Optional in both: mean p95 latency <= latency_max. The ILP gives the provable
optimum for one objective under hard limits (one model per class).

With relax=True the variables are continuous in [0, 1], i.e. traffic may be split
between models as NSGA-II allows. That LP is solved exactly, so it is the true
optimum of the split-allocation problem and the right yardstick for measuring how
close the NSGA-II front gets.
"""

from dataclasses import dataclass

import numpy as np
from ortools.linear_solver import pywraplp

from app.services.optimizer.problem import OptimizationInputs


@dataclass
class ILPResult:
    status: str  # optimal | feasible | infeasible
    allocation: np.ndarray | None
    monthly_cost: float | None
    quality: float | None
    latency_ms: float | None
    wall_ms: float


def solve(inp: OptimizationInputs, objective: str = "max_quality", *,
          budget: float | None = None, quality_floor: float | None = None,
          mean_latency_max_ms: float | None = None, time_limit_s: int = 20,
          relax: bool = False) -> ILPResult:
    if objective not in ("max_quality", "min_cost"):
        raise ValueError("objective must be 'max_quality' or 'min_cost'")
    solver = pywraplp.Solver.CreateSolver("GLOP" if relax else "SCIP")
    solver.SetTimeLimit(time_limit_s * 1000)
    C, M = inp.eligible.shape
    share = inp.shares
    var = (lambda name: solver.NumVar(0.0, 1.0, name)) if relax else solver.BoolVar
    y = {(i, j): var(f"y_{i}_{j}") for i in range(C) for j in range(M) if inp.eligible[i, j]}

    for i in range(C):
        terms = [y[i, j] for j in range(M) if (i, j) in y]
        if terms:
            solver.Add(sum(terms) == 1)

    cost = sum(inp.monthly_requests * share[i] * inp.cost[i, j] * v for (i, j), v in y.items())
    quality = sum(share[i] * inp.quality[i, j] * v for (i, j), v in y.items())
    latency = sum(share[i] * inp.latency[i, j] * v for (i, j), v in y.items())

    budget = budget if budget is not None else inp.budget
    if objective == "max_quality":
        if budget is not None:
            solver.Add(cost <= budget)
        solver.Maximize(quality)
    else:
        if quality_floor is not None:
            solver.Add(quality >= quality_floor)
        solver.Minimize(cost)
    if mean_latency_max_ms is not None:
        solver.Add(latency <= mean_latency_max_ms)

    code = solver.Solve()
    status = {pywraplp.Solver.OPTIMAL: "optimal",
              pywraplp.Solver.FEASIBLE: "feasible"}.get(code, "infeasible")
    wall = float(solver.wall_time())
    if status == "infeasible":
        return ILPResult(status, None, None, None, None, wall)
    X = np.zeros((C, M))
    for (i, j), v in y.items():
        X[i, j] = v.solution_value() if relax else round(v.solution_value())
    c, q, lat = inp.evaluate(X)
    return ILPResult(status, X, c, q, lat, wall)
