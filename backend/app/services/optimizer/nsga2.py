"""Multi-objective allocation with NSGA-II (report objective O5).

Genotype: one weight in [0, 1] per *eligible* (class, model) pair. Decoding
normalises each class's weights into traffic shares, drops shares below
MIN_SHARE (a 2% split is not a useful routing rule) and renormalises, so every
individual is a valid allocation by construction and ineligible pairs can never
receive traffic.

The initial population is seeded with the obvious single-objective allocations
(cheapest / highest quality / fastest eligible model per class) plus random
individuals, which speeds convergence towards the ends of the Pareto front.
"""

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize

from app.services.optimizer.problem import OptimizationInputs

MIN_SHARE = 0.05


@dataclass
class Solution:
    allocation: np.ndarray  # classes x models
    monthly_cost: float
    quality: float
    latency_ms: float
    feasible: bool = True

    def as_dict(self) -> dict:
        return {"monthly_cost": round(self.monthly_cost, 6), "quality": round(self.quality, 4),
                "latency_ms": round(self.latency_ms, 1), "feasible": self.feasible}


@dataclass
class FrontResult:
    solutions: list[Solution]
    presets: dict[str, int]  # preset name -> index into solutions
    hypervolume: float
    generations: int
    evaluations: int


def decode(z: np.ndarray, inp: OptimizationInputs) -> np.ndarray:
    X = np.zeros(inp.eligible.shape)
    X[inp.eligible] = z
    for i in range(X.shape[0]):
        row = X[i]
        if not inp.eligible[i].any():
            continue
        if row.sum() <= 0:
            row[inp.eligible[i]] = 1.0
        row /= row.sum()
        if (row >= MIN_SHARE).any():
            row[row < MIN_SHARE] = 0.0
            row /= row.sum()
    return X


class AllocationProblem(Problem):
    def __init__(self, inp: OptimizationInputs):
        self.inp = inp
        n_var = int(inp.eligible.sum())
        super().__init__(n_var=n_var, n_obj=3, n_ieq_constr=1 if inp.budget else 0,
                         xl=0.0, xu=1.0)

    def _evaluate(self, Z, out, *args, **kwargs):
        F = np.array([self._objectives(z) for z in Z])
        out["F"] = F
        if self.inp.budget:
            out["G"] = F[:, :1] - self.inp.budget  # monthly cost <= budget

    def _objectives(self, z: np.ndarray) -> list[float]:
        cost, quality, latency = self.inp.evaluate(decode(z, self.inp))
        return [cost, -quality, latency]


def _seed_population(inp: OptimizationInputs, size: int, rng: np.random.Generator) -> np.ndarray:
    seeds = []
    for metric, best in ((inp.cost, np.argmin), (inp.quality, np.argmax),
                         (inp.latency, np.argmin)):
        X = np.zeros(inp.eligible.shape)
        for i in range(X.shape[0]):
            if inp.eligible[i].any():
                vals = np.where(inp.eligible[i], metric[i], np.inf if best is np.argmin
                                else -np.inf)
                X[i, best(vals)] = 1.0
        seeds.append(X[inp.eligible])
    rand = rng.random((max(0, size - len(seeds)), int(inp.eligible.sum())))
    return np.vstack([np.array(seeds), rand])


def _knee(F: np.ndarray) -> int:
    """Point closest to the ideal after min-max normalising each objective."""
    lo, hi = F.min(axis=0), F.max(axis=0)
    N = (F - lo) / np.where(hi - lo > 0, hi - lo, 1)
    return int(np.argmin(np.linalg.norm(N, axis=1)))


def optimize(inp: OptimizationInputs, *, pop_size: int = 100, generations: int = 150,
             seed: int = 1) -> FrontResult:
    if not inp.eligible.any():
        raise ValueError("no eligible (class, model) pairs; check profiles and constraints")
    problem = AllocationProblem(inp)
    rng = np.random.default_rng(seed)
    algorithm = NSGA2(pop_size=pop_size, sampling=_seed_population(inp, pop_size, rng),
                      eliminate_duplicates=True)
    res = minimize(problem, algorithm, ("n_gen", generations), seed=seed, verbose=False)

    if res.X is None:  # budget infeasible for every individual: report the cheapest instead
        Z = res.pop.get("X")
        F = res.pop.get("F")
        idx = [int(np.argmin(F[:, 0]))]
        feasible = False
    else:
        Z = np.atleast_2d(res.X)
        F = np.atleast_2d(res.F)
        idx = list(range(len(Z)))
        feasible = True

    solutions, seen = [], set()
    for k in idx:
        X = decode(Z[k], inp)
        key = tuple(np.round(X, 3).ravel())
        if key in seen:  # different genotypes can decode to the same allocation
            continue
        seen.add(key)
        cost, quality, latency = inp.evaluate(X)
        solutions.append(Solution(X, cost, quality, latency, feasible))
    solutions.sort(key=lambda s: s.monthly_cost)

    Fs = np.array([[s.monthly_cost, -s.quality, s.latency_ms] for s in solutions])
    presets = {"cost_saver": int(np.argmin(Fs[:, 0])), "quality_first": int(np.argmin(Fs[:, 1])),
               "balanced": _knee(Fs)}

    # Hypervolume of the normalised front w.r.t. reference point (1.1, 1.1, 1.1)
    lo, hi = Fs.min(axis=0), Fs.max(axis=0)
    N = (Fs - lo) / np.where(hi - lo > 0, hi - lo, 1)
    hv = float(HV(ref_point=np.full(3, 1.1))(N)) if len(N) else 0.0
    return FrontResult(solutions, presets, hv, generations, int(res.algorithm.evaluator.n_eval))
