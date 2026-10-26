"""Allocation policies: the deployable form of an optimizer solution.

A policy maps each workload class to a traffic split, e.g.
    rules["coding:high"] = [{"model_id": 5, "model": "qwen/...", "share": 0.7},
                            {"model_id": 2, "model": "openai/...", "share": 0.3}]
At request time the router draws the primary model with these probabilities, so
over many requests the realised traffic matches the optimised allocation.
"""

import random

import numpy as np

from app.services.optimizer.problem import OptimizationInputs


def allocation_to_rules(inp: OptimizationInputs, X: np.ndarray) -> dict[str, list[dict]]:
    rules: dict[str, list[dict]] = {}
    for i, c in enumerate(inp.classes):
        split = [{"model_id": inp.models[j].model_id, "model": inp.models[j].name,
                  "share": round(float(X[i, j]), 4)}
                 for j in np.argsort(-X[i]) if X[i, j] > 1e-6]
        if split:
            rules[c.key] = split
    return rules


def pick_order(rule: list[dict], eligible_ids: set[int], rng: random.Random) -> list[int]:
    """Primary drawn by share among still-eligible policy models; the rest by share."""
    usable = [r for r in rule if r["model_id"] in eligible_ids and r["share"] > 0]
    if not usable:
        return []
    weights = [r["share"] for r in usable]
    primary = rng.choices(usable, weights=weights, k=1)[0]
    rest = sorted((r for r in usable if r is not primary), key=lambda r: -r["share"])
    return [primary["model_id"]] + [r["model_id"] for r in rest]
