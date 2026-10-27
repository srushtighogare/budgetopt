"""Workload analysis (report architecture item 2): task type, complexity and
token requirements for a request, computed before routing.

Three trained models are loaded from ./artifacts (built by the scripts in ml/):
  task_classifier.joblib   TF-IDF + calibrated linear SVM   -> task type + confidence
  complexity.joblib        gradient boosting on features    -> P(cheap model's answer
                                                               is not acceptable)
  token_estimator.joblib   gradient boosting regressor      -> expected output tokens

Missing artifacts degrade gracefully (and say so in `sources`): complexity falls
back to a transparent feature heuristic, output tokens to per-task medians.
"""

import logging
import time
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.db.models import ComplexityBand, TaskType
from app.services.providers.base import ChatMessage
from app.services.tokens import estimate_input_tokens
from app.services.workload.features import prompt_features

log = logging.getLogger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"
TASKS = [t.value for t in TaskType]

# complexity score -> band
LOW_MAX, MEDIUM_MAX = 0.33, 0.66

# Fallback output-token medians per task (used until token_estimator.joblib exists)
DEFAULT_OUTPUT_TOKENS = {
    "translation": 60, "summarization": 150, "simple_qa": 200, "extraction": 180,
    "creative_writing": 450, "coding": 600, "reasoning_math": 700, "analysis": 650,
}


@dataclass(frozen=True)
class WorkloadProfile:
    task_type: TaskType
    task_confidence: float
    task_alternatives: list[tuple[str, float]]
    complexity: float
    complexity_band: ComplexityBand
    est_input_tokens: int
    est_output_tokens: int
    analysis_ms: float
    sources: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        return (f"Classified as {self.task_type.value} (confidence {self.task_confidence:.2f}), "
                f"complexity {self.complexity_band.value} ({self.complexity:.2f}), "
                f"~{self.est_input_tokens} input / ~{self.est_output_tokens} output tokens.")


@cache
def _load(name: str) -> dict[str, Any] | None:
    path = ARTIFACTS / f"{name}.joblib"
    if not path.exists():
        log.warning("workload artifact %s missing; using fallback", path.name)
        return None
    return joblib.load(path)


def warm_up() -> None:
    """Load models at startup so the first request doesn't pay the load time."""
    for name in ("task_classifier", "complexity", "token_estimator"):
        _load(name)


def band_of(score: float, thresholds: tuple[float, float] = (LOW_MAX, MEDIUM_MAX)
            ) -> ComplexityBand:
    low_max, medium_max = thresholds
    if score <= low_max:
        return ComplexityBand.LOW
    return ComplexityBand.MEDIUM if score <= medium_max else ComplexityBand.HIGH


def _task_onehot(task: str) -> np.ndarray:
    v = np.zeros(len(TASKS))
    v[TASKS.index(task)] = 1.0
    return v


def model_input(text: str, task: str) -> np.ndarray:
    """Feature row shared by the complexity model and token estimator."""
    return np.concatenate([prompt_features(text), _task_onehot(task)])[None, :]


def _heuristic_complexity(feats: np.ndarray, task: str) -> float:
    # Transparent fallback: task prior nudged by length, constraints, math/code signals
    prior = {"translation": 0.1, "summarization": 0.2, "simple_qa": 0.2, "extraction": 0.25,
             "creative_writing": 0.3, "coding": 0.5, "reasoning_math": 0.6, "analysis": 0.45}
    log_tokens, n_constraints, code, math = feats[0], feats[5], feats[6], feats[7]
    score = prior[task] + 0.04 * max(0.0, log_tokens - 4) + 0.03 * min(n_constraints, 5)
    score += 0.02 * min(code, 5) + 0.03 * min(math, 5)
    return float(min(1.0, max(0.0, score)))


def request_text(messages: list[ChatMessage]) -> str:
    """The latest user turn defines the task; system prompts are style, not task."""
    users = [m.content for m in messages if m.role == "user"]
    return users[-1] if users else messages[-1].content


def analyze(messages: list[ChatMessage]) -> WorkloadProfile:
    t0 = time.perf_counter()
    text = request_text(messages)
    sources: dict[str, str] = {}

    clf = _load("task_classifier")
    if clf:
        probs = clf["pipeline"].predict_proba([text])[0]
        classes = list(clf["pipeline"].classes_)
        order = np.argsort(probs)[::-1]
        task, conf = classes[order[0]], float(probs[order[0]])
        alts = [(classes[i], round(float(probs[i]), 3)) for i in order[1:3]]
        sources["task"] = clf.get("family", "model")
    else:
        task, conf, alts = "simple_qa", 0.0, []
        sources["task"] = "fallback"

    x = model_input(text, task)
    cx = _load("complexity")
    thresholds = (LOW_MAX, MEDIUM_MAX)
    if cx:
        complexity = float(cx["model"].predict_proba(x)[0, 1])
        # the trained model ships band cut-offs measured on its own predictions
        thresholds = tuple(cx.get("band_thresholds", thresholds))
        sources["complexity"] = "model"
    else:
        complexity = _heuristic_complexity(x[0], task)
        sources["complexity"] = "heuristic"

    est_in = estimate_input_tokens(messages)
    te = _load("token_estimator")
    if te:
        # trained on log1p(output tokens) for the cheap model; clip to a sane range
        est_out = int(np.clip(np.expm1(te["model"].predict(x)[0]), 1, 16_000))
        sources["output_tokens"] = "model"
    else:
        est_out = DEFAULT_OUTPUT_TOKENS[task]
        sources["output_tokens"] = "task median"

    return WorkloadProfile(
        task_type=TaskType(task), task_confidence=round(conf, 3), task_alternatives=alts,
        complexity=round(complexity, 3), complexity_band=band_of(complexity, thresholds),
        est_input_tokens=est_in, est_output_tokens=est_out,
        analysis_ms=round((time.perf_counter() - t0) * 1000, 2), sources=sources,
    )
