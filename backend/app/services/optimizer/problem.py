"""Optimization problem definition shared by NSGA-II and the ILP.

Workload classes  c = (task type, complexity band), each with a traffic share and
                  typical input/output tokens (from the team's recent requests,
                  or a uniform prior when there is no traffic yet)
Models            m, each with current price, per-task quality (0–10) and p95 latency
Decision          X[c, m] = share of class c's traffic sent to model m, rows sum to 1

Objectives (for an allocation X):
  monthly cost  = N * Σ_c share_c Σ_m X[c,m] * cost(c,m)        (minimise)
  quality       = Σ_c share_c Σ_m X[c,m] * q(m, task_c)         (maximise)
  latency       = Σ_c share_c Σ_m X[c,m] * p95(m, task_c)       (minimise)

Hard constraints (report, Ethics: never pick a model only because it is cheap):
  a model may serve class c only if q(m, task_c) >= q_min[band_c], its p95 latency
  is within the limit, the class fits its context window and the model is cleared
  for the team's privacy tier. Optionally: monthly cost <= budget.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from statistics import median

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import (
    ComplexityBand,
    LLMModel,
    ModelPerformance,
    PrivacyTier,
    Request,
    RequestStatus,
    TaskType,
    Team,
)
from app.services.workload.analyzer import DEFAULT_OUTPUT_TOKENS

TASKS = [t.value for t in TaskType]
BANDS = [b.value for b in ComplexityBand]
DEFAULT_Q_MIN = {"low": 6.0, "medium": 7.0, "high": 8.0}
DEFAULT_INPUT_TOKENS = 200
DEFAULT_LATENCY_MS = 5_000.0
MIN_TASKS_FOR_ESTIMATE = 3
_PRIVACY_RANK = {PrivacyTier.PUBLIC: 0, PrivacyTier.INTERNAL: 1, PrivacyTier.CONFIDENTIAL: 2}


@dataclass(frozen=True)
class WorkloadClass:
    task: str
    band: str
    share: float
    in_tokens: float
    out_tokens: float

    @property
    def key(self) -> str:
        return f"{self.task}:{self.band}"


@dataclass(frozen=True)
class ModelProfile:
    model_id: int
    name: str
    display_name: str
    input_per_1m: float
    output_per_1m: float
    context_window: int
    privacy_rank: int
    quality: dict[str, float]          # task -> 0..10 (measured or estimated)
    quality_estimated: frozenset[str]  # tasks whose quality is the model's mean, not measured
    latency_p95: dict[str, float]      # task -> ms

    def cost(self, c: WorkloadClass) -> float:
        return (c.in_tokens * self.input_per_1m + c.out_tokens * self.output_per_1m) / 1e6


@dataclass
class OptimizationInputs:
    classes: list[WorkloadClass]
    models: list[ModelProfile]
    monthly_requests: int
    budget: float | None = None
    q_min: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_Q_MIN))
    latency_max_ms: float | None = None
    team_privacy_rank: int = 0
    traffic_source: str = "uniform prior"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        C, M = len(self.classes), len(self.models)
        self.cost = np.zeros((C, M))
        self.quality = np.zeros((C, M))
        self.latency = np.zeros((C, M))
        self.eligible = np.zeros((C, M), dtype=bool)
        self.reasons: dict[tuple[int, int], str] = {}
        for i, c in enumerate(self.classes):
            for j, m in enumerate(self.models):
                self.cost[i, j] = m.cost(c)
                self.quality[i, j] = m.quality.get(c.task, 0.0)
                self.latency[i, j] = m.latency_p95.get(c.task, DEFAULT_LATENCY_MS)
                reason = self._ineligible_reason(c, m)
                if reason:
                    self.reasons[(i, j)] = reason
                else:
                    self.eligible[i, j] = True
            if not self.eligible[i].any():
                # Never leave a class unserved: relax to the best-quality model(s) that pass
                # the non-quality constraints, and say so in the run notes
                ok = [j for j, m in enumerate(self.models) if c.task in m.quality
                      and self._ineligible_reason(c, m, check_quality=False) is None]
                if ok:
                    best = max(self.quality[i, j] for j in ok)
                    for j in ok:
                        if self.quality[i, j] == best:
                            self.eligible[i, j] = True
                            self.reasons.pop((i, j), None)
                    self.notes.append(f"{c.key}: no model reaches quality "
                                      f"{self.q_min[c.band]}; relaxed to best available "
                                      f"({best:.1f})")
                else:
                    self.notes.append(f"{c.key}: no eligible model at all")

    def _ineligible_reason(self, c: WorkloadClass, m: ModelProfile,
                           check_quality: bool = True) -> str | None:
        if c.task not in m.quality:
            return "no quality profile for this task"
        if m.privacy_rank < self.team_privacy_rank:
            return "privacy tier"
        if c.in_tokens + c.out_tokens > m.context_window:
            return "context window"
        lat = m.latency_p95.get(c.task, DEFAULT_LATENCY_MS)
        if self.latency_max_ms is not None and lat > self.latency_max_ms:
            return f"p95 latency {lat:.0f} ms > {self.latency_max_ms:.0f} ms"
        if check_quality and m.quality[c.task] < self.q_min[c.band]:
            return f"quality {m.quality[c.task]:.1f} < {self.q_min[c.band]:.1f}"
        return None

    @property
    def shares(self) -> np.ndarray:
        return np.array([c.share for c in self.classes])

    def evaluate(self, X: np.ndarray) -> tuple[float, float, float]:
        """(monthly cost USD, mean quality 0–10, mean p95 latency ms) for allocation X."""
        w = self.shares[:, None] * X
        return (float(self.monthly_requests * (w * self.cost).sum()),
                float((w * self.quality).sum()), float((w * self.latency).sum()))

    def single_model_allocation(self, j: int) -> np.ndarray | None:
        """All traffic to model j (the 'no optimisation' baseline), if it is allowed."""
        if not self.eligible[:, j].all():
            return None
        X = np.zeros(self.cost.shape)
        X[:, j] = 1.0
        return X


async def _model_profiles(db: AsyncSession) -> list[ModelProfile]:
    now = datetime.now(UTC)
    models = (await db.scalars(
        select(LLMModel).options(selectinload(LLMModel.pricing), selectinload(LLMModel.provider))
        .where(LLMModel.is_active).order_by(LLMModel.id))).all()
    perf = list(await db.scalars(select(ModelPerformance)))
    out = []
    for m in models:
        if not m.provider.is_active:
            continue
        price = next((p for p in m.pricing if p.effective_from <= now), None)
        if price is None:
            continue
        rows = [p for p in perf if p.model_id == m.id]
        quality = {p.task_type.value: p.quality_score for p in rows if p.quality_score is not None}
        estimated: set[str] = set()
        if len(quality) >= MIN_TASKS_FOR_ESTIMATE:
            mean_q = float(np.mean(list(quality.values())))
            for t in TASKS:
                if t not in quality:
                    quality[t] = mean_q
                    estimated.add(t)
        lats = {p.task_type.value: p.latency_p95_ms for p in rows if p.latency_p95_ms}
        fallback_lat = median(lats.values()) if lats else DEFAULT_LATENCY_MS
        latency = {t: lats.get(t, fallback_lat) for t in TASKS}
        out.append(ModelProfile(
            model_id=m.id, name=m.name, display_name=m.display_name,
            input_per_1m=float(price.input_per_1m), output_per_1m=float(price.output_per_1m),
            context_window=m.context_window, privacy_rank=_PRIVACY_RANK[m.max_privacy_tier],
            quality=quality, quality_estimated=frozenset(estimated), latency_p95=latency))
    return out


async def _workload_classes(db: AsyncSession, team: Team | None,
                            days: int) -> tuple[list[WorkloadClass], int, str]:
    since = datetime.now(UTC) - timedelta(days=days)
    q = (select(Request.task_type, Request.complexity_band, func.count(),
                func.percentile_cont(0.5).within_group(
                    func.coalesce(Request.input_tokens, Request.est_input_tokens)),
                func.percentile_cont(0.5).within_group(
                    func.coalesce(Request.output_tokens, Request.est_output_tokens)))
         .where(Request.created_at >= since, Request.task_type.is_not(None),
                Request.complexity_band.is_not(None),
                Request.status != RequestStatus.BLOCKED_BUDGET)
         .group_by(Request.task_type, Request.complexity_band))
    if team is not None:
        q = q.where(Request.team_id == team.id)
    rows = (await db.execute(q)).all()
    total = sum(r[2] for r in rows)
    if total == 0:
        n = len(TASKS) * len(BANDS)
        classes = [WorkloadClass(t, b, 1 / n, DEFAULT_INPUT_TOKENS, DEFAULT_OUTPUT_TOKENS[t])
                   for t in TASKS for b in BANDS]
        return classes, 0, "uniform prior (no traffic yet)"
    classes = [
        WorkloadClass(task.value, band.value, cnt / total,
                      float(in_med or DEFAULT_INPUT_TOKENS),
                      float(out_med or DEFAULT_OUTPUT_TOKENS[task.value]))
        for task, band, cnt, in_med, out_med in rows
    ]
    return classes, total, f"{total} requests over the last {days} days"


async def build_inputs(
    db: AsyncSession, *, team: Team | None = None, monthly_requests: int | None = None,
    budget: float | None = None, q_min: dict[str, float] | None = None,
    latency_max_ms: float | None = None, days: int = 30,
) -> OptimizationInputs:
    classes, observed, source = await _workload_classes(db, team, days)
    if monthly_requests is None:
        monthly_requests = max(1, round(observed * 30 / days)) if observed else 10_000
    if budget is None and team is not None:
        budget = float(team.monthly_budget)
    return OptimizationInputs(
        classes=classes, models=await _model_profiles(db), monthly_requests=monthly_requests,
        budget=budget, q_min={**DEFAULT_Q_MIN, **(q_min or {})}, latency_max_ms=latency_max_ms,
        team_privacy_rank=_PRIVACY_RANK[team.privacy_tier] if team else 0,
        traffic_source=source,
    )
