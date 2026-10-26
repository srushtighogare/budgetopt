"""ORM schema for BudgetOpt.

Money is stored as Numeric(14, 8) USD so per-request costs of fractions of a cent
are exact. Model prices are versioned (`model_pricing.effective_from`) so that
historical request costs stay correct after a provider changes its pricing.
"""

import enum
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

Money = Numeric(14, 8)
JsonType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JsonType, list[Any]: JsonType}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


# --- Enums ---

class Role(enum.StrEnum):
    ORG_ADMIN = "org_admin"
    TEAM_MANAGER = "team_manager"
    MEMBER = "member"


class TaskType(enum.StrEnum):
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    SIMPLE_QA = "simple_qa"
    EXTRACTION = "extraction"
    CREATIVE_WRITING = "creative_writing"
    CODING = "coding"
    REASONING_MATH = "reasoning_math"
    ANALYSIS = "analysis"


class ComplexityBand(enum.StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PrivacyTier(enum.StrEnum):
    """Data-handling level a model/provider is cleared for (ordered low → high)."""

    PUBLIC = "public"          # any provider
    INTERNAL = "internal"      # providers with a no-training / retention agreement
    CONFIDENTIAL = "confidential"  # self-hosted or contractually restricted only


class RequestStatus(enum.StrEnum):
    SUCCESS = "success"
    FALLBACK_SUCCESS = "fallback_success"
    CACHE_HIT = "cache_hit"
    BLOCKED_BUDGET = "blocked_budget"
    BLOCKED_CONSTRAINT = "blocked_constraint"
    FAILED = "failed"


class AlertKind(enum.StrEnum):
    BUDGET_SOFT = "budget_soft"
    BUDGET_HARD = "budget_hard"
    FORECAST_OVERRUN = "forecast_overrun"
    ANOMALY = "anomaly"


def _enum(e: type[enum.Enum], name: str) -> Enum:
    return Enum(e, name=name, values_callable=lambda x: [m.value for m in x])


# --- Organization, teams, users ---

class Organization(TimestampMixin, Base):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)

    teams: Mapped[list["Team"]] = relationship(back_populates="organization")


class Team(TimestampMixin, Base):
    __tablename__ = "teams"
    __table_args__ = (UniqueConstraint("organization_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    name: Mapped[str] = mapped_column(String(200))
    monthly_budget: Mapped[Decimal] = mapped_column(Money)
    # Fraction of budget at which routing switches to the cost-saver policy
    alert_threshold: Mapped[float] = mapped_column(Float, default=0.8)
    privacy_tier: Mapped[PrivacyTier] = mapped_column(
        _enum(PrivacyTier, "privacy_tier"), default=PrivacyTier.PUBLIC
    )
    # If false (default) only a SHA-256 hash of each prompt is stored
    store_prompts: Mapped[bool] = mapped_column(Boolean, default=False)
    # use_alter breaks the teams -> policies -> optimization_runs -> teams FK cycle
    policy_id: Mapped[int | None] = mapped_column(
        ForeignKey("allocation_policies.id", use_alter=True, name="fk_teams_policy_id")
    )
    savings_policy_id: Mapped[int | None] = mapped_column(
        ForeignKey("allocation_policies.id", use_alter=True, name="fk_teams_savings_policy_id")
    )

    organization: Mapped[Organization] = relationship(back_populates="teams")
    users: Mapped[list["User"]] = relationship(back_populates="team")


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(200))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[Role] = mapped_column(_enum(Role, "user_role"), default=Role.MEMBER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    team: Mapped[Team | None] = relationship(back_populates="users")


class ApiKey(TimestampMixin, Base):
    """Key an application uses to call the gateway on behalf of a team."""

    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(100))
    prefix: Mapped[str] = mapped_column(String(16))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# --- Model & pricing layer ---

class Provider(TimestampMixin, Base):
    __tablename__ = "providers"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)  # groq, google, openai, anthropic
    encrypted_api_key: Mapped[str | None] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    models: Mapped[list["LLMModel"]] = relationship(back_populates="provider")


class LLMModel(TimestampMixin, Base):
    __tablename__ = "llm_models"
    __table_args__ = (UniqueConstraint("provider_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("providers.id"))
    name: Mapped[str] = mapped_column(String(100))
    display_name: Mapped[str] = mapped_column(String(100))
    context_window: Mapped[int] = mapped_column(Integer)
    max_output_tokens: Mapped[int] = mapped_column(Integer)
    max_privacy_tier: Mapped[PrivacyTier] = mapped_column(
        _enum(PrivacyTier, "privacy_tier"), default=PrivacyTier.PUBLIC
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    provider: Mapped[Provider] = relationship(back_populates="models")
    pricing: Mapped[list["ModelPricing"]] = relationship(
        back_populates="model", order_by="ModelPricing.effective_from.desc()"
    )


class ModelPricing(TimestampMixin, Base):
    __tablename__ = "model_pricing"

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("llm_models.id"), index=True)
    input_per_1m: Mapped[Decimal] = mapped_column(Money)
    output_per_1m: Mapped[Decimal] = mapped_column(Money)
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    model: Mapped[LLMModel] = relationship(back_populates="pricing")


class ModelPerformance(Base):
    """Measured quality (offline LLM-as-judge) and latency (live EWMA) per task type."""

    __tablename__ = "model_performance"
    __table_args__ = (UniqueConstraint("model_id", "task_type"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("llm_models.id"))
    task_type: Mapped[TaskType] = mapped_column(_enum(TaskType, "task_type"))
    quality_score: Mapped[float | None] = mapped_column(Float)  # 0–10
    quality_samples: Mapped[int] = mapped_column(Integer, default=0)
    latency_p50_ms: Mapped[float | None] = mapped_column(Float)
    latency_p95_ms: Mapped[float | None] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ProfilingResult(TimestampMixin, Base):
    """One evaluated answer from the offline quality profiling harness.

    `model_performance` is the per-(model, task) aggregate of these rows, so every
    quality number shown in the dashboard can be traced to individual judged answers.
    """

    __tablename__ = "profiling_results"
    __table_args__ = (UniqueConstraint("model_id", "item_id"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("llm_models.id"), index=True)
    item_id: Mapped[str] = mapped_column(String(50))  # id in the fixed profiling set
    task_type: Mapped[TaskType] = mapped_column(_enum(TaskType, "task_type"))
    score: Mapped[float | None] = mapped_column(Float)  # 0-10
    correct: Mapped[bool | None] = mapped_column(Boolean)  # gold-answer check, if any
    judge: Mapped[str | None] = mapped_column(String(100))
    judge_reason: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    cost: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))
    answer: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)


# --- Optimization ---

class OptimizationRun(TimestampMixin, Base):
    __tablename__ = "optimization_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int | None] = mapped_column(ForeignKey("teams.id"))
    method: Mapped[str] = mapped_column(String(20))  # nsga2 | ilp
    inputs: Mapped[dict[str, Any]]
    pareto_front: Mapped[list[Any] | None]
    selected_index: Mapped[int | None] = mapped_column(Integer)
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))


class AllocationPolicy(TimestampMixin, Base):
    """Maps each (task_type, complexity_band) to an ordered list of model ids.

    `rules` = {"coding:high": [model_id, fallback_id, ...], ...}
    """

    __tablename__ = "allocation_policies"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    optimization_run_id: Mapped[int | None] = mapped_column(ForeignKey("optimization_runs.id"))
    rules: Mapped[dict[str, Any]]
    expected_metrics: Mapped[dict[str, Any] | None]


# --- Requests & explainability ---

class Request(TimestampMixin, Base):
    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    api_key_id: Mapped[int | None] = mapped_column(ForeignKey("api_keys.id"))
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
    prompt_text: Mapped[str | None] = mapped_column(Text)  # only if team.store_prompts
    task_type: Mapped[TaskType | None] = mapped_column(_enum(TaskType, "task_type"))
    complexity: Mapped[float | None] = mapped_column(Float)
    complexity_band: Mapped[ComplexityBand | None] = mapped_column(
        _enum(ComplexityBand, "complexity_band")
    )
    est_input_tokens: Mapped[int | None] = mapped_column(Integer)
    est_output_tokens: Mapped[int | None] = mapped_column(Integer)
    model_id: Mapped[int | None] = mapped_column(ForeignKey("llm_models.id"))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cost: Mapped[Decimal] = mapped_column(Money, default=Decimal(0))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    status: Mapped[RequestStatus] = mapped_column(_enum(RequestStatus, "request_status"))

    decision: Mapped["RoutingDecision | None"] = relationship(back_populates="request")


class RoutingDecision(TimestampMixin, Base):
    __tablename__ = "routing_decisions"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("requests.id"), unique=True)
    policy_id: Mapped[int | None] = mapped_column(ForeignKey("allocation_policies.id"))
    budget_state: Mapped[str] = mapped_column(String(20))
    # {"strategy": str, "considered": [{model_id, est_cost, rejected_reason}], "attempts": [...]}
    candidates: Mapped[dict[str, Any]]
    fallback_used: Mapped[bool] = mapped_column(Boolean, default=False)
    baseline_model_id: Mapped[int | None] = mapped_column(ForeignKey("llm_models.id"))
    baseline_cost: Mapped[Decimal | None] = mapped_column(Money)
    savings: Mapped[Decimal | None] = mapped_column(Money)
    explanation: Mapped[str] = mapped_column(Text)

    request: Mapped[Request] = relationship(back_populates="decision")


EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5


class SemanticCacheEntry(TimestampMixin, Base):
    """A cached answer findable by meaning, not only by exact text (pgvector HNSW index).

    `context_key` hashes everything except the last user message (system prompt, earlier
    turns, route and generation settings); a hit requires it to match exactly.
    """

    __tablename__ = "semantic_cache_entries"
    __table_args__ = (
        Index("ix_semantic_cache_embedding", "embedding", postgresql_using="hnsw",
              postgresql_ops={"embedding": "vector_cosine_ops"}),
        Index("ix_semantic_cache_lookup", "team_id", "context_key"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"))
    context_key: Mapped[str] = mapped_column(String(64))
    task_type: Mapped[TaskType] = mapped_column(_enum(TaskType, "task_type"))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    numbers: Mapped[str] = mapped_column(String(500), default="")  # sorted numbers in prompt
    # Fernet-encrypted last user message, needed only by the verifier; expires with the entry
    prompt_encrypted: Mapped[str] = mapped_column(Text)
    response_text: Mapped[str] = mapped_column(Text)
    model_id: Mapped[int] = mapped_column(ForeignKey("llm_models.id"))
    model_name: Mapped[str] = mapped_column(String(100))
    original_cost: Mapped[Decimal] = mapped_column(Money)
    original_request_id: Mapped[int] = mapped_column(ForeignKey("requests.id"))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    hits: Mapped[int] = mapped_column(Integer, default=0)


# --- Budget monitoring ---

class Forecast(TimestampMixin, Base):
    __tablename__ = "forecasts"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    method: Mapped[str] = mapped_column(String(30))
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date] = mapped_column(Date)
    projected_spend: Mapped[Decimal] = mapped_column(Money)
    lower: Mapped[Decimal] = mapped_column(Money)
    upper: Mapped[Decimal] = mapped_column(Money)
    overrun_date: Mapped[date | None] = mapped_column(Date)
    overrun_probability: Mapped[float | None] = mapped_column(Float)
    daily: Mapped[list[Any]]  # [{date, actual, mean, lower, upper, cumulative_mean}]
    backtest: Mapped[dict[str, Any] | None]


class Alert(TimestampMixin, Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), index=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    kind: Mapped[AlertKind] = mapped_column(_enum(AlertKind, "alert_kind"))
    message: Mapped[str] = mapped_column(Text)
    details: Mapped[dict[str, Any] | None]
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)


class AuditEvent(TimestampMixin, Base):
    """Who viewed or changed financial / configuration data."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(100))
    resource: Mapped[str] = mapped_column(String(100))
    resource_id: Mapped[str | None] = mapped_column(String(50))
    allowed: Mapped[bool] = mapped_column(Boolean, default=True)
    details: Mapped[dict[str, Any] | None]
