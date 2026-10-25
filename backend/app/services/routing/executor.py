"""Fallback executor: tries an ordered list of candidate models until one succeeds.

The router (phase 6) decides the order; this module only executes it, records
every attempt for the explanation/audit trail, and prices the successful call
with the pricing version in effect at that moment.
"""

from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import LLMModel
from app.services.pricing import compute_cost, price_at
from app.services.providers.base import ChatMessage, Completion, InvalidRequest, ProviderError
from app.services.providers.registry import AdapterFactory, build_adapter
from app.services.routing.breaker import CircuitBreaker, default_breaker


@dataclass(frozen=True)
class Attempt:
    model_id: int
    model: str
    provider: str
    outcome: str  # success | error | skipped_circuit_open | skipped_inactive
    error_type: str | None = None
    error: str | None = None
    latency_ms: float | None = None


@dataclass
class ExecutionResult:
    completion: Completion | None
    model: LLMModel | None
    cost: Decimal = Decimal(0)
    attempts: list[Attempt] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.completion is not None

    @property
    def fallback_used(self) -> bool:
        """True if the model that answered was not the first candidate."""
        return self.succeeded and self.attempts[0].outcome != "success"


async def execute_with_fallback(
    db: AsyncSession,
    candidates: list[LLMModel],
    messages: list[ChatMessage],
    *,
    max_tokens: int = 1024,
    temperature: float | None = None,
    timeout: float = 60.0,
    breaker: CircuitBreaker = default_breaker,
    adapter_factory: AdapterFactory = build_adapter,
) -> ExecutionResult:
    """`candidates` must have `.provider` loaded."""
    result = ExecutionResult(completion=None, model=None)

    for model in candidates:
        base = {"model_id": model.id, "model": model.name, "provider": model.provider.name}

        if not (model.is_active and model.provider.is_active):
            result.attempts.append(Attempt(**base, outcome="skipped_inactive"))
            continue
        if not breaker.allow(model.id):
            result.attempts.append(Attempt(**base, outcome="skipped_circuit_open"))
            continue

        try:
            adapter = adapter_factory(model.provider)
            completion = await adapter.generate(
                model.name, messages, max_tokens=min(max_tokens, model.max_output_tokens),
                temperature=temperature, timeout=timeout,
            )
        except ProviderError as e:
            # A request-specific rejection says nothing about the model's health
            if not isinstance(e, InvalidRequest):
                breaker.record_failure(model.id)
            result.attempts.append(
                Attempt(**base, outcome="error", error_type=type(e).__name__, error=str(e))
            )
            continue

        breaker.record_success(model.id)
        pricing = await price_at(db, model.id)
        result.completion = completion
        result.model = model
        result.cost = compute_cost(pricing, completion.input_tokens, completion.output_tokens)
        result.attempts.append(Attempt(**base, outcome="success", latency_ms=completion.latency_ms))
        return result

    return result
