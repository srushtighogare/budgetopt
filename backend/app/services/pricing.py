"""Versioned model pricing and exact cost calculation."""

from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ModelPricing

ONE_MILLION = Decimal(1_000_000)
COST_QUANTUM = Decimal("0.00000001")


async def price_at(db: AsyncSession, model_id: int, at: datetime | None = None) -> ModelPricing:
    """Pricing row in effect for `model_id` at time `at` (default: now)."""
    at = at or datetime.now(UTC)
    row = await db.scalar(
        select(ModelPricing)
        .where(ModelPricing.model_id == model_id, ModelPricing.effective_from <= at)
        .order_by(ModelPricing.effective_from.desc())
        .limit(1)
    )
    if row is None:
        raise LookupError(f"no pricing in effect for model {model_id} at {at.isoformat()}")
    return row


def compute_cost(pricing: ModelPricing, input_tokens: int, output_tokens: int) -> Decimal:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    cost = (
        Decimal(input_tokens) * pricing.input_per_1m
        + Decimal(output_tokens) * pricing.output_per_1m
    ) / ONE_MILLION
    return cost.quantize(COST_QUANTUM, rounding=ROUND_HALF_UP)
