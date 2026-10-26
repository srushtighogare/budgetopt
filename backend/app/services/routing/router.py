"""Candidate selection: which models may serve a request, and in what order.

Hard constraints are applied first and every rejected model is recorded with a
reason (report, Ethics: a model must never be chosen just because it is cheap
if it violates a requirement):
  - model and provider active
  - model cleared for the team's privacy tier
  - prompt + requested output fits the model's context window

Ordering. With an explicit model the client's choice comes first and the
remaining eligible models follow as fallbacks. With `auto` the INTERIM strategy
is cheapest-eligible-first; phase 6 replaces it with the optimizer's allocation
policy for the request's task type and complexity.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import LLMModel, ModelPricing, PrivacyTier, Team
from app.services.pricing import compute_cost

AUTO = "auto"

_PRIVACY_RANK = {PrivacyTier.PUBLIC: 0, PrivacyTier.INTERNAL: 1, PrivacyTier.CONFIDENTIAL: 2}


@dataclass
class CandidateInfo:
    model_id: int
    model: str
    provider: str
    est_cost: Decimal | None
    rejected_reason: str | None = None

    def as_dict(self) -> dict:
        return {"model_id": self.model_id, "model": self.model, "provider": self.provider,
                "est_cost": str(self.est_cost) if self.est_cost is not None else None,
                "rejected_reason": self.rejected_reason}


@dataclass
class RoutingPlan:
    route: str
    strategy: str
    est_input_tokens: int
    est_output_tokens: int
    ordered: list[LLMModel] = field(default_factory=list)
    considered: list[CandidateInfo] = field(default_factory=list)
    baseline: LLMModel | None = None
    baseline_pricing: ModelPricing | None = None

    @property
    def rejected(self) -> list[CandidateInfo]:
        return [c for c in self.considered if c.rejected_reason]


def _current_price(model: LLMModel, now: datetime) -> ModelPricing | None:
    return next((p for p in model.pricing if p.effective_from <= now), None)


async def plan_route(
    db: AsyncSession, team: Team, route: str, est_input_tokens: int, est_output_tokens: int,
    max_tokens: int,
) -> RoutingPlan:
    """`est_*` come from workload analysis and drive cost estimates; `max_tokens`
    (the hard output cap) drives the context-window check."""
    now = datetime.now(UTC)
    est_output = est_output_tokens
    models = (await db.scalars(
        select(LLMModel)
        .options(selectinload(LLMModel.provider), selectinload(LLMModel.pricing))
        .order_by(LLMModel.id)
    )).all()

    plan = RoutingPlan(route=route, strategy="", est_input_tokens=est_input_tokens,
                       est_output_tokens=est_output)
    eligible: list[tuple[Decimal, LLMModel]] = []
    priced: list[tuple[Decimal, LLMModel, ModelPricing]] = []

    for m in models:
        price = _current_price(m, now)
        est = compute_cost(price, est_input_tokens, est_output) if price else None
        info = CandidateInfo(m.id, m.name, m.provider.name, est)
        if not (m.is_active and m.provider.is_active):
            info.rejected_reason = "inactive"
        elif price is None:
            info.rejected_reason = "no pricing in effect"
        elif _PRIVACY_RANK[m.max_privacy_tier] < _PRIVACY_RANK[team.privacy_tier]:
            info.rejected_reason = (f"privacy: cleared for {m.max_privacy_tier.value}, "
                                    f"team requires {team.privacy_tier.value}")
        elif est_input_tokens + min(max_tokens, m.max_output_tokens) > m.context_window:
            info.rejected_reason = (f"context: needs ~{est_input_tokens + max_tokens} tokens, "
                                    f"window is {m.context_window}")
        plan.considered.append(info)
        if price is not None and m.is_active and m.provider.is_active:
            priced.append((est, m, price))
        if info.rejected_reason is None:
            eligible.append((est, m))

    # Baseline for savings = what an un-optimised setup would use: the priciest model
    if priced:
        _, plan.baseline, plan.baseline_pricing = max(priced, key=lambda t: t[0])

    eligible.sort(key=lambda t: (t[0], t[1].id))
    by_cost = [m for _, m in eligible]
    if route == AUTO:
        plan.strategy = "cheapest eligible first (interim, until optimizer policies)"
        plan.ordered = by_cost
    else:
        requested = next((m for m in by_cost if m.name == route), None)
        if requested is None:
            plan.strategy = f"requested model '{route}' not eligible"
            plan.ordered = []
        else:
            plan.strategy = "requested model first, cheapest eligible as fallback"
            plan.ordered = [requested] + [m for m in by_cost if m.id != requested.id]
    return plan
