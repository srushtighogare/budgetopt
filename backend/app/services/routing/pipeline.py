"""End-to-end handling of one LLM request through BudgetOpt.

    budget gate -> exact cache -> route planning (constraints) -> fallback execution
                -> pricing -> request + routing-decision log -> cache store

Every outcome, including blocked requests, is logged with a human-readable
explanation so admins can audit why a model was (or wasn't) used.

Note on concurrency: the budget check reads the month's spend before the call,
so a burst of simultaneous requests can overshoot a budget by at most those
in-flight requests; the next request after that is blocked.
"""

import random
from dataclasses import dataclass, field
from decimal import Decimal

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import hash_prompt
from app.db.models import AllocationPolicy, Request, RequestStatus, RoutingDecision, Team
from app.services import cache
from app.services.alerts import budget_alerts
from app.services.budget import BudgetState, budget_status
from app.services.pricing import compute_cost
from app.services.privacy import redact
from app.services.providers.base import ChatMessage
from app.services.providers.registry import AdapterFactory, build_adapter
from app.services.routing.breaker import CircuitBreaker, default_breaker
from app.services.routing.executor import Attempt, execute_with_fallback
from app.services.routing.router import AUTO, RoutingPlan, plan_route
from app.services.workload.analyzer import WorkloadProfile, analyze


class UnknownModel(Exception):
    pass


@dataclass
class GatewayRequest:
    team: Team
    messages: list[ChatMessage]
    route: str = AUTO
    max_tokens: int | None = None
    temperature: float | None = None
    use_cache: bool = True
    user_id: int | None = None
    api_key_id: int | None = None


@dataclass
class GatewayResult:
    status: RequestStatus
    request_id: int
    explanation: str
    budget_state: str
    text: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: Decimal = Decimal(0)
    baseline_cost: Decimal | None = None
    savings: Decimal | None = None
    latency_ms: float | None = None
    finish_reason: str = "stop"
    attempts: list[dict] = field(default_factory=list)
    workload: dict | None = None

    @property
    def ok(self) -> bool:
        return self.status in (RequestStatus.SUCCESS, RequestStatus.FALLBACK_SUCCESS,
                               RequestStatus.CACHE_HIT)


def _prompt_fields(team: Team, messages: list[ChatMessage]) -> dict:
    joined = "\n".join(f"{m.role}: {m.content}" for m in messages)
    return {"prompt_hash": hash_prompt(joined),
            "prompt_text": redact(joined) if team.store_prompts else None}


def _attempt_dict(a: Attempt) -> dict:
    return {k: v for k, v in a.__dict__.items() if v is not None}


async def _log(db: AsyncSession, req: GatewayRequest, *, status: RequestStatus,
               budget_state: str, explanation: str, plan: RoutingPlan | None = None,
               model_id: int | None = None, input_tokens: int | None = None,
               output_tokens: int | None = None, cost: Decimal = Decimal(0),
               latency_ms: float | None = None, baseline_cost: Decimal | None = None,
               savings: Decimal | None = None, attempts: list[dict] | None = None,
               fallback_used: bool = False, profile: WorkloadProfile | None = None) -> int:
    row = Request(
        team_id=req.team.id, user_id=req.user_id, api_key_id=req.api_key_id,
        **_prompt_fields(req.team, req.messages),
        task_type=profile.task_type if profile else None,
        complexity=profile.complexity if profile else None,
        complexity_band=profile.complexity_band if profile else None,
        est_input_tokens=profile.est_input_tokens if profile else None,
        est_output_tokens=profile.est_output_tokens if profile else None,
        model_id=model_id, input_tokens=input_tokens, output_tokens=output_tokens,
        cost=cost, latency_ms=latency_ms, status=status,
    )
    db.add(row)
    await db.flush()
    db.add(RoutingDecision(
        request_id=row.id, budget_state=budget_state,
        policy_id=plan.policy_id if plan else None,
        candidates={"considered": [c.as_dict() for c in plan.considered] if plan else [],
                    "attempts": attempts or [], "strategy": plan.strategy if plan else None,
                    "workload": _workload_dict(profile)},
        fallback_used=fallback_used,
        baseline_model_id=plan.baseline.id if plan and plan.baseline else None,
        baseline_cost=baseline_cost, savings=savings, explanation=explanation,
    ))
    await db.commit()
    return row.id


# Provider finish reasons -> OpenAI's vocabulary
_FINISH = {"stop": "stop", "end_turn": "stop", "STOP": "stop", "stop_sequence": "stop",
           "length": "length", "max_tokens": "length", "MAX_TOKENS": "length",
           "content_filter": "content_filter", "SAFETY": "content_filter"}


def normalize_finish(reason: str | None) -> str:
    return _FINISH.get(reason or "stop", "stop")


def _workload_dict(p: WorkloadProfile | None) -> dict | None:
    if p is None:
        return None
    return {"task_type": p.task_type.value, "task_confidence": p.task_confidence,
            "task_alternatives": p.task_alternatives, "complexity": p.complexity,
            "complexity_band": p.complexity_band.value, "est_input_tokens": p.est_input_tokens,
            "est_output_tokens": p.est_output_tokens, "analysis_ms": p.analysis_ms,
            "sources": p.sources}


def _fmt(d: Decimal | None) -> str:
    return "n/a" if d is None else f"${d:.8f}"


async def handle(
    db: AsyncSession, redis: Redis, req: GatewayRequest, *,
    adapter_factory: AdapterFactory | None = None,
    breaker: CircuitBreaker | None = None,
    rng: random.Random | None = None,
) -> GatewayResult:
    settings = get_settings()
    adapter_factory = adapter_factory or build_adapter
    breaker = breaker or default_breaker
    max_tokens = req.max_tokens or settings.default_max_tokens

    # 1. Budget gate: an over-budget team never reaches an LLM, so blocking is free
    budget = await budget_status(db, req.team)
    state = budget.state.value
    await budget_alerts(db, req.team, budget)  # once per team, month and threshold
    if budget.state == BudgetState.HARD_BLOCK:
        why = (f"Blocked before any LLM call: month-to-date spend {_fmt(budget.spend)} "
               f"has reached the team budget {_fmt(budget.monthly_budget)}.")
        rid = await _log(db, req, status=RequestStatus.BLOCKED_BUDGET, budget_state=state,
                         explanation=why)
        return GatewayResult(RequestStatus.BLOCKED_BUDGET, rid, why, state)

    # 2. Workload analysis (~1 ms): task type, complexity, token estimates
    profile = analyze(req.messages)
    workload = _workload_dict(profile)

    # 3. Exact cache
    key = cache.cache_key(req.team.id, req.route, req.messages, max_tokens, req.temperature)
    use_cache = req.use_cache and cache.is_cacheable(req.temperature)
    if use_cache and (hit := await cache.get(redis, key)):
        saved = Decimal(hit.original_cost)
        why = (f"Served from cache (identical request #{hit.original_request_id} answered by "
               f"{hit.model}); no LLM call, saved {_fmt(saved)}.")
        rid = await _log(db, req, status=RequestStatus.CACHE_HIT, budget_state=state,
                         explanation=why, model_id=hit.model_id, input_tokens=0,
                         output_tokens=0, baseline_cost=saved, savings=saved, profile=profile)
        return GatewayResult(RequestStatus.CACHE_HIT, rid, why, state, text=hit.text,
                             model=hit.model, baseline_cost=saved, savings=saved,
                             workload=workload)

    # 4. Plan: constraints + ordering (the team's optimizer policy; its cost-saver policy
    #    once the budget alert threshold is crossed)
    policy_id = req.team.policy_id
    if budget.state == BudgetState.SOFT_ALERT and req.team.savings_policy_id:
        policy_id = req.team.savings_policy_id
    policy = await db.get(AllocationPolicy, policy_id) if policy_id else None
    plan = await plan_route(
        db, req.team, req.route, profile.est_input_tokens,
        min(max_tokens, profile.est_output_tokens), max_tokens, policy=policy,
        class_key=f"{profile.task_type.value}:{profile.complexity_band.value}", rng=rng)
    if req.route != AUTO and all(c.model != req.route for c in plan.considered):
        raise UnknownModel(req.route)
    rejected = "; ".join(f"{c.model} ({c.rejected_reason})" for c in plan.rejected)
    if not plan.ordered:
        why = (f"{profile.describe()} No model satisfies this request's constraints. "
               f"Strategy: {plan.strategy}. Rejected: {rejected or 'none'}.")
        rid = await _log(db, req, status=RequestStatus.BLOCKED_CONSTRAINT, budget_state=state,
                         explanation=why, plan=plan, profile=profile)
        return GatewayResult(RequestStatus.BLOCKED_CONSTRAINT, rid, why, state,
                             workload=workload)

    # 5. Execute with fallback
    res = await execute_with_fallback(
        db, plan.ordered, req.messages, max_tokens=max_tokens, temperature=req.temperature,
        timeout=settings.provider_timeout_s, breaker=breaker, adapter_factory=adapter_factory,
    )
    attempts = [_attempt_dict(a) for a in res.attempts]
    failed = "; ".join(f"{a.model}: {a.error_type or a.outcome}"
                       for a in res.attempts if a.outcome != "success")

    if not res.succeeded:
        why = (f"{profile.describe()} All eligible models failed ({failed}). "
               f"Strategy: {plan.strategy}.")
        rid = await _log(db, req, status=RequestStatus.FAILED, budget_state=state,
                         explanation=why, plan=plan, attempts=attempts, profile=profile)
        return GatewayResult(RequestStatus.FAILED, rid, why, state, attempts=attempts,
                             workload=workload)

    # 6. Price against the baseline and explain
    c, model = res.completion, res.model
    baseline_cost = (compute_cost(plan.baseline_pricing, c.input_tokens, c.output_tokens)
                     if plan.baseline_pricing else None)
    savings = baseline_cost - res.cost if baseline_cost is not None else None
    status = RequestStatus.FALLBACK_SUCCESS if res.fallback_used else RequestStatus.SUCCESS

    parts = [profile.describe(), f"Route '{req.route}', strategy: {plan.strategy}."]
    if rejected:
        parts.append(f"Rejected by constraints: {rejected}.")
    if res.fallback_used:
        parts.append(f"Fallback used after: {failed}.")
    parts.append(f"Answered by {model.display_name} ({c.input_tokens} in / "
                 f"{c.output_tokens} out tokens) for {_fmt(res.cost)}.")
    if normalize_finish(c.finish_reason) == "length":
        parts.append("Output was cut off at the max_tokens limit.")
    if plan.baseline and savings is not None:
        pct = f" ({savings / baseline_cost:.0%})" if baseline_cost else ""
        parts.append(f"Baseline {plan.baseline.display_name} would cost {_fmt(baseline_cost)}; "
                     f"saved {_fmt(savings)}{pct}.")
    if budget.state == BudgetState.SOFT_ALERT:
        parts.append(f"Budget alert: {budget.usage_ratio:.0%} of monthly budget used.")
    why = " ".join(parts)

    rid = await _log(db, req, status=status, budget_state=state, explanation=why, plan=plan,
                     model_id=model.id, input_tokens=c.input_tokens,
                     output_tokens=c.output_tokens, cost=res.cost, latency_ms=c.latency_ms,
                     baseline_cost=baseline_cost, savings=savings, attempts=attempts,
                     fallback_used=res.fallback_used, profile=profile)

    if use_cache:
        await cache.put(redis, key, cache.CachedResponse(
            text=c.text, model_id=model.id, model=model.name, input_tokens=c.input_tokens,
            output_tokens=c.output_tokens, original_cost=str(res.cost), original_request_id=rid,
        ), settings.cache_ttl_s)

    return GatewayResult(
        status, rid, why, state, text=c.text, model=model.name, input_tokens=c.input_tokens,
        output_tokens=c.output_tokens, cost=res.cost, baseline_cost=baseline_cost,
        savings=savings, latency_ms=c.latency_ms,
        finish_reason=normalize_finish(c.finish_reason), attempts=attempts, workload=workload,
    )
