"""LLM integration layer (report architecture item 6).

`/v1/chat/completions` speaks the OpenAI Chat Completions format, so existing
applications can use BudgetOpt by changing only their base URL and API key:

    client = OpenAI(base_url="http://localhost:8000/v1", api_key="bo_...")
    client.chat.completions.create(model="auto", messages=[...])

`/api/v1/route` is the same pipeline for logged-in dashboard users (playground).
"""

import time
from decimal import Decimal

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.deps import get_accessible_team, get_api_key, get_current_user
from app.core.redis import get_redis
from app.db.models import ApiKey, LLMModel, Provider, RequestStatus, Role, Team, User
from app.db.session import get_db
from app.schemas.gateway import ChatCompletionIn, PlaygroundIn, RouteResultOut
from app.services import rate_limit
from app.services.providers.base import ChatMessage
from app.services.routing.pipeline import GatewayRequest, GatewayResult, UnknownModel, handle
from app.services.routing.router import AUTO

openai_router = APIRouter(prefix="/v1", tags=["gateway (OpenAI-compatible)"])
router = APIRouter(tags=["gateway"])

# Status code + OpenAI error type for each non-success outcome
_FAILURES = {
    RequestStatus.BLOCKED_BUDGET: (status.HTTP_402_PAYMENT_REQUIRED, "budget_exceeded"),
    RequestStatus.BLOCKED_CONSTRAINT: (status.HTTP_422_UNPROCESSABLE_CONTENT, "no_eligible_model"),
    RequestStatus.FAILED: (status.HTTP_502_BAD_GATEWAY, "upstream_failed"),
}


def openai_error(status_code: int, message: str, code: str,
                 headers: dict[str, str] | None = None, **extra) -> JSONResponse:
    return JSONResponse(
        status_code=status_code, headers=headers,
        content={"error": {"message": message, "type": code, "code": code, **extra}},
    )


def _money(d: Decimal | None) -> str | None:
    """Fixed-point string (never exponent notation like '0E-8')."""
    return None if d is None else format(d.normalize(), "f")


def _messages(body: ChatCompletionIn) -> list[ChatMessage]:
    return [ChatMessage("system" if m.role == "developer" else m.role, m.text())
            for m in body.messages]


async def _run(db: AsyncSession, redis: Redis, team: Team, body: ChatCompletionIn,
               subject: str, *, user_id: int | None = None, api_key_id: int | None = None,
               use_cache: bool = True) -> GatewayResult | JSONResponse:
    settings = get_settings()
    if body.stream:
        return openai_error(400, "Streaming is not supported yet; send stream=false.",
                            "unsupported_parameter")
    messages = _messages(body)
    if sum(len(m.content) for m in messages) > settings.max_prompt_chars:
        return openai_error(413, f"Prompt exceeds {settings.max_prompt_chars} characters.",
                            "prompt_too_large")

    rl = await rate_limit.check(redis, subject, settings.rate_limit_per_minute)
    if not rl.allowed:
        return openai_error(429, "Rate limit exceeded.", "rate_limit_exceeded",
                            headers={"Retry-After": str(rl.reset_s),
                                     "X-RateLimit-Limit": str(rl.limit)})

    try:
        return await handle(db, redis, GatewayRequest(
            team=team, messages=messages, route=body.model or AUTO,
            max_tokens=body.output_limit, temperature=body.temperature,
            use_cache=use_cache, user_id=user_id, api_key_id=api_key_id,
        ))
    except UnknownModel:
        return openai_error(404, f"Model '{body.model}' is not registered. Use 'auto' or "
                                 "GET /v1/models.", "model_not_found")


@openai_router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionIn, response: Response,
    key: ApiKey = Depends(get_api_key),
    cache_control: str | None = Header(default=None),
    db: AsyncSession = Depends(get_db), redis: Redis = Depends(get_redis),
):
    team = await db.get(Team, key.team_id)
    use_cache = "no-cache" not in (cache_control or "").lower()
    res = await _run(db, redis, team, body, f"key:{key.id}", api_key_id=key.id,
                     use_cache=use_cache)
    if isinstance(res, JSONResponse):
        return res

    headers = {"X-BudgetOpt-Request-Id": str(res.request_id),
               "X-BudgetOpt-Budget-State": res.budget_state}
    meta = {"request_id": res.request_id, "status": res.status.value,
            "cost": _money(res.cost), "baseline_cost": _money(res.baseline_cost),
            "savings": _money(res.savings), "budget_state": res.budget_state,
            "explanation": res.explanation, "workload": res.workload}
    if not res.ok:
        code, kind = _FAILURES[res.status]
        return openai_error(code, res.explanation, kind, headers=headers, budgetopt=meta)

    response.headers.update(headers)
    return {
        "id": f"chatcmpl-bo-{res.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": res.model,
        "choices": [{"index": 0, "finish_reason": res.finish_reason,
                     "message": {"role": "assistant", "content": res.text}}],
        "usage": {"prompt_tokens": res.input_tokens, "completion_tokens": res.output_tokens,
                  "total_tokens": res.input_tokens + res.output_tokens},
        "budgetopt": meta,
    }


@openai_router.get("/models")
async def list_gateway_models(
    _: ApiKey = Depends(get_api_key), db: AsyncSession = Depends(get_db)
) -> dict:
    rows = await db.execute(
        select(LLMModel.name, Provider.name).join(Provider)
        .where(LLMModel.is_active, Provider.is_active).order_by(LLMModel.id)
    )
    data = [{"id": AUTO, "object": "model", "owned_by": "budgetopt"}]
    data += [{"id": name, "object": "model", "owned_by": prov} for name, prov in rows]
    return {"object": "list", "data": data}


@router.post("/route", response_model=RouteResultOut)
async def playground_route(
    body: PlaygroundIn, user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db), redis: Redis = Depends(get_redis),
):
    team_id = body.team_id if body.team_id is not None else user.team_id
    if team_id is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT,
                            "team_id is required for users without a team")
    if user.role != Role.ORG_ADMIN and team_id != user.team_id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "You can only route for your own team")
    team = await get_accessible_team(team_id, user, db, action="route")

    res = await _run(db, redis, team, body, f"user:{user.id}", user_id=user.id,
                     use_cache=body.use_cache)
    if isinstance(res, JSONResponse):
        return res
    return RouteResultOut(
        request_id=res.request_id, status=res.status.value, text=res.text, model=res.model,
        input_tokens=res.input_tokens, output_tokens=res.output_tokens, cost=res.cost,
        baseline_cost=res.baseline_cost, savings=res.savings, latency_ms=res.latency_ms,
        budget_state=res.budget_state, explanation=res.explanation, attempts=res.attempts,
        workload=res.workload,
    )
