"""Exact-match response cache (Redis).

A cache hit returns a previous answer at zero LLM cost. The key covers
everything that changes the answer: team (no cross-team leakage), requested
model/route, the full message list and generation parameters. Only
deterministic requests (temperature unset or 0) are cached, since a client
asking for sampling expects a fresh answer.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.services.providers.base import ChatMessage

log = logging.getLogger(__name__)


@dataclass
class CachedResponse:
    text: str
    model_id: int
    model: str
    input_tokens: int
    output_tokens: int
    original_cost: str  # Decimal as string
    original_request_id: int


def is_cacheable(temperature: float | None) -> bool:
    return temperature is None or temperature == 0


def cache_key(team_id: int, route: str, messages: list[ChatMessage], max_tokens: int,
              temperature: float | None) -> str:
    material = json.dumps(
        {"team": team_id, "route": route, "max_tokens": max_tokens, "temperature": temperature,
         "messages": [[m.role, m.content] for m in messages]},
        sort_keys=True, ensure_ascii=False,
    )
    return "cache:exact:" + hashlib.sha256(material.encode()).hexdigest()


async def get(redis: Redis, key: str) -> CachedResponse | None:
    try:
        raw = await redis.get(key)
    except RedisError as e:  # cache is an optimisation; never fail a request over it
        log.warning("cache read failed: %s", e)
        return None
    return CachedResponse(**json.loads(raw)) if raw else None


async def put(redis: Redis, key: str, value: CachedResponse, ttl_s: int) -> None:
    try:
        await redis.set(key, json.dumps(asdict(value)), ex=ttl_s)
    except RedisError as e:
        log.warning("cache write failed: %s", e)
