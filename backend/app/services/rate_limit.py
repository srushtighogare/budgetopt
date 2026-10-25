"""Fixed-window rate limiting per caller (API key or user), backed by Redis.

INCR + EXPIRE on a key per (caller, minute) is atomic in Redis, so the limit
holds across multiple API workers. If Redis is unreachable the limiter fails
open (logs a warning) so an infrastructure fault doesn't take the gateway down;
budget enforcement, which lives in PostgreSQL, still applies.
"""

import logging
import time
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

WINDOW_S = 60


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_s: int


async def check(
    redis: Redis, subject: str, limit: int, now: float | None = None
) -> RateLimitResult:
    now = time.time() if now is None else now
    window = int(now // WINDOW_S)
    reset_s = WINDOW_S - int(now % WINDOW_S)
    key = f"ratelimit:{subject}:{window}"
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, WINDOW_S + 5)
            count, _ = await pipe.execute()
    except RedisError as e:
        log.warning("rate limiter unavailable, failing open: %s", e)
        return RateLimitResult(True, limit, limit, reset_s)
    return RateLimitResult(count <= limit, limit, max(0, limit - count), reset_s)
