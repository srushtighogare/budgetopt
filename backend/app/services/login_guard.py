"""Brute-force protection for the login endpoint.

Failed attempts are counted in Redis per account (email) and per client IP over a
15-minute window. Past the limit, logins are refused with 429 until the window
expires, so password guessing is capped at a handful of tries per account however
many requests an attacker sends. A successful login clears that account's counter.

Like the gateway's rate limiter, the guard fails open if Redis is unreachable
(logged), so an infrastructure fault can't lock every user out.
"""

import logging
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import RedisError

log = logging.getLogger(__name__)

WINDOW_S = 15 * 60
MAX_FAILURES_PER_ACCOUNT = 5
MAX_FAILURES_PER_IP = 30


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    retry_after_s: int = 0


def _keys(email: str, ip: str) -> tuple[str, str]:
    return f"login:fail:acct:{email.lower()}", f"login:fail:ip:{ip}"


async def check(redis: Redis, email: str, ip: str) -> Verdict:
    acct, ipk = _keys(email, ip)
    try:
        async with redis.pipeline(transaction=False) as p:
            p.get(acct)
            p.get(ipk)
            p.ttl(acct)
            p.ttl(ipk)
            a, i, ta, ti = await p.execute()
    except RedisError as e:
        log.warning("login guard unavailable, failing open: %s", e)
        return Verdict(True)
    if a and int(a) >= MAX_FAILURES_PER_ACCOUNT:
        return Verdict(False, max(int(ta), 1))
    if i and int(i) >= MAX_FAILURES_PER_IP:
        return Verdict(False, max(int(ti), 1))
    return Verdict(True)


async def record_failure(redis: Redis, email: str, ip: str) -> None:
    try:
        async with redis.pipeline(transaction=True) as p:
            for key in _keys(email, ip):
                p.incr(key)
                p.expire(key, WINDOW_S, nx=True)  # window starts at the first failure
            await p.execute()
    except RedisError as e:
        log.warning("login guard could not record a failure: %s", e)


async def record_success(redis: Redis, email: str, ip: str) -> None:
    try:
        await redis.delete(_keys(email, ip)[0])
    except RedisError as e:
        log.warning("login guard could not reset a counter: %s", e)
