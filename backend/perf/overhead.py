"""Routing-overhead load test: how much latency does BudgetOpt itself add per request?

Runs the real FastAPI app in-process against the real PostgreSQL and Redis, with the
real task classifier, complexity scoring, routing, logging and (optionally) the
semantic-cache embedding, but with an LLM adapter that answers instantly. Whatever
latency is measured is therefore BudgetOpt's own overhead on top of the provider call.

Scenarios: cache misses (every prompt unique) sequentially and with concurrency, and
exact-cache hits. Uses a temporary "Load test" team that is deleted afterwards.

Usage (dev DB running):  cd backend && .venv/bin/python -m perf.overhead [-n 400] [-c 20]
Writes ml/eval/overhead_report.md and overhead_metrics.json.
"""

import argparse
import asyncio
import json
import statistics
import time
from decimal import Decimal
from pathlib import Path

import numpy as np
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.security import generate_api_key
from app.db.models import (
    ApiKey,
    Organization,
    Request,
    RoutingDecision,
    SemanticCacheEntry,
    Team,
    User,
)
from app.db.session import SessionLocal
from app.main import app
from app.services import semantic_cache
from app.services.providers.base import Completion, ProviderAdapter
from app.services.routing import pipeline
from app.services.routing.breaker import CircuitBreaker
from app.services.workload.analyzer import warm_up

TEAM = "Load test (temporary)"
OUT = Path(__file__).resolve().parents[2] / "ml" / "eval"
PROMPTS = [
    "What is the capital of {x}?", "Translate 'good morning {x}' into French.",
    "Write a Python function that returns the {x}th Fibonacci number.",
    "Summarize this note in one sentence: the {x} team shipped the release on time.",
    "Compare {x} and its main competitor for a small business.",
    "If a train travels {x} km in 2 hours, what is its average speed?",
]


class Instant(ProviderAdapter):
    def __init__(self):
        super().__init__(api_key="x")

    async def generate(self, model, messages, **kw):
        return Completion("ok", 50, 20, 0.0)

    def build_request(self, *a):  # pragma: no cover
        raise NotImplementedError

    def parse_response(self, *a):  # pragma: no cover
        raise NotImplementedError


async def setup() -> tuple[int, str]:
    async with SessionLocal() as db:
        org = await db.scalar(select(Organization).order_by(Organization.id))
        admin = await db.scalar(select(User).order_by(User.id))
        team = Team(organization_id=org.id, name=TEAM, monthly_budget=Decimal(1_000_000))
        db.add(team)
        await db.flush()
        key, prefix, key_hash = generate_api_key()
        db.add(ApiKey(team_id=team.id, created_by=admin.id, name="load", prefix=prefix,
                      key_hash=key_hash))
        await db.commit()
        return team.id, key


async def teardown(team_id: int) -> None:
    async with SessionLocal() as db:
        ids = select(Request.id).where(Request.team_id == team_id)
        await db.execute(delete(RoutingDecision).where(RoutingDecision.request_id.in_(ids)))
        await db.execute(delete(SemanticCacheEntry).where(SemanticCacheEntry.team_id == team_id))
        await db.execute(delete(Request).where(Request.team_id == team_id))
        await db.execute(delete(ApiKey).where(ApiKey.team_id == team_id))
        await db.execute(delete(Team).where(Team.id == team_id))
        await db.commit()


def pct(xs: list[float]) -> dict:
    a = np.array(xs)
    return {"n": len(xs), "p50_ms": round(float(np.percentile(a, 50)), 2),
            "p95_ms": round(float(np.percentile(a, 95)), 2),
            "p99_ms": round(float(np.percentile(a, 99)), 2),
            "mean_ms": round(statistics.fmean(xs), 2)}


async def run(n: int, concurrency: int, semantic: bool) -> dict:
    pipeline.build_adapter = lambda provider: Instant()
    pipeline.default_breaker = CircuitBreaker()
    get_settings().rate_limit_per_minute = 10**9
    get_settings().semantic_cache_enabled = semantic
    warm_up()
    if semantic:
        semantic_cache.warm_up()
    team_id, key = await setup()
    headers = {"Authorization": f"Bearer {key}"}
    results: dict[str, dict] = {}
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://bench",
                               timeout=60) as client:
            async def one(prompt: str) -> tuple[float, float | None]:
                t0 = time.perf_counter()
                r = await client.post("/v1/chat/completions", headers=headers, json={
                    "messages": [{"role": "user", "content": prompt}]})
                elapsed = (time.perf_counter() - t0) * 1000
                assert r.status_code == 200, r.text
                return elapsed, r.json()["budgetopt"]["workload"]["analysis_ms"]

            # warm-up (connection pools, model caches)
            for i in range(10):
                await one(f"warm-up prompt {i}")

            prompts = [PROMPTS[i % len(PROMPTS)].format(x=f"item-{i}") for i in range(n)]
            seq = [await one(p) for p in prompts]
            results["cache miss, sequential"] = {**pct([s[0] for s in seq]),
                                                 "analysis": pct([s[1] for s in seq])}

            sem = asyncio.Semaphore(concurrency)

            async def guarded(p):
                async with sem:
                    return await one(p)
            t0 = time.perf_counter()
            conc = await asyncio.gather(*(guarded(p + " (c)") for p in prompts))
            wall = time.perf_counter() - t0
            results[f"cache miss, {concurrency} concurrent"] = {
                **pct([c[0] for c in conc]), "throughput_rps": round(n / wall, 1)}

            hits = [await one(prompts[i % 50]) for i in range(min(n, 200))]
            results["exact cache hit, sequential"] = pct([h[0] for h in hits])
    finally:
        await teardown(team_id)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=400)
    ap.add_argument("-c", type=int, default=20)
    args = ap.parse_args()
    async def both() -> dict:  # one event loop: the DB/Redis pools are bound to it
        return {"without semantic cache": await run(args.n, args.c, semantic=False),
                "with semantic cache": await run(args.n, args.c, semantic=True)}

    out = asyncio.run(both())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "overhead_metrics.json").write_text(json.dumps(out, indent=2))
    rows = []
    for mode, res in out.items():
        for name, r in res.items():
            extra = f"{r['throughput_rps']} req/s" if "throughput_rps" in r else ""
            rows.append(f"| {mode} | {name} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | "
                        f"{extra} |")
    seq = out["with semantic cache"]["cache miss, sequential"]
    (OUT / "overhead_report.md").write_text(f"""# Routing overhead

BudgetOpt's own added latency per request (the LLM call is replaced by an instant answer;
real PostgreSQL, Redis, classifier, routing and logging). Target from the design: p95 < 50 ms.

| Mode | Scenario | p50 ms | p95 ms | p99 ms | Throughput |
|---|---|---|---|---|---|
{chr(10).join(rows)}

Workload analysis alone (classifier + complexity + token estimates):
p50 {seq['analysis']['p50_ms']} ms, p95 {seq['analysis']['p95_ms']} ms.
Measured in-process on a development laptop, single API worker.

Under concurrency the per-request latency includes queueing: one worker serves requests one
event-loop slice at a time, so with 20 in flight each waits for the others. The throughput
column is the capacity figure: the service time per request is about 1000 / throughput ms.
Production scales by running several API workers behind the same PostgreSQL and Redis.
""")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
