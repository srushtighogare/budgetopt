"""Semantic cache: two-stage matching, guards, costs and expiry (fake embeddings so every
similarity in a test is exact)."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import select, update

from app.core.config import get_settings
from app.db.models import (
    PrivacyTier,
    Request,
    RequestStatus,
    RoutingDecision,
    SemanticCacheEntry,
    TaskType,
    Team,
)
from app.services import semantic_cache
from app.services.providers.base import ChatMessage, Completion, ProviderAdapter
from app.services.routing import pipeline
from app.services.routing.breaker import CircuitBreaker

BASE = "What is the capital of Australia?"
VECS: dict[str, list[float]] = {}


def with_similarity(sim: float) -> list[float]:
    v = np.zeros(384)
    v[0], v[1] = sim, np.sqrt(1 - sim**2)
    return v.tolist()


class Fake(ProviderAdapter):
    def __init__(self, verdict: str = "YES"):
        super().__init__(api_key="x")
        self.calls: list[str] = []
        self.verdict = verdict

    async def generate(self, model, messages, **kw):
        self.calls.append(model)
        if messages[0].role == "system" and "Reply with exactly one word" in messages[0].content:
            return Completion(self.verdict, 120, 2, 5.0)
        return Completion(f"answer to: {messages[-1].content}", 100, 300, 10.0)

    def build_request(self, *a):  # pragma: no cover
        raise NotImplementedError

    def parse_response(self, *a):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def fake(monkeypatch) -> Fake:
    monkeypatch.setattr(get_settings(), "semantic_cache_enabled", True)
    f = Fake()
    monkeypatch.setattr(pipeline, "build_adapter", lambda p: f)
    monkeypatch.setattr(pipeline, "default_breaker", CircuitBreaker())
    VECS.clear()
    VECS[BASE] = with_similarity(1.0)
    monkeypatch.setattr(semantic_cache, "embed", lambda t: VECS.get(t, with_similarity(0.1)))
    return f


async def _key(client, headers, db, team="Engineering"):
    team_id = await db.scalar(select(Team.id).where(Team.name == team))
    r = await client.post(f"/api/v1/teams/{team_id}/api-keys", headers=headers,
                          json={"name": "s"})
    return {"Authorization": f"Bearer {r.json()['key']}"}


async def ask(client, key, text, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": text}]
    r = await client.post("/v1/chat/completions", headers=key, json={"messages": msgs})
    assert r.status_code == 200, r.text
    return r.json()


async def test_high_similarity_served_without_llm(client, admin_headers, db, fake):
    key = await _key(client, admin_headers, db)
    first = await ask(client, key, BASE)
    assert first["budgetopt"]["workload"]["task_type"] == "simple_qa"
    VECS["Which city is Australia's capital?"] = with_similarity(0.98)
    second = await ask(client, key, "Which city is Australia's capital?")
    meta = second["budgetopt"]
    assert meta["status"] == "cache_hit" and "semantic cache" in meta["explanation"]
    assert second["choices"][0]["message"]["content"] == first["choices"][0]["message"][
        "content"]
    assert len(fake.calls) == 1  # one real answer, no verifier needed above 0.97
    entry = await db.scalar(select(SemanticCacheEntry))
    assert entry.hits == 1 and BASE not in entry.prompt_encrypted  # prompt stored encrypted


async def test_grey_zone_verifier_yes_is_a_cheap_hit(client, admin_headers, db, fake):
    key = await _key(client, admin_headers, db)
    first = await ask(client, key, BASE)
    VECS["Australia's capital city is?"] = with_similarity(0.90)
    second = await ask(client, key, "Australia's capital city is?")
    meta = second["budgetopt"]
    assert meta["status"] == "cache_hit" and "confirmed by verifier" in meta["explanation"]
    assert fake.calls[-1] == "gemini-3.5-flash-lite"
    verifier_cost = Decimal(meta["cost"])
    assert 0 < verifier_cost < Decimal(first["budgetopt"]["cost"])
    assert Decimal(meta["savings"]) == Decimal(first["budgetopt"]["cost"]) - verifier_cost


async def test_grey_zone_verifier_no_is_a_miss_and_charged(client, admin_headers, db, fake):
    fake.verdict = "NO"
    key = await _key(client, admin_headers, db)
    await ask(client, key, BASE)
    VECS["What is the capital of Austria?"] = with_similarity(0.93)
    second = await ask(client, key, "What is the capital of Austria?")
    meta = second["budgetopt"]
    assert meta["status"] == "success"
    assert "rejected by the verifier" in meta["explanation"]
    req = await db.get(Request, meta["request_id"])
    d = await db.scalar(select(RoutingDecision).where(RoutingDecision.request_id == req.id))
    # charged = answer at the chosen model's price + the verifier call
    assert req.cost > Decimal(0) and d.explanation.count("verifier cost") == 1


async def test_numbers_guard_context_and_task_eligibility(client, admin_headers, db, fake):
    key = await _key(client, admin_headers, db)
    VECS["What is the population of India in 2011?"] = with_similarity(1.0)
    VECS["What is the population of India in 2021?"] = with_similarity(0.99)
    await ask(client, key, "What is the population of India in 2011?")
    r = await ask(client, key, "What is the population of India in 2021?")
    assert r["budgetopt"]["status"] == "success"
    assert "different numbers" in r["budgetopt"]["explanation"]

    await ask(client, key, BASE)
    VECS["Capital of Australia?"] = with_similarity(0.99)
    other_context = await ask(client, key, "Capital of Australia?", system="Answer in French.")
    assert other_context["budgetopt"]["status"] == "success"  # system prompt differs

    n_entries = len(list(await db.scalars(select(SemanticCacheEntry))))
    await ask(client, key, "Write a Python function that reverses a string.")
    assert len(list(await db.scalars(select(SemanticCacheEntry)))) == n_entries  # coding: off


async def test_disabled_for_confidential_teams(client, admin_headers, db, fake):
    await db.execute(update(Team).where(Team.name == "Engineering")
                     .values(privacy_tier=PrivacyTier.CONFIDENTIAL))
    await db.commit()
    team = await db.scalar(select(Team).where(Team.name == "Engineering"))
    assert not semantic_cache.eligible(team, TaskType.SIMPLE_QA)


async def test_expired_entries_are_purged(db, fake):
    team = await db.scalar(select(Team).where(Team.name == "Engineering"))
    req = Request(team_id=team.id, prompt_hash="p" * 64, status=RequestStatus.SUCCESS)
    db.add(req)
    await db.flush()
    await semantic_cache.store(db, team, "ctx", TaskType.SIMPLE_QA, with_similarity(1.0), "",
                               prompt="hello", response_text="hi", model_id=1,
                               model_name="m", cost=Decimal("0.001"), request_id=req.id)
    later = datetime.now(UTC) + timedelta(days=2)
    assert await semantic_cache.purge_expired(db, now=later) == 1
    hit = await semantic_cache.lookup(db, team, "ctx", with_similarity(1.0), "", now=later)
    assert hit.hit is None


def test_helpers():
    assert semantic_cache.numbers_in("3 users, 1,200 tokens, v2.5") == "1200 2.5 3"
    a = semantic_cache.context_key(1, "auto", [ChatMessage("user", "x")], 100, None)
    b = semantic_cache.context_key(1, "auto", [ChatMessage("user", "y")], 100, None)
    c = semantic_cache.context_key(1, "auto", [ChatMessage("system", "s"),
                                               ChatMessage("user", "x")], 100, None)
    assert a == b and a != c  # last user message excluded, system prompt included
