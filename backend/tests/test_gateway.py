"""Gateway pipeline: OpenAI compatibility, routing + constraints, fallback, cache,
budget gate, rate limits, privacy and the request log."""

from decimal import Decimal

import pytest
from sqlalchemy import select, update

from app.core.config import get_settings
from app.core.redis import get_redis
from app.db.models import (
    LLMModel,
    PrivacyTier,
    Request,
    RequestStatus,
    Role,
    RoutingDecision,
    Team,
)
from app.services import rate_limit
from app.services.privacy import redact
from app.services.providers.base import (
    ChatMessage,
    Completion,
    ProviderAdapter,
    ProviderUnavailable,
    RateLimited,
)
from app.services.routing import pipeline
from app.services.routing.breaker import CircuitBreaker
from app.services.tokens import estimate_input_tokens
from tests.conftest import login, make_user

CHEAPEST = "openai/gpt-oss-20b"        # groq, $0.075 / $0.30 per 1M
SECOND = "openai/gpt-oss-120b"         # groq, $0.15 / $0.60
PRICIEST = "gemini-3.5-flash"          # google, $1.50 / $9.00 -> savings baseline


class Fake(ProviderAdapter):
    """Records calls; returns or raises per model name."""

    def __init__(self):
        super().__init__(api_key="x")
        self.calls: list[str] = []
        self.outcomes: dict[str, Exception | Completion] = {}

    async def generate(self, model, messages, **kw):
        self.calls.append(model)
        out = self.outcomes.get(model, Completion(f"answer from {model}", 1000, 200, 50.0))
        if isinstance(out, Exception):
            raise out
        return out

    def build_request(self, *a):  # pragma: no cover
        raise NotImplementedError

    def parse_response(self, *a):  # pragma: no cover
        raise NotImplementedError


@pytest.fixture
def fake(monkeypatch) -> Fake:
    f = Fake()
    monkeypatch.setattr(pipeline, "build_adapter", lambda provider: f)
    monkeypatch.setattr(pipeline, "default_breaker", CircuitBreaker())
    return f


async def _key(client, headers, db, team="Engineering") -> dict[str, str]:
    team_id = await db.scalar(select(Team.id).where(Team.name == team))
    r = await client.post(f"/api/v1/teams/{team_id}/api-keys", headers=headers,
                          json={"name": "app"})
    return {"Authorization": f"Bearer {r.json()['key']}"}


@pytest.fixture
async def key(client, admin_headers, db) -> dict[str, str]:
    return await _key(client, admin_headers, db)


def chat(content="What is 2+2?", **kw) -> dict:
    return {"model": "auto", "messages": [{"role": "user", "content": content}], **kw}


# --- Happy path & OpenAI compatibility ---

async def test_auto_routes_to_cheapest_eligible_and_explains(client, key, fake, db):
    r = await client.post("/v1/chat/completions", headers=key, json=chat())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == CHEAPEST
    assert body["choices"][0]["message"] == {"role": "assistant",
                                             "content": f"answer from {CHEAPEST}"}
    assert body["usage"] == {"prompt_tokens": 1000, "completion_tokens": 200,
                             "total_tokens": 1200}

    meta = body["budgetopt"]
    # 1000*0.075/1M + 200*0.30/1M = 0.000135 ; baseline 1000*1.5/1M + 200*9/1M = 0.0033
    assert Decimal(meta["cost"]) == Decimal("0.000135")
    assert Decimal(meta["baseline_cost"]) == Decimal("0.0033")
    assert Decimal(meta["savings"]) == Decimal("0.003165")
    assert "Gemini 3.5 Flash would cost" in meta["explanation"]
    assert r.headers["X-BudgetOpt-Request-Id"] == str(meta["request_id"])

    req = await db.get(Request, meta["request_id"])
    assert req.status == RequestStatus.SUCCESS and req.cost == Decimal("0.000135")
    assert req.prompt_text is None and len(req.prompt_hash) == 64  # privacy default
    d = await db.scalar(select(RoutingDecision).where(RoutingDecision.request_id == req.id))
    assert len(d.candidates["considered"]) == 4
    assert d.candidates["attempts"][0]["outcome"] == "success"


async def test_openai_content_parts_and_developer_role(client, key, fake):
    body = {"model": "auto", "messages": [
        {"role": "developer", "content": "Be terse."},
        {"role": "user", "content": [{"type": "text", "text": "Hi "},
                                     {"type": "text", "text": "there"}]},
    ]}
    r = await client.post("/v1/chat/completions", headers=key, json=body)
    assert r.status_code == 200


async def test_explicit_model_is_honoured(client, key, fake):
    r = await client.post("/v1/chat/completions", headers=key, json=chat(model=SECOND))
    assert r.json()["model"] == SECOND and fake.calls == [SECOND]


async def test_unknown_model_404_in_openai_shape(client, key, fake):
    r = await client.post("/v1/chat/completions", headers=key, json=chat(model="gpt-9"))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"
    assert fake.calls == []


async def test_auth_and_validation_errors_in_openai_shape(client, key):
    bad = await client.post("/v1/chat/completions", json=chat(),
                            headers={"Authorization": "Bearer bo_wrong"})
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "invalid_api_key"
    missing = await client.post("/v1/chat/completions", json=chat())
    assert missing.status_code == 401
    no_user = await client.post("/v1/chat/completions", headers=key,
                                json={"messages": [{"role": "system", "content": "x"}]})
    assert no_user.status_code == 400 and "error" in no_user.json()
    stream = await client.post("/v1/chat/completions", headers=key, json=chat(stream=True))
    assert stream.status_code == 400


async def test_revoked_key_rejected(client, admin_headers, db, fake):
    team_id = await db.scalar(select(Team.id).where(Team.name == "Engineering"))
    created = (await client.post(f"/api/v1/teams/{team_id}/api-keys", headers=admin_headers,
                                 json={"name": "tmp"})).json()
    await client.delete(f"/api/v1/teams/{team_id}/api-keys/{created['id']}",
                        headers=admin_headers)
    r = await client.post("/v1/chat/completions", json=chat(),
                          headers={"Authorization": f"Bearer {created['key']}"})
    assert r.status_code == 401


async def test_models_endpoint(client, key):
    ids = [m["id"] for m in (await client.get("/v1/models", headers=key)).json()["data"]]
    assert ids[0] == "auto" and len(ids) == 5


# --- Fallback & failures ---

async def test_fallback_to_next_cheapest(client, key, fake, db):
    fake.outcomes[CHEAPEST] = RateLimited("groq", "slow down", 429)
    r = await client.post("/v1/chat/completions", headers=key, json=chat())
    meta = r.json()["budgetopt"]
    assert r.json()["model"] == SECOND and meta["status"] == "fallback_success"
    assert "Fallback used after" in meta["explanation"]
    assert fake.calls == [CHEAPEST, SECOND]


async def test_all_models_failing_returns_502_and_is_logged(client, key, fake, db):
    for name in (CHEAPEST, SECOND, "gemini-3.1-flash-lite", PRICIEST):
        fake.outcomes[name] = ProviderUnavailable("x", "down", 503)
    r = await client.post("/v1/chat/completions", headers=key, json=chat())
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_failed"
    req = await db.get(Request, r.json()["error"]["budgetopt"]["request_id"])
    assert req.status == RequestStatus.FAILED and req.cost == 0


# --- Cache ---

async def test_identical_request_served_from_cache(client, key, fake, db):
    first = (await client.post("/v1/chat/completions", headers=key, json=chat())).json()
    second = (await client.post("/v1/chat/completions", headers=key, json=chat())).json()
    assert fake.calls == [CHEAPEST]  # only one real LLM call
    assert second["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"]
    meta = second["budgetopt"]
    assert meta["status"] == "cache_hit" and Decimal(meta["cost"]) == 0
    assert Decimal(meta["savings"]) == Decimal(first["budgetopt"]["cost"])
    assert (await db.get(Request, meta["request_id"])).status == RequestStatus.CACHE_HIT


async def test_cache_bypass_rules(client, key, admin_headers, db, fake):
    await client.post("/v1/chat/completions", headers=key, json=chat())
    # explicit bypass
    await client.post("/v1/chat/completions", headers={**key, "Cache-Control": "no-cache"},
                      json=chat())
    # sampling requests are never cached
    await client.post("/v1/chat/completions", headers=key, json=chat(temperature=0.7))
    await client.post("/v1/chat/completions", headers=key, json=chat(temperature=0.7))
    # caches are per team
    other = await _key(client, admin_headers, db, team="Support")
    await client.post("/v1/chat/completions", headers=other, json=chat())
    assert len(fake.calls) == 5


# --- Budget gate ---

async def _spend(db, team_name: str, amount: str) -> int:
    team = await db.scalar(select(Team).where(Team.name == team_name))
    db.add(Request(team_id=team.id, prompt_hash="0" * 64, cost=Decimal(amount),
                   status=RequestStatus.SUCCESS))
    await db.commit()
    return team.id


async def test_over_budget_team_blocked_without_llm_call(client, key, fake, db):
    await _spend(db, "Engineering", "5.00")  # budget is 5.00
    r = await client.post("/v1/chat/completions", headers=key, json=chat())
    assert r.status_code == 402 and r.json()["error"]["code"] == "budget_exceeded"
    assert fake.calls == []
    req = await db.get(Request, r.json()["error"]["budgetopt"]["request_id"])
    assert req.status == RequestStatus.BLOCKED_BUDGET and req.cost == 0


async def test_soft_alert_is_reported(client, key, fake, db):
    await _spend(db, "Engineering", "4.10")  # 82% of 5.00, threshold 80%
    r = await client.post("/v1/chat/completions", headers=key, json=chat())
    assert r.status_code == 200
    assert r.headers["X-BudgetOpt-Budget-State"] == "soft_alert"
    assert "Budget alert: 82%" in r.json()["budgetopt"]["explanation"]


# --- Constraints ---

async def test_privacy_tier_excludes_all_public_models(client, key, fake, db):
    await db.execute(update(Team).where(Team.name == "Engineering")
                     .values(privacy_tier=PrivacyTier.CONFIDENTIAL))
    await db.commit()
    r = await client.post("/v1/chat/completions", headers=key, json=chat())
    assert r.status_code == 422 and r.json()["error"]["code"] == "no_eligible_model"
    assert "privacy" in r.json()["error"]["message"] and fake.calls == []


async def test_context_window_constraint(client, key, fake, db):
    await db.execute(update(LLMModel).where(LLMModel.name == CHEAPEST).values(context_window=50))
    await db.commit()
    r = await client.post("/v1/chat/completions", headers=key, json=chat(max_tokens=100))
    body = r.json()
    assert body["model"] == SECOND
    assert f"{CHEAPEST} (context" in body["budgetopt"]["explanation"]


async def test_prompt_size_limit(client, key, fake, monkeypatch):
    monkeypatch.setattr(get_settings(), "max_prompt_chars", 10)
    r = await client.post("/v1/chat/completions", headers=key, json=chat("x" * 11))
    assert r.status_code == 413 and fake.calls == []


# --- Rate limiting ---

async def test_rate_limit_per_api_key(client, key, fake, monkeypatch):
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    codes = [(await client.post("/v1/chat/completions", headers=key,
                                json=chat(f"q{i}"))).status_code for i in range(3)]
    assert codes == [200, 200, 429]


async def test_fixed_window_resets():
    redis = get_redis()
    assert (await rate_limit.check(redis, "t", 1, now=120.0)).allowed
    assert not (await rate_limit.check(redis, "t", 1, now=150.0)).allowed
    assert (await rate_limit.check(redis, "t", 1, now=180.0)).allowed  # next minute


# --- Privacy ---

async def test_opted_in_prompts_are_stored_redacted(client, key, fake, db):
    await db.execute(update(Team).where(Team.name == "Engineering").values(store_prompts=True))
    await db.commit()
    r = await client.post("/v1/chat/completions", headers=key,
                          json=chat("Email priya@corp.in, PAN ABCDE1234F"))
    req = await db.get(Request, r.json()["budgetopt"]["request_id"])
    assert req.prompt_text == "user: Email [EMAIL], PAN [PAN]"


def test_redaction_and_token_estimate():
    assert redact("call 98765 43210 now") == "call [PHONE] now"
    assert estimate_input_tokens([ChatMessage("user", "hello world")]) == 3 + 4 + 2


# --- Playground & request log (RBAC) ---

async def test_playground_and_request_log_scoping(client, db, fake):
    await make_user(db, "a@demo.io", Role.MEMBER, "Engineering")
    await make_user(db, "b@demo.io", Role.MEMBER, "Engineering")
    await make_user(db, "lead@demo.io", Role.TEAM_MANAGER, "Engineering")
    a = await login(client, "a@demo.io")
    b = await login(client, "b@demo.io")
    lead = await login(client, "lead@demo.io")

    ra = await client.post("/api/v1/route", headers=a, json=chat("from a"))
    assert ra.status_code == 200 and ra.json()["model"] == CHEAPEST
    rb = await client.post("/api/v1/route", headers=b, json=chat("from b"))

    mkt = await db.scalar(select(Team.id).where(Team.name == "Marketing"))
    denied = await client.post("/api/v1/route", headers=a, json={**chat(), "team_id": mkt})
    assert denied.status_code == 403

    mine = (await client.get("/api/v1/requests", headers=a)).json()
    assert [r["id"] for r in mine] == [ra.json()["request_id"]]
    other = await client.get(f"/api/v1/requests/{rb.json()['request_id']}", headers=a)
    assert other.status_code == 404

    team_view = (await client.get("/api/v1/requests", headers=lead)).json()
    assert len(team_view) == 2
    detail = (await client.get(f"/api/v1/requests/{ra.json()['request_id']}",
                               headers=lead)).json()
    assert detail["candidates"]["strategy"].startswith("cheapest eligible")
    assert detail["baseline_cost"] is not None


async def test_truncated_output_reports_length(client, key, fake):
    fake.outcomes[CHEAPEST] = Completion("partial", 10, 50, 5.0, finish_reason="MAX_TOKENS")
    body = (await client.post("/v1/chat/completions", headers=key, json=chat())).json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert "cut off" in body["budgetopt"]["explanation"]


async def test_money_fields_are_plain_decimals(client, key, fake):
    body = (await client.post("/v1/chat/completions", headers=key,
                              json=chat(model=PRICIEST))).json()
    assert body["budgetopt"]["savings"] == "0"
