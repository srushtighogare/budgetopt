"""Provider adapters (wire format + error mapping), circuit breaker, fallback executor."""

import json
from decimal import Decimal

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.db.models import AuditEvent, LLMModel, Provider
from app.services.providers.adapters import AnthropicAdapter, GeminiAdapter, GroqAdapter
from app.services.providers.base import (
    AuthenticationFailed,
    ChatMessage,
    Completion,
    InvalidRequest,
    MissingCredentials,
    ProviderAdapter,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from app.services.routing.breaker import CircuitBreaker
from app.services.routing.executor import execute_with_fallback

MSGS = [ChatMessage("system", "Be brief."), ChatMessage("user", "Hi")]


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- Wire formats ---

async def test_groq_request_and_response():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers["authorization"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        })

    c = await GroqAdapter("gsk_test", mock_client(handler)).generate(
        "openai/gpt-oss-20b", MSGS, max_tokens=50, temperature=0.2)
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer gsk_test"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "Be brief."}
    assert seen["body"]["max_completion_tokens"] == 50 and seen["body"]["temperature"] == 0.2
    assert (c.text, c.input_tokens, c.output_tokens, c.finish_reason) == ("Hello!", 12, 3, "stop")
    assert c.latency_ms >= 0


async def test_gemini_maps_roles_system_and_thinking_tokens():
    seen = {}

    def handler(req):
        seen["url"] = str(req.url)
        seen["key"] = req.headers["x-goog-api-key"]
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [
                {"text": "thinking...", "thought": True}, {"text": "Hi "}, {"text": "there"},
            ]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 4,
                              "thoughtsTokenCount": 20},
        })

    msgs = [*MSGS, ChatMessage("assistant", "Hello"), ChatMessage("user", "Again")]
    c = await GeminiAdapter("g_key", mock_client(handler)).generate("gemini-x", msgs)
    assert seen["url"].endswith("/models/gemini-x:generateContent")
    assert seen["key"] == "g_key"
    assert seen["body"]["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    assert [m["role"] for m in seen["body"]["contents"]] == ["user", "model", "user"]
    assert c.text == "Hi there"  # thought parts excluded from the answer
    assert c.output_tokens == 24  # thinking tokens are billed as output


async def test_gemini_blocked_prompt_is_invalid_request():
    def handler(req):
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    with pytest.raises(InvalidRequest, match="SAFETY"):
        await GeminiAdapter("k", mock_client(handler)).generate("m", MSGS)


async def test_anthropic_format():
    seen = {}

    def handler(req):
        seen["headers"] = req.headers
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "Yo"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 9, "output_tokens": 2},
        })

    c = await AnthropicAdapter("sk-ant", mock_client(handler)).generate("anthropic-model-x", MSGS)
    assert seen["headers"]["x-api-key"] == "sk-ant"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "Be brief."
    assert seen["body"]["messages"] == [{"role": "user", "content": "Hi"}]
    assert (c.text, c.input_tokens, c.output_tokens) == ("Yo", 9, 2)


# --- Error mapping ---

@pytest.mark.parametrize(("status", "exc"), [
    (429, RateLimited), (401, AuthenticationFailed), (403, AuthenticationFailed),
    (500, ProviderUnavailable), (503, ProviderUnavailable), (400, InvalidRequest),
    (404, InvalidRequest),
])
async def test_http_errors_mapped(status, exc):
    client = mock_client(lambda req: httpx.Response(status, text="boom"))
    with pytest.raises(exc) as info:
        await GroqAdapter("k", client).generate("m", MSGS)
    assert info.value.status_code == status


async def test_timeout_and_network_errors():
    def timeout(req):
        raise httpx.ReadTimeout("slow")

    def down(req):
        raise httpx.ConnectError("refused")

    with pytest.raises(ProviderTimeout):
        await GroqAdapter("k", mock_client(timeout)).generate("m", MSGS)
    with pytest.raises(ProviderUnavailable):
        await GroqAdapter("k", mock_client(down)).generate("m", MSGS)


async def test_malformed_response_and_missing_key():
    client = mock_client(lambda req: httpx.Response(200, json={"unexpected": True}))
    with pytest.raises(ProviderUnavailable, match="unexpected response"):
        await GroqAdapter("k", client).generate("m", MSGS)
    with pytest.raises(MissingCredentials):
        await GroqAdapter(None).generate("m", MSGS)


# --- Circuit breaker ---

def test_breaker_opens_and_half_opens():
    now = [0.0]
    b = CircuitBreaker(failure_threshold=2, cooldown_s=10, clock=lambda: now[0])
    b.record_failure(1)
    assert b.allow(1)
    b.record_failure(1)
    assert not b.allow(1) and b.is_open(1)
    now[0] = 10.0
    assert b.allow(1)        # half-open trial
    assert not b.allow(1)    # only one trial per cooldown
    b.record_success(1)
    assert b.allow(1) and not b.is_open(1)


# --- Fallback executor (fake adapters, real DB pricing) ---

class FakeAdapter(ProviderAdapter):
    def __init__(self, outcomes: dict[str, Exception | Completion]):
        super().__init__(api_key="x")
        self.outcomes = outcomes
        self.calls: list[str] = []

    async def generate(self, model, messages, **kw):
        self.calls.append(model)
        out = self.outcomes[model]
        if isinstance(out, Exception):
            raise out
        return out

    def build_request(self, *a):  # pragma: no cover
        raise NotImplementedError

    def parse_response(self, *a):  # pragma: no cover
        raise NotImplementedError


async def _models(db, *names: str) -> list[LLMModel]:
    rows = (await db.scalars(
        select(LLMModel).options(selectinload(LLMModel.provider)).where(LLMModel.name.in_(names))
    )).all()
    by_name = {m.name: m for m in rows}
    return [by_name[n] for n in names]


OK = Completion("answer", input_tokens=1000, output_tokens=500, latency_ms=120.0)


async def test_primary_success_is_priced(db):
    cands = await _models(db, "openai/gpt-oss-120b", "gemini-3.1-flash-lite")
    fake = FakeAdapter({"openai/gpt-oss-120b": OK})
    res = await execute_with_fallback(db, cands, MSGS, breaker=CircuitBreaker(),
                                      adapter_factory=lambda p: fake)
    assert res.succeeded and not res.fallback_used
    assert res.model.name == "openai/gpt-oss-120b"
    # 1000 * 0.15/1M + 500 * 0.60/1M
    assert res.cost == Decimal("0.00045000")
    assert fake.calls == ["openai/gpt-oss-120b"]


async def test_falls_back_on_rate_limit(db):
    cands = await _models(db, "gemini-3.5-flash", "openai/gpt-oss-120b")
    fake = FakeAdapter({
        "gemini-3.5-flash": RateLimited("google", "quota exhausted", 429),
        "openai/gpt-oss-120b": OK,
    })
    res = await execute_with_fallback(db, cands, MSGS, breaker=CircuitBreaker(),
                                      adapter_factory=lambda p: fake)
    assert res.succeeded and res.fallback_used
    assert res.model.name == "openai/gpt-oss-120b"
    assert [a.outcome for a in res.attempts] == ["error", "success"]
    assert res.attempts[0].error_type == "RateLimited"


async def test_all_fail_returns_attempts(db):
    cands = await _models(db, "gemini-3.5-flash", "openai/gpt-oss-120b")
    fake = FakeAdapter({
        "gemini-3.5-flash": ProviderUnavailable("google", "503", 503),
        "openai/gpt-oss-120b": ProviderTimeout("groq", "timeout"),
    })
    res = await execute_with_fallback(db, cands, MSGS, breaker=CircuitBreaker(),
                                      adapter_factory=lambda p: fake)
    assert not res.succeeded and res.cost == 0
    assert [a.error_type for a in res.attempts] == ["ProviderUnavailable", "ProviderTimeout"]


async def test_open_circuit_skips_model_and_invalid_request_does_not_trip(db):
    cands = await _models(db, "gemini-3.5-flash", "openai/gpt-oss-120b")
    breaker = CircuitBreaker(failure_threshold=1)
    fake = FakeAdapter({
        "gemini-3.5-flash": ProviderUnavailable("google", "down", 503),
        "openai/gpt-oss-120b": OK,
    })
    await execute_with_fallback(db, cands, MSGS, breaker=breaker, adapter_factory=lambda p: fake)
    res = await execute_with_fallback(db, cands, MSGS, breaker=breaker,
                                      adapter_factory=lambda p: fake)
    assert res.attempts[0].outcome == "skipped_circuit_open"
    assert fake.calls == ["gemini-3.5-flash", "openai/gpt-oss-120b", "openai/gpt-oss-120b"]

    fake.outcomes["openai/gpt-oss-120b"] = InvalidRequest("groq", "context too long", 400)
    await execute_with_fallback(db, cands[1:], MSGS, breaker=breaker,
                                adapter_factory=lambda p: fake)
    assert not breaker.is_open(cands[1].id)


async def test_inactive_provider_skipped(db):
    cands = await _models(db, "gemini-3.5-flash", "openai/gpt-oss-120b")
    cands[0].provider.is_active = False
    fake = FakeAdapter({"openai/gpt-oss-120b": OK})
    res = await execute_with_fallback(db, cands, MSGS, breaker=CircuitBreaker(),
                                      adapter_factory=lambda p: fake)
    assert res.attempts[0].outcome == "skipped_inactive" and res.fallback_used


# --- Provider admin API ---

async def test_provider_key_is_write_only(client, admin_headers, db):
    r = await client.put("/api/v1/providers/openai/key", headers=admin_headers,
                         json={"api_key": "sk-secret-value-123"})
    assert r.status_code == 200 and r.json()["has_api_key"] is True
    listed = await client.get("/api/v1/providers", headers=admin_headers)
    assert "sk-secret-value-123" not in listed.text

    stored = await db.scalar(select(Provider).where(Provider.name == "openai"))
    assert stored.encrypted_api_key and "sk-secret" not in stored.encrypted_api_key
    assert await db.scalar(select(AuditEvent).where(AuditEvent.action == "set_provider_key"))

    r = await client.delete("/api/v1/providers/openai/key", headers=admin_headers)
    assert r.json()["has_api_key"] is False
