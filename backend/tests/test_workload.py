"""Workload analysis: classifier on clear-cut prompts, fallbacks, and pipeline logging."""

import pytest
from sqlalchemy import select

from app.db.models import ComplexityBand, Request, TaskType, Team
from app.services.providers.base import ChatMessage, Completion, ProviderAdapter
from app.services.routing import pipeline
from app.services.routing.breaker import CircuitBreaker
from app.services.workload import analyzer
from app.services.workload.analyzer import analyze, band_of, request_text


@pytest.mark.parametrize(("prompt", "task"), [
    ("Translate 'good morning, how are you?' into German.", TaskType.TRANSLATION),
    ("Write a Python function that merges two sorted lists in linear time.", TaskType.CODING),
    ("If 3x + 7 = 25, what is x? Show your working.", TaskType.REASONING_MATH),
    ("Compare and contrast monolithic and microservice architectures.", TaskType.ANALYSIS),
    ("Summarize the following passage in two sentences:\n\nThe council voted to expand "
     "cycle lanes across the city after a year-long consultation.", TaskType.SUMMARIZATION),
    ("Write a short poem about the monsoon.", TaskType.CREATIVE_WRITING),
])
def test_classifier_on_clear_prompts(prompt, task):
    p = analyze([ChatMessage("user", prompt)])
    assert p.task_type == task
    assert 0 <= p.task_confidence <= 1 and 0 <= p.complexity <= 1
    assert p.est_input_tokens > 0 and p.est_output_tokens > 0


def test_latest_user_turn_defines_task():
    msgs = [ChatMessage("system", "You are a translator."),
            ChatMessage("user", "Translate 'hello' to French."),
            ChatMessage("assistant", "Bonjour."),
            ChatMessage("user", "Now write a Python function that reverses a string.")]
    assert request_text(msgs).startswith("Now write")
    assert analyze(msgs).task_type == TaskType.CODING


def test_bands():
    assert band_of(0.1) == ComplexityBand.LOW
    assert band_of(0.5) == ComplexityBand.MEDIUM
    assert band_of(0.9) == ComplexityBand.HIGH


def test_missing_artifacts_fall_back(monkeypatch):
    monkeypatch.setattr(analyzer, "_load", lambda name: None)
    p = analyze([ChatMessage("user", "anything")])
    assert p.sources == {"task": "fallback", "complexity": "heuristic",
                         "output_tokens": "task median"}
    assert p.task_type == TaskType.SIMPLE_QA


class _Fake(ProviderAdapter):
    def __init__(self):
        super().__init__(api_key="x")

    async def generate(self, model, messages, **kw):
        return Completion("ok", 50, 20, 5.0)

    def build_request(self, *a):  # pragma: no cover
        raise NotImplementedError

    def parse_response(self, *a):  # pragma: no cover
        raise NotImplementedError


async def test_profile_is_logged_and_explained(client, admin_headers, db, monkeypatch):
    fake = _Fake()
    monkeypatch.setattr(pipeline, "build_adapter", lambda p: fake)
    monkeypatch.setattr(pipeline, "default_breaker", CircuitBreaker())
    team_id = await db.scalar(select(Team.id).where(Team.name == "Engineering"))
    key = (await client.post(f"/api/v1/teams/{team_id}/api-keys", headers=admin_headers,
                             json={"name": "w"})).json()["key"]
    r = await client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                          json={"messages": [{"role": "user", "content":
                                "Write a SQL query listing customers with more than 5 orders."}]})
    meta = r.json()["budgetopt"]
    assert meta["workload"]["task_type"] == "coding"
    assert meta["explanation"].startswith("Classified as coding")
    req = await db.get(Request, meta["request_id"])
    assert req.task_type == TaskType.CODING and req.complexity_band is not None
    assert req.est_output_tokens == meta["workload"]["est_output_tokens"]
