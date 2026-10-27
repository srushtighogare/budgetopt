"""Quality profiling harness, latency profiles and the profiling API."""

import asyncio
from decimal import Decimal

from sqlalchemy import select

from app.api.v1 import profiling as profiling_api
from app.db.models import (
    LLMModel,
    ModelPerformance,
    ProfilingResult,
    Request,
    RequestStatus,
    TaskType,
    Team,
)
from app.services.profiling.answers import contains_answer, last_boxed
from app.services.profiling.latency import refresh_latency
from app.services.profiling.quality import load_items, run_quality_profile
from app.services.providers.base import Completion, ProviderAdapter, RateLimited

CHEAP, JUDGE, JUDGE2 = "openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"


class Router(ProviderAdapter):
    """One fake adapter for every provider; behaviour chosen per model name."""

    def __init__(self, judge_reply='{"score": 7, "reason": "fine"}'):
        super().__init__(api_key="x")
        self.calls: list[str] = []
        self.judge_reply = judge_reply
        self.fail: dict[str, Exception] = {}
        self.gold_answers: dict[str, str] = {}

    async def generate(self, model, messages, **kw):
        self.calls.append(model)
        if model in self.fail:
            raise self.fail[model]
        if messages[0].role == "system":  # judge call
            return Completion(self.judge_reply, 200, 30, 40.0)
        text = messages[0].content
        return Completion(self.gold_answers.get(text, "an answer"), 50, 100, 300.0)

    def build_request(self, *a):  # pragma: no cover
        raise NotImplementedError

    def parse_response(self, *a):  # pragma: no cover
        raise NotImplementedError


async def _no_sleep(_):
    return None


def test_profiling_set_is_balanced_with_gold_math():
    items = load_items()
    assert len(items) == 64
    assert {it.task_type for it in items} == {t.value for t in TaskType}
    math = [it for it in items if it.task_type == "reasoning_math"]
    assert all(it.gold for it in math)


def test_gold_answer_checks():
    assert last_boxed(r"so \boxed{\frac{3}{4}} done") == r"\frac{3}{4}"
    assert contains_answer(r"The answer is \boxed{42}.", "42")
    assert contains_answer("He earns 1,250 dollars in total. Final: 1250", "1250")
    assert not contains_answer("I think it's 41", "42")


async def test_quality_run_scores_aggregates_and_resumes(db):
    fake = Router()
    math = next(it for it in load_items() if it.task_type == "reasoning_math")
    fake.gold_answers[math.text] = f"Working... final answer {math.gold}"

    rep = await run_quality_profile(db, model_names=[CHEAP], judge_names=[CHEAP, JUDGE],
                                    per_task=1, adapter_factory=lambda p: fake, sleep=_no_sleep)
    assert rep.answered == 8 and rep.judged == 7  # math item is gold-checked, not judged
    model = await db.scalar(select(LLMModel).where(LLMModel.name == CHEAP))
    rows = list(await db.scalars(select(ProfilingResult).where(
        ProfilingResult.model_id == model.id)))
    assert len(rows) == 8
    gold_row = next(r for r in rows if r.task_type == TaskType.REASONING_MATH)
    assert gold_row.correct is True and gold_row.score == 10 and gold_row.judge == "gold-answer"
    assert all(r.judge == JUDGE for r in rows if r.judge != "gold-answer")  # never self-judged
    assert all(r.cost > 0 for r in rows)

    perf = {p.task_type: p for p in await db.scalars(
        select(ModelPerformance).where(ModelPerformance.model_id == model.id))}
    assert perf[TaskType.CODING].quality_score == 7 and perf[TaskType.CODING].quality_samples == 1
    assert perf[TaskType.REASONING_MATH].quality_score == 10

    again = await run_quality_profile(db, model_names=[CHEAP], judge_names=[JUDGE],
                                      per_task=1, adapter_factory=lambda p: fake,
                                      sleep=_no_sleep)
    assert again.answered == 0 and again.skipped_existing == 8


async def test_rate_limited_model_is_skipped_and_judge_falls_back(db):
    fake = Router()
    fake.fail[JUDGE] = RateLimited("groq", "daily quota", 429)
    rep = await run_quality_profile(db, model_names=[CHEAP], judge_names=[JUDGE, JUDGE2],
                                    per_task=1, adapter_factory=lambda p: fake, sleep=_no_sleep)
    assert JUDGE in rep.exhausted_models and rep.judged == 7
    assert set(await db.scalars(select(ProfilingResult.judge).where(
        ProfilingResult.judge != "gold-answer"))) == {JUDGE2}

    fake.fail[CHEAP] = RateLimited("groq", "daily quota", 429)
    await db.execute(ProfilingResult.__table__.delete())
    await db.commit()
    rep = await run_quality_profile(db, model_names=[CHEAP], judge_names=[JUDGE2],
                                    per_task=1, adapter_factory=lambda p: fake, sleep=_no_sleep)
    assert CHEAP in rep.exhausted_models and rep.answered == 0


async def test_unparseable_judge_leaves_item_for_next_run(db):
    fake = Router(judge_reply="I cannot grade this")
    rep = await run_quality_profile(db, model_names=[CHEAP], judge_names=[JUDGE], per_task=1,
                                    adapter_factory=lambda p: fake, sleep=_no_sleep)
    assert rep.judged == 0 and rep.errors == 7
    # only the gold-checked item was stored
    assert len(list(await db.scalars(select(ProfilingResult)))) == 1


async def test_latency_percentiles(db):
    team = await db.scalar(select(Team).where(Team.name == "Engineering"))
    model = await db.scalar(select(LLMModel).where(LLMModel.name == CHEAP))
    for ms in (100, 200, 300, 400, 1000):
        db.add(Request(team_id=team.id, prompt_hash="x" * 64, model_id=model.id,
                       task_type=TaskType.CODING, latency_ms=ms, cost=Decimal(0),
                       status=RequestStatus.SUCCESS))
    # failed calls and too-small groups are ignored
    db.add(Request(team_id=team.id, prompt_hash="y" * 64, model_id=model.id,
                   task_type=TaskType.CODING, latency_ms=99999, status=RequestStatus.FAILED))
    db.add(Request(team_id=team.id, prompt_hash="z" * 64, model_id=model.id,
                   task_type=TaskType.ANALYSIS, latency_ms=50, status=RequestStatus.SUCCESS))
    await db.commit()

    assert await refresh_latency(db) == 1
    p = await db.scalar(select(ModelPerformance).where(
        ModelPerformance.model_id == model.id, ModelPerformance.task_type == TaskType.CODING))
    assert p.latency_p50_ms == 300 and p.latency_p95_ms == 880


async def test_profiling_api(client, admin_headers, monkeypatch):
    started = asyncio.Event()

    async def fake_run(body):
        started.set()
        await asyncio.sleep(0.2)

    monkeypatch.setattr(profiling_api, "_run_in_background", fake_run)
    monkeypatch.setattr(profiling_api, "_run_task", None)

    r = await client.post("/api/v1/profiling/quality/run", headers=admin_headers,
                          json={"per_task": 1})
    assert r.status_code == 202
    await asyncio.wait_for(started.wait(), 1)
    busy = await client.post("/api/v1/profiling/quality/run", headers=admin_headers, json={})
    assert busy.status_code == 409

    s = (await client.get("/api/v1/profiling/summary", headers=admin_headers)).json()
    assert len(s["task_types"]) == 8 and s["quality_run_active"] is True
    assert (await client.post("/api/v1/profiling/latency/refresh",
                              headers=admin_headers)).json() == {"updated": 0}


async def test_latency_ignores_synthetic_demo_rows(db):
    from app.services.profiling.latency import SYNTHETIC_HASH

    team = await db.scalar(select(Team).where(Team.name == "Engineering"))
    model = await db.scalar(select(LLMModel).where(LLMModel.name == CHEAP))
    for ms in (100, 200, 300, 400):
        db.add(Request(team_id=team.id, prompt_hash=SYNTHETIC_HASH, model_id=model.id,
                       task_type=TaskType.CODING, latency_ms=ms, status=RequestStatus.SUCCESS))
    await db.commit()
    assert await refresh_latency(db) == 0


async def test_per_minute_limits_are_waited_out_daily_caps_are_not(db):
    class Flaky(Router):
        def __init__(self, fails, wait):
            super().__init__()
            self.left, self.wait = fails, wait

        async def generate(self, model, messages, **kw):
            if self.left:
                self.left -= 1
                raise RateLimited("groq", "limit", 429, retry_after_s=self.wait)
            return await super().generate(model, messages, **kw)

    waits = []

    async def record(s):
        waits.append(s)

    minute = Flaky(fails=3, wait=12)
    rep = await run_quality_profile(db, model_names=[CHEAP], judge_names=[JUDGE], per_task=1,
                                    adapter_factory=lambda p: minute, sleep=record)
    assert rep.answered == 8 and CHEAP not in rep.exhausted_models
    assert waits[:3] == [13, 13, 13]  # provider's Retry-After + 1 s

    await db.execute(ProfilingResult.__table__.delete())
    await db.commit()
    daily = Flaky(fails=99, wait=8000)
    rep = await run_quality_profile(db, model_names=[CHEAP], judge_names=[JUDGE], per_task=1,
                                    adapter_factory=lambda p: daily, sleep=record)
    assert CHEAP in rep.exhausted_models and rep.answered == 0
