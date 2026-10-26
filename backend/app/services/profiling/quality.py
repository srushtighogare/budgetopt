"""Offline quality profiling (report architecture item 3: model quality per task).

Every registered model answers the same fixed prompt set (`profiling_set.jsonl`,
8 prompts per task type). Each answer is scored 0–10:
  - items with a gold answer (GSM8K/MATH): 10 if the final answer is correct, else 0
  - all other items: an LLM judge grades the answer on an absolute rubric
    (MT-Bench-style single-answer grading); a model never judges its own answers

Results are stored per answer in `profiling_results` (resumable, auditable) and
aggregated into `model_performance.quality_score` for the optimizer.

Free-tier friendly: rate-limit errors are retried with backoff; a model that keeps
hitting limits is skipped for the rest of the run, and re-running continues
where it stopped.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import LLMModel, ModelPerformance, ProfilingResult, TaskType
from app.services.pricing import compute_cost, price_at
from app.services.profiling.answers import contains_answer, parse_json
from app.services.providers.base import (
    ChatMessage,
    Completion,
    InvalidRequest,
    ProviderError,
    RateLimited,
)
from app.services.providers.registry import AdapterFactory, build_adapter

log = logging.getLogger(__name__)

PROFILING_SET = Path(__file__).parent / "profiling_set.jsonl"
ANSWER_MAX_TOKENS = 1024
JUDGE_MAX_TOKENS = 1024
RETRIES = 3

JUDGE_SYSTEM = """You are a strict, impartial evaluator of AI assistant answers.
Rate the ANSWER to the user's REQUEST from 1 to 10 for correctness, completeness,
helpfulness and how well it follows the request's instructions.
10 = excellent and fully correct; 7 = good with minor issues; 4 = partially correct or
incomplete; 1 = wrong, off-topic or refused. Do not reward length for its own sake.
Reply ONLY with JSON: {"score": <1-10>, "reason": "<one short sentence>"}"""


@dataclass(frozen=True)
class Item:
    item_id: str
    task_type: str
    text: str
    gold: str | None


@dataclass
class RunReport:
    answered: int = 0
    judged: int = 0
    skipped_existing: int = 0
    errors: int = 0
    exhausted_models: set[str] = field(default_factory=set)


def load_items(per_task: int | None = None) -> list[Item]:
    with open(PROFILING_SET) as f:
        items = [Item(r["item_id"], r["task_type"], r["text"], r.get("gold"))
                 for r in map(json.loads, f)]
    if per_task is None:
        return items
    counts: dict[str, int] = {}
    out = []
    for it in items:
        if counts.get(it.task_type, 0) < per_task:
            counts[it.task_type] = counts.get(it.task_type, 0) + 1
            out.append(it)
    return out


async def _with_retries(call: Callable[[], Awaitable[Completion]],
                        sleep: Callable[[float], Awaitable[None]]) -> Completion:
    for attempt in range(RETRIES):
        try:
            return await call()
        except RateLimited:
            if attempt == RETRIES - 1:
                raise
            await sleep(20 * (attempt + 1))
    raise AssertionError("unreachable")


async def run_quality_profile(
    db: AsyncSession,
    *,
    model_names: list[str] | None = None,
    judge_names: list[str],
    per_task: int | None = None,
    adapter_factory: AdapterFactory = build_adapter,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> RunReport:
    q = select(LLMModel).options(selectinload(LLMModel.provider)).where(LLMModel.is_active)
    if model_names:
        q = q.where(LLMModel.name.in_(model_names))
    models = list(await db.scalars(q.order_by(LLMModel.id)))
    all_models = {m.name: m for m in await db.scalars(
        select(LLMModel).options(selectinload(LLMModel.provider)))}
    judges = [all_models[n] for n in judge_names if n in all_models]
    items = load_items(per_task)
    report = RunReport()

    for model in models:
        done = set(await db.scalars(
            select(ProfilingResult.item_id).where(ProfilingResult.model_id == model.id)))
        pricing = await price_at(db, model.id)
        adapter = adapter_factory(model.provider)
        for item in items:
            if model.name in report.exhausted_models:
                break
            if item.item_id in done:
                report.skipped_existing += 1
                continue
            row = ProfilingResult(model_id=model.id, item_id=item.item_id,
                                  task_type=TaskType(item.task_type))
            try:
                c = await _with_retries(lambda m=model, a=adapter, it=item: a.generate(
                    m.name, [ChatMessage("user", it.text)], max_tokens=ANSWER_MAX_TOKENS,
                    timeout=120), sleep)
            except RateLimited:
                report.exhausted_models.add(model.name)
                log.warning("rate limits persist for %s; skipping for this run", model.name)
                break
            except ProviderError as e:
                # Recorded with no score: a failed answer is a reliability signal, and
                # re-running won't retry it (delete the row to retry)
                row.error = f"{type(e).__name__}: {e}"[:500]
                report.errors += 1
                db.add(row)
                await db.commit()
                continue

            report.answered += 1
            row.answer = c.text
            row.input_tokens, row.output_tokens = c.input_tokens, c.output_tokens
            row.latency_ms = c.latency_ms
            row.cost = compute_cost(pricing, c.input_tokens, c.output_tokens)

            if item.gold:
                row.correct = contains_answer(c.text, item.gold)
                row.score = 10.0 if row.correct else 0.0
                row.judge = "gold-answer"
            else:
                verdict = await _judge(item, c.text, model, judges, report, adapter_factory,
                                       sleep)
                if verdict is None:  # no judge available now; retry this item next run
                    report.errors += 1
                    continue
                row.score, row.judge, row.judge_reason = verdict
                report.judged += 1
            db.add(row)
            await db.commit()

    await aggregate_quality(db)
    return report


async def _judge(item: Item, answer: str, model: LLMModel, judges: list[LLMModel],
                 report: RunReport, adapter_factory: AdapterFactory,
                 sleep: Callable[[float], Awaitable[None]]) -> tuple[float, str, str] | None:
    messages = [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", (
        f"### REQUEST\n{item.text[:6000]}\n\n### ANSWER\n{answer[:8000]}"))]
    for judge in judges:
        if judge.id == model.id or judge.name in report.exhausted_models:
            continue
        adapter = adapter_factory(judge.provider)
        try:
            c = await _with_retries(lambda j=judge, a=adapter: a.generate(
                j.name, messages, max_tokens=JUDGE_MAX_TOKENS, timeout=120), sleep)
            v = parse_json(c.text)
            score = float(v["score"])
            if not 1 <= score <= 10:
                raise ValueError(score)
            return score, judge.name, str(v.get("reason", ""))[:500]
        except RateLimited:
            report.exhausted_models.add(judge.name)
        except (InvalidRequest, ProviderError, ValueError, KeyError, TypeError) as e:
            log.warning("judge %s failed on %s: %s", judge.name, item.item_id, e)
    return None


async def aggregate_quality(db: AsyncSession) -> int:
    """Upsert mean score and sample count per (model, task) into model_performance."""
    rows = (await db.execute(
        select(ProfilingResult.model_id, ProfilingResult.task_type,
               func.avg(ProfilingResult.score), func.count(ProfilingResult.score))
        .where(ProfilingResult.score.is_not(None))
        .group_by(ProfilingResult.model_id, ProfilingResult.task_type)
    )).all()
    for model_id, task, avg, n in rows:
        stmt = insert(ModelPerformance).values(
            model_id=model_id, task_type=task, quality_score=round(float(avg), 3),
            quality_samples=n)
        await db.execute(stmt.on_conflict_do_update(
            index_elements=["model_id", "task_type"],
            set_={"quality_score": stmt.excluded.quality_score,
                  "quality_samples": stmt.excluded.quality_samples, "updated_at": func.now()}))
    await db.commit()
    return len(rows)
