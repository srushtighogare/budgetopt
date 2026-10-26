"""Model profiling endpoints (admin): quality/latency matrix, runs and raw results."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, require_admin
from app.db.models import AuditEvent, LLMModel, ModelPerformance, ProfilingResult, TaskType, User
from app.db.session import SessionLocal, get_db
from app.services.profiling.cli import DEFAULT_JUDGES
from app.services.profiling.latency import refresh_latency
from app.services.profiling.quality import run_quality_profile

log = logging.getLogger(__name__)
router = APIRouter(prefix="/profiling", tags=["profiling"])

_run_task: asyncio.Task | None = None  # at most one quality run per API process


class QualityRunIn(BaseModel):
    models: list[str] | None = None
    judges: list[str] = Field(default_factory=lambda: list(DEFAULT_JUDGES))
    per_task: int | None = Field(default=None, ge=1, le=50)


@router.get("/summary")
async def summary(
    _: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    """Quality (0–10) and latency per model × task type: the optimizer's inputs."""
    models = {m.id: m for m in await db.scalars(select(LLMModel).order_by(LLMModel.id))}
    perf = list(await db.scalars(select(ModelPerformance)))
    out = []
    for mid, m in models.items():
        cells = {p.task_type.value: {
            "quality": p.quality_score, "samples": p.quality_samples,
            "latency_p50_ms": p.latency_p50_ms, "latency_p95_ms": p.latency_p95_ms,
        } for p in perf if p.model_id == mid}
        out.append({"model_id": mid, "model": m.name, "display_name": m.display_name,
                    "tasks": cells})
    return {"task_types": [t.value for t in TaskType], "models": out,
            "quality_run_active": bool(_run_task and not _run_task.done())}


@router.post("/latency/refresh")
async def latency_refresh(
    days: int = Query(default=14, ge=1, le=90),
    user: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> dict:
    n = await refresh_latency(db, days)
    db.add(AuditEvent(user_id=user.id, action="refresh_latency", resource="profiling"))
    await db.commit()
    return {"updated": n}


async def _run_in_background(body: QualityRunIn) -> None:
    try:
        async with SessionLocal() as db:
            rep = await run_quality_profile(db, model_names=body.models,
                                            judge_names=body.judges, per_task=body.per_task)
            await refresh_latency(db)
        log.info("quality profiling finished: %s", rep)
    except Exception:
        log.exception("quality profiling failed")


@router.post("/quality/run", status_code=status.HTTP_202_ACCEPTED)
async def quality_run(
    body: QualityRunIn, user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> dict:
    """Start a (long, resumable) quality profiling run in the background."""
    global _run_task
    if _run_task and not _run_task.done():
        raise HTTPException(status.HTTP_409_CONFLICT, "A profiling run is already in progress")
    db.add(AuditEvent(user_id=user.id, action="start_quality_profiling", resource="profiling",
                      details=body.model_dump()))
    await db.commit()
    _run_task = asyncio.create_task(_run_in_background(body))
    return {"status": "started"}


@router.get("/results")
async def results(
    model_id: int | None = None, task_type: TaskType | None = None,
    limit: int = Query(default=100, ge=1, le=1000),
    _: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> list[dict]:
    q = select(ProfilingResult).order_by(ProfilingResult.id.desc()).limit(limit)
    if model_id is not None:
        q = q.where(ProfilingResult.model_id == model_id)
    if task_type is not None:
        q = q.where(ProfilingResult.task_type == task_type)
    return [{
        "id": r.id, "model_id": r.model_id, "item_id": r.item_id, "task_type": r.task_type.value,
        "score": r.score, "correct": r.correct, "judge": r.judge, "reason": r.judge_reason,
        "latency_ms": r.latency_ms, "output_tokens": r.output_tokens, "cost": str(r.cost),
        "error": r.error, "answer": r.answer,
    } for r in await db.scalars(q)]
