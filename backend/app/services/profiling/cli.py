"""Command-line entry points for model profiling.

  python -m app.services.profiling.cli quality [--models a,b] [--judges x,y] [--per-task N]
  python -m app.services.profiling.cli latency [--days 14]
  python -m app.services.profiling.cli summary
"""

import argparse
import asyncio
import logging

from sqlalchemy import select

from app.db.models import LLMModel, ModelPerformance
from app.db.session import SessionLocal
from app.services.profiling.latency import refresh_latency
from app.services.profiling.quality import run_quality_profile

DEFAULT_JUDGES = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "gemini-3-flash-preview",
                  "gemini-3.5-flash", "gemini-3.1-flash-lite"]


async def summary() -> None:
    async with SessionLocal() as db:
        names = {m.id: m.name for m in await db.scalars(select(LLMModel))}
        rows = list(await db.scalars(select(ModelPerformance)))
    tasks = sorted({r.task_type.value for r in rows})
    print(f"{'model':26}" + "".join(f"{t[:10]:>12}" for t in tasks))
    for mid in sorted({r.model_id for r in rows}):
        by = {r.task_type.value: r for r in rows if r.model_id == mid}
        cells = []
        for t in tasks:
            r = by.get(t)
            q = f"{r.quality_score:.1f}" if r and r.quality_score is not None else "-"
            lat = f"{r.latency_p50_ms / 1000:.1f}s" if r and r.latency_p50_ms else ""
            cells.append(f"{q + ' ' + lat:>12}")
        print(f"{names[mid][:26]:26}" + "".join(cells))


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quality")
    q.add_argument("--models", help="comma-separated model names (default: all active)")
    q.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    q.add_argument("--per-task", type=int)
    lat = sub.add_parser("latency")
    lat.add_argument("--days", type=int, default=14)
    sub.add_parser("summary")
    args = ap.parse_args()

    if args.cmd == "quality":
        async with SessionLocal() as db:
            rep = await run_quality_profile(
                db, model_names=args.models.split(",") if args.models else None,
                judge_names=args.judges.split(","), per_task=args.per_task)
            await refresh_latency(db)
        print(f"answered {rep.answered}, judged {rep.judged}, already done "
              f"{rep.skipped_existing}, errors {rep.errors}; "
              f"rate-limited this run: {sorted(rep.exhausted_models) or 'none'}")
        await summary()
    elif args.cmd == "latency":
        async with SessionLocal() as db:
            print(f"updated {await refresh_latency(db, args.days)} latency profiles")
    else:
        await summary()


if __name__ == "__main__":
    asyncio.run(main())
