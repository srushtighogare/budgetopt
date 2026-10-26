"""Generate synthetic request history for demonstrating forecasting and anomaly detection.

Creates (or resets) a separate team, "Demo (synthetic traffic)", so real teams'
data is never touched, and fills it with ~6 weeks of requests:
  - weekday/weekend pattern and steady growth
  - a realistic task mix, priced with each model's real current pricing, routed mostly
    to cheaper models (as BudgetOpt would) with a routing decision and savings vs the
    priciest-model baseline on every row, and ~5% cache hits
  - one API key whose usage spikes in the last complete hour (for the anomaly demo)

Every row is marked synthetic (prompt hash of "synthetic-demo").

The demo budget defaults to 90% of this month's expected spend, so the forecast
shows a likely overrun; pass --budget to set it explicitly.

Usage: python -m app.demo_traffic [--days 42] [--budget 6.5]
"""

import argparse
import asyncio
import hashlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
from sqlalchemy import delete, select
from sqlalchemy.orm import selectinload

from app.core.security import generate_api_key
from app.db.models import (
    Alert,
    ApiKey,
    ComplexityBand,
    Forecast,
    LLMModel,
    Organization,
    Request,
    RequestStatus,
    RoutingDecision,
    SemanticCacheEntry,
    TaskType,
    Team,
    User,
)
from app.db.session import SessionLocal
from app.services.pricing import compute_cost

DEMO_TEAM = "Demo (synthetic traffic)"
SYNTHETIC_HASH = hashlib.sha256(b"synthetic-demo").hexdigest()
TASK_MIX = {"simple_qa": 0.22, "summarization": 0.14, "coding": 0.18, "extraction": 0.12,
            "creative_writing": 0.1, "analysis": 0.1, "translation": 0.08,
            "reasoning_math": 0.06}
OUT_TOKENS = {"translation": 60, "summarization": 150, "simple_qa": 200, "extraction": 180,
              "creative_writing": 450, "coding": 600, "reasoning_math": 700, "analysis": 650}


async def generate(days: int, budget: float | None = None, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    async with SessionLocal() as db:
        org = await db.scalar(select(Organization).order_by(Organization.id))
        admin = await db.scalar(select(User).order_by(User.id))
        team = await db.scalar(select(Team).where(Team.name == DEMO_TEAM))
        if team:  # reset previous demo data
            ids = select(Request.id).where(Request.team_id == team.id)
            await db.execute(delete(RoutingDecision).where(RoutingDecision.request_id.in_(ids)))
            for model in (SemanticCacheEntry, Request, Forecast, Alert):
                await db.execute(delete(model).where(model.team_id == team.id))
        else:
            team = Team(organization_id=org.id, name=DEMO_TEAM, monthly_budget=Decimal(1),
                        alert_threshold=0.8)
            db.add(team)
        await db.flush()

        keys = []
        for name in ("support-bot", "analytics-batch", "dev-tools"):
            k = await db.scalar(select(ApiKey).where(ApiKey.team_id == team.id,
                                                     ApiKey.name == name))
            if k is None:
                _, prefix, key_hash = generate_api_key()
                k = ApiKey(team_id=team.id, created_by=admin.id, name=name, prefix=prefix,
                           key_hash=key_hash)
                db.add(k)
                await db.flush()
            keys.append(k)

        models = list(await db.scalars(select(LLMModel).options(
            selectinload(LLMModel.pricing)).where(LLMModel.is_active)))
        now = datetime.now(UTC)
        pricing = {m.id: next(p for p in m.pricing if p.effective_from <= now) for m in models}
        tasks, probs = list(TASK_MIX), np.array(list(TASK_MIX.values()))
        unit = np.array([float(pricing[m.id].input_per_1m + 3 * pricing[m.id].output_per_1m)
                         for m in models])
        pick = (1 / unit) / (1 / unit).sum()  # cheaper models get more traffic
        priciest = models[int(np.argmax(unit))]

        def add_request(ts: datetime, key: ApiKey) -> Decimal:
            task = tasks[rng.choice(len(tasks), p=probs)]
            band = rng.choice(["low", "medium", "high"], p=[0.5, 0.35, 0.15])
            model = models[rng.choice(len(models), p=pick)]
            tin = int(rng.lognormal(np.log(200), 0.6))
            tout = int(rng.lognormal(np.log(OUT_TOKENS[task]), 0.5))
            full = compute_cost(pricing[model.id], tin, tout)
            baseline = compute_cost(pricing[priciest.id], tin, tout)
            cached = rng.random() < 0.05
            req = Request(team_id=team.id, api_key_id=key.id, prompt_hash=SYNTHETIC_HASH,
                          task_type=TaskType(task), complexity_band=ComplexityBand(band),
                          complexity={"low": 0.2, "medium": 0.5, "high": 0.8}[band],
                          est_input_tokens=tin, est_output_tokens=tout, model_id=model.id,
                          input_tokens=0 if cached else tin, output_tokens=0 if cached else tout,
                          cost=Decimal(0) if cached else full,
                          latency_ms=None if cached else float(rng.lognormal(np.log(1200), 0.4)),
                          status=RequestStatus.CACHE_HIT if cached else RequestStatus.SUCCESS,
                          created_at=ts)
            req.decision = RoutingDecision(
                budget_state="ok", candidates={"synthetic": True}, created_at=ts,
                baseline_model_id=priciest.id,
                baseline_cost=full if cached else baseline,
                savings=full if cached else baseline - full,
                explanation="Synthetic demo request" + (" (cache hit)" if cached else "."))
            db.add(req)
            return req.cost

        n, total = 0, Decimal(0)
        start = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        spike_hour = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        hour = start
        while hour < now - timedelta(hours=1):
            day_idx = (hour - start).days
            weekday = hour.weekday() < 5
            work = 8 <= hour.hour < 20
            # ~120 requests/weekday growing ~1.5%/day, far fewer at night and weekends
            rate = 10 * (1.015 ** day_idx) * (1.0 if weekday else 0.35) * (1.0 if work else 0.15)
            for _ in range(rng.poisson(rate)):
                key = keys[rng.choice(3, p=[0.5, 0.2, 0.3])]
                total += add_request(hour + timedelta(seconds=int(rng.integers(3600))), key)
                n += 1
            hour += timedelta(hours=1)
        # anomaly: the "dev-tools" key bursts in the last complete hour
        for _ in range(300):
            total += add_request(spike_hour + timedelta(seconds=int(rng.integers(3600))), keys[2])
            n += 1
        if budget is None:
            # 90% of the month's run-rate projection, so an overrun is likely but not certain
            period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            month_so_far = sum((r.cost for r in db.new if isinstance(r, Request)
                                and r.created_at >= period_start), Decimal(0))
            days_in_month = ((period_start + timedelta(days=32)).replace(day=1)
                             - period_start).days
            elapsed = max(1.0, (now - period_start).total_seconds() / 86400)
            budget = round(0.9 * float(month_so_far) / elapsed * days_in_month, 2)
        team.monthly_budget = Decimal(str(budget))
        await db.commit()
        return {"team_id": team.id, "requests": n, "total_cost": float(total), "days": days,
                "monthly_budget": budget}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=42)
    ap.add_argument("--budget", type=float, help="default: 90%% of expected month spend")
    args = ap.parse_args()
    print(await generate(args.days, args.budget))


if __name__ == "__main__":
    asyncio.run(main())
