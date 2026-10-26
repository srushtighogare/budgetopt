"""What-if replay and savings report."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import select

from app.db.models import (
    ComplexityBand,
    LLMModel,
    ModelPerformance,
    ProfilingResult,
    Request,
    RequestStatus,
    Role,
    RoutingDecision,
    TaskType,
    Team,
)
from tests.conftest import login, make_user

CHEAP, MID, PREMIUM = "openai/gpt-oss-20b", "openai/gpt-oss-120b", "gemini-3.5-flash"


async def _setup(db) -> tuple[Team, dict[str, LLMModel]]:
    models = {m.name: m for m in await db.scalars(select(LLMModel))}
    for name, m in models.items():
        for t in TaskType:
            q = {CHEAP: 3.0 if t == TaskType.CODING else 8.5, MID: 9.0}.get(name, 7.5)
            db.add(ModelPerformance(model_id=m.id, task_type=t, quality_score=q,
                                    quality_samples=8, latency_p95_ms=1000))
    team = await db.scalar(select(Team).where(Team.name == "Engineering"))
    now = datetime.now(UTC)
    # 10 coding + 10 summarization requests, all served by the cheap model, 1000 in / 500 out
    for i in range(20):
        task = TaskType.CODING if i < 10 else TaskType.SUMMARIZATION
        db.add(Request(team_id=team.id, prompt_hash="w" * 64, task_type=task,
                       complexity_band=ComplexityBand.LOW, model_id=models[CHEAP].id,
                       input_tokens=1000, output_tokens=500, cost=Decimal("0.000225"),
                       status=RequestStatus.SUCCESS, created_at=now - timedelta(hours=i)))
    await db.commit()
    return team, models


def per_request(p_in, p_out, tin=1000, tout=500) -> float:
    return (tin * p_in + tout * p_out) / 1e6


async def _whatif(client, headers, team, *scenarios):
    r = await client.post("/api/v1/whatif", headers=headers,
                          json={"team_id": team.id, "scenarios": list(scenarios)})
    assert r.status_code == 200, r.text
    return {s["name"]: s for s in r.json()["scenarios"]}


async def test_replay_costs_match_hand_calculation(client, admin_headers, db):
    team, _ = await _setup(db)
    res = await _whatif(client, admin_headers, team,
                        {"name": "actual"},
                        {"name": "all mid", "allocation": "model", "model": MID},
                        {"name": "mid, prices x2", "allocation": "model", "model": MID,
                         "price_multipliers": {MID: 2.0}},
                        {"name": "actual +50% volume", "volume_growth_pct": 50})
    assert res["actual"]["cost"] == pytest.approx(20 * per_request(0.075, 0.30))
    assert res["all mid"]["cost"] == pytest.approx(20 * per_request(0.15, 0.60))
    assert res["mid, prices x2"]["cost"] == pytest.approx(2 * res["all mid"]["cost"])
    assert res["actual +50% volume"]["cost"] == pytest.approx(1.5 * res["actual"]["cost"])
    base = next(s for s in res.values() if s["name"].startswith("baseline"))
    assert base["cost"] == pytest.approx(20 * per_request(1.50, 9.00))
    assert res["actual"]["savings_vs_baseline_pct"] > 90
    assert res["all mid"]["mean_quality"] == pytest.approx(9.0)
    # actual: coding answered by a model scoring 3.0 there -> quality is visibly worse
    assert res["actual"]["mean_quality"] == pytest.approx((10 * 3.0 + 10 * 8.5) / 20)


async def test_cheapest_respects_quality_floor(client, admin_headers, db):
    team, _ = await _setup(db)
    res = await _whatif(client, admin_headers, team,
                        {"name": "cheapest", "allocation": "cheapest"})
    cheapest = res["cheapest"]
    # coding (cheap model scores 3.0 < 6.0 floor) must move off the cheapest model
    assert cheapest["mean_quality"] >= 6.0
    assert "GPT-OSS 20B (Groq)" in cheapest["cost_by_model"]  # still used for summarization
    assert len(cheapest["cost_by_model"]) == 2


async def test_output_length_scaled_by_model_verbosity(client, admin_headers, db):
    team, models = await _setup(db)
    for i, (name, out) in enumerate([(CHEAP, 100), (MID, 200)] * 3):
        db.add(ProfilingResult(model_id=models[name].id, item_id=f"x{i}",
                               task_type=TaskType.SUMMARIZATION, output_tokens=out, score=8))
    await db.commit()
    res = await _whatif(client, admin_headers, team,
                        {"name": "all mid", "allocation": "model", "model": MID})
    # summarization outputs double (200 vs 100 median), coding unchanged
    expected = 10 * per_request(0.15, 0.60) + 10 * per_request(0.15, 0.60, tout=1000)
    assert res["all mid"]["cost"] == pytest.approx(expected)


async def test_whatif_validation_and_permissions(client, admin_headers, db):
    team, _ = await _setup(db)
    bad = await client.post("/api/v1/whatif", headers=admin_headers, json={
        "team_id": team.id, "scenarios": [{"name": "x", "allocation": "current_policy"}]})
    assert bad.status_code == 422 and "no allocation policy" in bad.text
    bad = await client.post("/api/v1/whatif", headers=admin_headers, json={
        "team_id": team.id, "scenarios": [{"name": "x", "allocation": "model",
                                           "model": "nope"}]})
    assert bad.status_code == 422
    await make_user(db, "m@demo.io", Role.MEMBER, "Engineering")
    member = await login(client, "m@demo.io")
    assert (await client.post("/api/v1/whatif", headers=member, json={
        "team_id": team.id, "scenarios": [{"name": "x"}]})).status_code == 403


async def test_savings_report(client, admin_headers, db):
    team, models = await _setup(db)
    reqs = list(await db.scalars(select(Request).where(Request.team_id == team.id)))
    for r in reqs:
        db.add(RoutingDecision(request_id=r.id, budget_state="ok", candidates={},
                               baseline_cost=Decimal("0.0060"),
                               savings=Decimal("0.0060") - r.cost, explanation="x"))
    hit = Request(team_id=team.id, prompt_hash="c" * 64, status=RequestStatus.CACHE_HIT,
                  cost=Decimal(0), model_id=models[CHEAP].id, task_type=TaskType.CODING)
    db.add(hit)
    await db.flush()
    db.add(RoutingDecision(request_id=hit.id, budget_state="ok", candidates={},
                           baseline_cost=Decimal("0.000225"), savings=Decimal("0.000225"),
                           explanation="cache"))
    await db.commit()

    r = (await client.get(f"/api/v1/reports/savings?team_id={team.id}"
                          "&platform_cost_per_month=0.05", headers=admin_headers)).json()
    assert r["requests"] == 21 and r["cache_hits"] == 1
    assert r["spend"] == pytest.approx(20 * 0.000225)
    assert r["routing_savings"] == pytest.approx(20 * (0.006 - 0.000225))
    assert r["cache_savings"] == pytest.approx(0.000225)
    assert r["savings_pct"] > 90
    feas = r["economic_feasibility"]
    assert feas["economically_feasible"] is True
    assert feas["net_benefit_per_month"] == pytest.approx(r["savings"] - 0.05)
    assert {m["name"] for m in r["by_model"]} == {"GPT-OSS 20B (Groq)"}

    await make_user(db, "m2@demo.io", Role.MEMBER, "Marketing")
    other = await login(client, "m2@demo.io")
    assert (await client.get(f"/api/v1/reports/savings?team_id={team.id}",
                             headers=other)).status_code == 403
    own = (await client.get("/api/v1/reports/savings", headers=other)).json()
    assert own["team_id"] != team.id and own["requests"] == 0
