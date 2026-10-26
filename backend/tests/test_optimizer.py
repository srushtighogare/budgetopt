"""Optimizer: problem constraints, NSGA-II front quality, exact ILP/LP, policies,
and end-to-end policy routing through the gateway."""

import random
from decimal import Decimal

import numpy as np
import pytest
from sqlalchemy import select

from app.db.models import (
    LLMModel,
    ModelPerformance,
    Request,
    RequestStatus,
    RoutingDecision,
    TaskType,
    Team,
)
from app.services.optimizer import ilp, nsga2
from app.services.optimizer.policy import allocation_to_rules, pick_order
from app.services.optimizer.problem import (
    BANDS,
    TASKS,
    ModelProfile,
    OptimizationInputs,
    WorkloadClass,
)
from app.services.providers.base import Completion, ProviderAdapter
from app.services.routing import pipeline
from app.services.routing.breaker import CircuitBreaker
from tests.conftest import login, make_user


def profile(mid, name, p_in, p_out, q, lat=1000.0, privacy=0, ctx=131_072):
    qd = q if isinstance(q, dict) else {t: q for t in TASKS}
    return ModelProfile(mid, name, name, p_in, p_out, ctx, privacy, qd, frozenset(),
                        {t: lat for t in TASKS})


CLASSES = [WorkloadClass(t, b, 1 / 24, 200, 400) for t in TASKS for b in BANDS]
MODELS = [profile(0, "cheap", 0.1, 0.4, 6.5, 800), profile(1, "mid", 0.3, 1.2, 7.5, 1500),
          profile(2, "premium", 2.0, 8.0, 9.0, 4000)]


def inputs(**kw) -> OptimizationInputs:
    return OptimizationInputs(classes=kw.pop("classes", CLASSES),
                              models=kw.pop("models", MODELS), monthly_requests=100_000, **kw)


# --- problem ---

def test_quality_floor_per_band_controls_eligibility():
    inp = inputs()
    low = next(i for i, c in enumerate(inp.classes) if c.band == "low")
    high = next(i for i, c in enumerate(inp.classes) if c.band == "high")
    assert inp.eligible[low].tolist() == [True, True, True]    # q_min 6
    assert inp.eligible[high].tolist() == [False, False, True]  # q_min 8
    assert "quality 6.5 < 8.0" in inp.reasons[(high, 0)]


def test_unservable_class_is_relaxed_to_best_model_with_note():
    inp = inputs(models=MODELS[:2])  # nobody reaches 8.0 for 'high'
    high = next(i for i, c in enumerate(inp.classes) if c.band == "high")
    assert inp.eligible[high].tolist() == [False, True]
    assert any("relaxed to best available (7.5)" in n for n in inp.notes)


def test_privacy_and_latency_constraints():
    models = [profile(0, "public", 0.1, 0.4, 9.0, 800, privacy=0),
              profile(1, "internal", 0.3, 1.2, 9.0, 5000, privacy=1)]
    inp = inputs(models=models, team_privacy_rank=1)
    assert not inp.eligible[:, 0].any() and inp.eligible[:, 1].all()
    inp = inputs(models=models, latency_max_ms=2000)
    assert inp.eligible[:, 0].all() and not inp.eligible[:, 1].any()


def test_evaluate_matches_hand_calculation():
    inp = inputs(classes=[WorkloadClass("coding", "low", 1.0, 1000, 1000)])
    X = np.array([[0.5, 0.5, 0.0]])
    cost, q, lat = inp.evaluate(X)
    # per request: 0.5*(1000*0.1+1000*0.4)/1e6 + 0.5*(1000*0.3+1000*1.2)/1e6 = 0.001
    assert cost == pytest.approx(100_000 * 0.001)
    assert q == pytest.approx(7.0) and lat == pytest.approx(1150)


# --- NSGA-II ---

def test_decode_produces_valid_allocations():
    inp = inputs()
    rng = np.random.default_rng(0)
    for _ in range(50):
        X = nsga2.decode(rng.random(int(inp.eligible.sum())), inp)
        assert np.allclose(X.sum(axis=1), 1.0)
        assert (X[~inp.eligible] == 0).all()
        assert ((X == 0) | (X >= nsga2.MIN_SHARE - 1e-12)).all()


@pytest.fixture(scope="module")
def front():
    return inputs(), nsga2.optimize(inputs(), pop_size=60, generations=80)


def test_front_is_non_dominated_and_presets_ordered(front):
    _, fr = front
    F = np.array([[s.monthly_cost, -s.quality, s.latency_ms] for s in fr.solutions])
    for a in F:
        dominated = ((a >= F).all(axis=1) & (a > F).any(axis=1)).any()
        assert not dominated
    s = fr.solutions
    assert s[fr.presets["cost_saver"]].monthly_cost <= s[fr.presets["balanced"]].monthly_cost
    assert s[fr.presets["balanced"]].quality <= s[fr.presets["quality_first"]].quality
    assert 0 < fr.hypervolume <= 1.1 ** 3


def test_nsga2_close_to_exact_lp_optimum(front):
    inp, fr = front
    budget = fr.solutions[len(fr.solutions) // 2].monthly_cost
    lp = ilp.solve(inp, "max_quality", budget=budget, relax=True)
    best = max(s.quality for s in fr.solutions if s.monthly_cost <= budget + 1e-9)
    assert lp.status == "optimal" and lp.monthly_cost <= budget + 1e-6
    assert (lp.quality - best) / lp.quality < 0.02


def test_nsga2_respects_budget():
    # The cheapest allowed allocation costs $144/month (high-band traffic needs premium)
    inp = inputs(budget=200.0)
    fr = nsga2.optimize(inp, pop_size=40, generations=40)
    assert all(s.monthly_cost <= 200.0 + 1e-9 and s.feasible for s in fr.solutions)


def test_nsga2_reports_infeasible_budget():
    fr = nsga2.optimize(inputs(budget=50.0), pop_size=30, generations=20)
    assert len(fr.solutions) == 1 and not fr.solutions[0].feasible
    assert fr.solutions[0].monthly_cost == pytest.approx(144.0)


# --- ILP ---

def test_ilp_objectives_and_infeasibility():
    inp = inputs()
    r = ilp.solve(inp, "max_quality", budget=250.0)
    assert r.status == "optimal" and r.monthly_cost <= 250.0 + 1e-6
    assert set(np.unique(r.allocation)) <= {0.0, 1.0}  # one model per class
    r = ilp.solve(inp, "min_cost", quality_floor=8.0)
    assert r.quality >= 8.0 - 1e-9
    assert ilp.solve(inp, "max_quality", budget=100.0).status == "infeasible"


# --- policies ---

def test_rules_and_weighted_pick():
    inp = inputs(classes=[WorkloadClass("coding", "low", 1.0, 100, 100)])
    rules = allocation_to_rules(inp, np.array([[0.7, 0.3, 0.0]]))
    assert rules == {"coding:low": [{"model_id": 0, "model": "cheap", "share": 0.7},
                                    {"model_id": 1, "model": "mid", "share": 0.3}]}
    rng = random.Random(0)
    firsts = [pick_order(rules["coding:low"], {0, 1}, rng)[0] for _ in range(4000)]
    assert 0.66 < firsts.count(0) / len(firsts) < 0.74
    assert pick_order(rules["coding:low"], {1}, rng) == [1]  # ineligible model dropped
    assert pick_order(rules["coding:low"], set(), rng) == []


# --- end to end: run -> apply -> gateway follows the policy ---

class Fake(ProviderAdapter):
    def __init__(self):
        super().__init__(api_key="x")
        self.calls: list[str] = []

    async def generate(self, model, messages, **kw):
        self.calls.append(model)
        return Completion("ok", 100, 100, 10.0)

    def build_request(self, *a):  # pragma: no cover
        raise NotImplementedError

    def parse_response(self, *a):  # pragma: no cover
        raise NotImplementedError


async def _seed_profiles(db):
    """Quality makes the cheapest model unfit for coding, so policies must avoid it there."""
    models = {m.name: m for m in await db.scalars(select(LLMModel))}
    for name, m in models.items():
        for t in TaskType:
            q = 9.5 if "120b" in name else 7.5
            if name == "openai/gpt-oss-20b":
                q = 3.0 if t == TaskType.CODING else 8.5
            db.add(ModelPerformance(model_id=m.id, task_type=t, quality_score=q,
                                    quality_samples=8, latency_p50_ms=900,
                                    latency_p95_ms=1500))
    await db.commit()


async def test_run_apply_and_route_by_policy(client, admin_headers, db, monkeypatch):
    await _seed_profiles(db)
    fake = Fake()
    monkeypatch.setattr(pipeline, "build_adapter", lambda p: fake)
    monkeypatch.setattr(pipeline, "default_breaker", CircuitBreaker())
    eng = await db.scalar(select(Team).where(Team.name == "Engineering"))

    r = await client.post("/api/v1/optimization/runs", headers=admin_headers, json={
        "team_id": eng.id, "monthly_requests": 50_000, "pop_size": 40, "generations": 30})
    assert r.status_code == 201, r.text
    run = r.json()
    assert set(run["presets"]) == {"cost_saver", "balanced", "quality_first"}
    assert run["metrics"]["lp_validation"] and run["traffic_source"].startswith("uniform")
    assert any(b["name"].startswith("all traffic to") for b in run["baselines"])

    applied = await client.post(f"/api/v1/optimization/runs/{run['id']}/apply",
                                headers=admin_headers,
                                json={"team_id": eng.id, "preset": "cost_saver"})
    assert applied.status_code == 200, applied.text
    rules = applied.json()["rules"]
    assert all(r["model"] != "openai/gpt-oss-20b" for r in rules["coding:low"])
    await db.refresh(eng)
    assert eng.policy_id == applied.json()["policy_id"]

    key = (await client.post(f"/api/v1/teams/{eng.id}/api-keys", headers=admin_headers,
                             json={"name": "p"})).json()["key"]
    res = await client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                            json={"messages": [{"role": "user", "content":
                                  "Write a Python function that reverses a linked list."}]})
    meta = res.json()["budgetopt"]
    assert "policy 'cost_saver (run" in meta["explanation"]
    assert fake.calls[0] != "openai/gpt-oss-20b"  # policy keeps coding off the weak model
    d = await db.scalar(select(RoutingDecision).where(
        RoutingDecision.request_id == meta["request_id"]))
    assert d.policy_id == eng.policy_id


async def test_savings_policy_used_at_soft_alert(client, admin_headers, db, monkeypatch):
    await _seed_profiles(db)
    fake = Fake()
    monkeypatch.setattr(pipeline, "build_adapter", lambda p: fake)
    monkeypatch.setattr(pipeline, "default_breaker", CircuitBreaker())
    eng = await db.scalar(select(Team).where(Team.name == "Engineering"))
    run = (await client.post("/api/v1/optimization/runs", headers=admin_headers, json={
        "team_id": eng.id, "pop_size": 30, "generations": 20})).json()
    for preset, slot in (("quality_first", "primary"), ("cost_saver", "savings")):
        await client.post(f"/api/v1/optimization/runs/{run['id']}/apply", headers=admin_headers,
                          json={"team_id": eng.id, "preset": preset, "slot": slot})
    db.add(Request(team_id=eng.id, prompt_hash="0" * 64, cost=Decimal("4.2"),
                   status=RequestStatus.SUCCESS))  # 84% of the 5.00 budget
    await db.commit()
    key = (await client.post(f"/api/v1/teams/{eng.id}/api-keys", headers=admin_headers,
                             json={"name": "s"})).json()["key"]
    res = await client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {key}"},
                            json={"messages": [{"role": "user", "content": "Hi there"}]})
    assert "policy 'cost_saver" in res.json()["budgetopt"]["explanation"]


async def test_ilp_run_and_permissions(client, admin_headers, db):
    await _seed_profiles(db)
    r = await client.post("/api/v1/optimization/runs", headers=admin_headers,
                          json={"method": "ilp", "ilp_objective": "min_cost",
                                "quality_floor": 8.0, "monthly_requests": 10_000})
    assert r.status_code == 201 and r.json()["metrics"]["status"] == "optimal"
    assert r.json()["presets"]["optimal"]["quality"] >= 8.0 - 1e-9

    await make_user(db, "m@demo.io", "member", "Engineering")
    member = await login(client, "m@demo.io")
    assert (await client.post("/api/v1/optimization/runs", headers=member,
                              json={})).status_code == 403
    assert (await client.get(f"/api/v1/optimization/runs/{r.json()['id']}",
                             headers=member)).status_code == 403
    bad = await client.post(f"/api/v1/optimization/runs/{r.json()['id']}/apply",
                            headers=admin_headers, json={"team_id": 1})
    assert bad.status_code == 422


async def test_infeasible_budget_is_explained(client, admin_headers, db):
    await _seed_profiles(db)
    r = (await client.post("/api/v1/optimization/runs", headers=admin_headers, json={
        "monthly_requests": 1_000_000, "budget": 0.5, "pop_size": 20, "generations": 10})).json()
    assert r["presets"]["cost_saver"]["feasible"] is False
    assert any("cannot cover 1,000,000 requests" in n for n in r["notes"])
