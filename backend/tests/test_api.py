"""API tests: authentication, RBAC (incl. the report's cross-team access example),
API keys, versioned pricing and spend tracking."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sqlalchemy import select

from app.db.models import (
    AuditEvent,
    LLMModel,
    Organization,
    Request,
    RequestStatus,
    Role,
    Team,
)
from app.seed import MODELS as SEED_MODELS
from app.services.pricing import price_at
from tests.conftest import login, make_user


async def test_health(client):
    assert (await client.get("/health")).json() == {"status": "ok"}


async def test_login_and_me(client, admin_headers):
    r = await client.get("/api/v1/auth/me", headers=admin_headers)
    assert r.status_code == 200
    assert r.json()["role"] == "org_admin"


async def test_login_wrong_password_and_unknown_email_look_identical(client):
    bad_pw = await client.post("/api/v1/auth/login",
                               data={"username": "admin@example.com", "password": "nope"})
    unknown = await client.post("/api/v1/auth/login",
                                data={"username": "ghost@x.com", "password": "nope"})
    assert bad_pw.status_code == unknown.status_code == 401
    assert bad_pw.json() == unknown.json()


async def test_requires_auth(client):
    assert (await client.get("/api/v1/teams")).status_code == 401
    r = await client.get("/api/v1/teams", headers={"Authorization": "Bearer garbage"})
    assert r.status_code == 401


async def test_refresh_token_flow(client):
    r = await client.post("/api/v1/auth/login",
                          data={"username": "admin@example.com", "password": "ChangeMe!2026"})
    tokens = r.json()
    # A refresh token must not work as an access token
    me = await client.get("/api/v1/auth/me",
                          headers={"Authorization": f"Bearer {tokens['refresh_token']}"})
    assert me.status_code == 401
    r2 = await client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r2.status_code == 200 and r2.json()["access_token"]


async def test_admin_creates_team_and_validation(client, admin_headers):
    r = await client.post("/api/v1/teams", headers=admin_headers,
                          json={"name": "Research", "monthly_budget": "25.5"})
    assert r.status_code == 201, r.text
    assert Decimal(r.json()["monthly_budget"]) == Decimal("25.5")

    dup = await client.post("/api/v1/teams", headers=admin_headers,
                            json={"name": "Research", "monthly_budget": "1"})
    assert dup.status_code == 409
    bad = await client.post("/api/v1/teams", headers=admin_headers,
                            json={"name": "X", "monthly_budget": "-5"})
    assert bad.status_code == 422


async def test_member_cannot_view_other_teams_spend(client, db):
    """Report, Safety & Security: an employee accessing another team's AI expenditure
    must be verified and denied."""
    await make_user(db, "dev@demo.io", Role.MEMBER, "Engineering")
    headers = await login(client, "dev@demo.io")
    eng = await db.scalar(select(Team).where(Team.name == "Engineering"))
    mkt = await db.scalar(select(Team).where(Team.name == "Marketing"))

    assert (await client.get(f"/api/v1/teams/{eng.id}/spend", headers=headers)).status_code == 200
    denied = await client.get(f"/api/v1/teams/{mkt.id}/spend", headers=headers)
    assert denied.status_code == 403

    audit = await db.scalar(select(AuditEvent).where(AuditEvent.allowed.is_(False)))
    assert audit is not None and audit.resource_id == str(mkt.id)

    listed = (await client.get("/api/v1/teams", headers=headers)).json()
    assert [t["name"] for t in listed] == ["Engineering"]


async def test_other_organization_team_is_hidden(client, db, admin_headers):
    other = Organization(name="Other Corp")
    db.add(other)
    await db.flush()
    foreign = Team(organization_id=other.id, name="Secret", monthly_budget=Decimal(1))
    db.add(foreign)
    await db.commit()
    r = await client.get(f"/api/v1/teams/{foreign.id}", headers=admin_headers)
    assert r.status_code == 404


async def test_member_cannot_create_team_or_model(client, db):
    await make_user(db, "m@demo.io", Role.MEMBER, "Support")
    headers = await login(client, "m@demo.io")
    r = await client.post("/api/v1/teams", headers=headers,
                          json={"name": "Hack", "monthly_budget": "1"})
    assert r.status_code == 403


async def test_manager_can_only_add_members_to_own_team(client, db):
    await make_user(db, "lead@demo.io", Role.TEAM_MANAGER, "Engineering")
    headers = await login(client, "lead@demo.io")
    eng_id = await db.scalar(select(Team.id).where(Team.name == "Engineering"))
    mkt_id = await db.scalar(select(Team.id).where(Team.name == "Marketing"))
    body = {"email": "new@demo.io", "full_name": "New", "password": "LongPass123"}

    ok = await client.post("/api/v1/users", headers=headers, json={**body, "team_id": eng_id})
    assert ok.status_code == 201
    other = await client.post("/api/v1/users", headers=headers,
                              json={**body, "email": "x@demo.io", "team_id": mkt_id})
    assert other.status_code == 403
    escalate = await client.post("/api/v1/users", headers=headers,
                                 json={**body, "email": "y@demo.io", "team_id": eng_id,
                                       "role": "org_admin"})
    assert escalate.status_code == 403


async def test_weak_password_rejected(client, admin_headers, db):
    eng_id = await db.scalar(select(Team.id).where(Team.name == "Engineering"))
    r = await client.post("/api/v1/users", headers=admin_headers,
                          json={"email": "w@demo.io", "full_name": "W",
                                "password": "onlyletters", "team_id": eng_id})
    assert r.status_code == 422


async def test_api_key_shown_once(client, admin_headers, db):
    eng_id = await db.scalar(select(Team.id).where(Team.name == "Engineering"))
    created = await client.post(f"/api/v1/teams/{eng_id}/api-keys", headers=admin_headers,
                                json={"name": "chatbot"})
    assert created.status_code == 201
    key = created.json()["key"]
    assert key.startswith("bo_")

    listed = await client.get(f"/api/v1/teams/{eng_id}/api-keys", headers=admin_headers)
    assert "key" not in listed.json()[0]
    assert key not in listed.text

    kid = created.json()["id"]
    r = await client.delete(f"/api/v1/teams/{eng_id}/api-keys/{kid}", headers=admin_headers)
    assert r.status_code == 204
    listed = await client.get(f"/api/v1/teams/{eng_id}/api-keys", headers=admin_headers)
    assert listed.json()[0]["is_active"] is False


async def test_models_listed_with_pricing(client, admin_headers):
    models = (await client.get("/api/v1/models", headers=admin_headers)).json()
    assert len(models) == len(SEED_MODELS)
    assert all(m["current_pricing"] is not None for m in models)


async def test_pricing_is_versioned(client, admin_headers, db):
    model = await db.scalar(select(LLMModel).where(LLMModel.name == "openai/gpt-oss-120b"))
    before = datetime.now(UTC)
    future = before + timedelta(days=1)
    r = await client.post(f"/api/v1/models/{model.id}/pricing", headers=admin_headers,
                          json={"input_per_1m": "0.20", "output_per_1m": "0.80",
                                "effective_from": future.isoformat()})
    assert r.status_code == 201

    assert (await price_at(db, model.id, before)).input_per_1m == Decimal("0.15")
    assert (await price_at(db, model.id, future + timedelta(hours=1))).input_per_1m == Decimal(
        "0.20"
    )
    history = await client.get(f"/api/v1/models/{model.id}/pricing", headers=admin_headers)
    assert len(history.json()) == 2


async def test_spend_and_budget_state(client, admin_headers, db):
    mkt = await db.scalar(select(Team).where(Team.name == "Marketing"))  # budget 0.50
    last_month = datetime.now(UTC).replace(day=1) - timedelta(days=2)
    db.add_all([
        Request(team_id=mkt.id, prompt_hash="a" * 64, cost=Decimal("0.30"),
                status=RequestStatus.SUCCESS),
        Request(team_id=mkt.id, prompt_hash="b" * 64, cost=Decimal("0.12"),
                status=RequestStatus.SUCCESS),
        # Last month's spend must not count against this month's budget
        Request(team_id=mkt.id, prompt_hash="c" * 64, cost=Decimal("5"),
                status=RequestStatus.SUCCESS, created_at=last_month),
    ])
    await db.commit()

    r = (await client.get(f"/api/v1/teams/{mkt.id}/spend", headers=admin_headers)).json()
    assert Decimal(r["spend"]) == Decimal("0.42")
    assert r["budget_state"] == "soft_alert"  # 84% of budget
