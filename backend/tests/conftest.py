"""Test fixtures. Tests run against a real PostgreSQL database (TEST_DATABASE_URL),
recreated from the ORM metadata for every test so each one starts clean."""

import os

os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://budgetopt:budgetopt@localhost:5433/budgetopt_test",
)
os.environ.setdefault("FERNET_KEY", "TlUtFm5JnEXw2b0iY4Yqk0ZfVxH0lH0nZ8j3w3G0p3Q=")

from collections.abc import AsyncIterator  # noqa: E402

import pytest  # noqa: E402
from httpx import ASGITransport, AsyncClient  # noqa: E402
from sqlalchemy import select, text  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.db.models import Base, Organization, Role, Team, User  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.seed import seed  # noqa: E402

PASSWORD = "Password123"


@pytest.fixture(autouse=True)
async def clean_db() -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
        await conn.run_sync(Base.metadata.create_all)
    async with SessionLocal() as db:
        await seed(db)
    yield


@pytest.fixture
async def db() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def make_user(db: AsyncSession, email: str, role: Role, team_name: str | None) -> User:
    org = await db.scalar(select(Organization))
    team_id = None
    if team_name:
        team_id = await db.scalar(select(Team.id).where(Team.name == team_name))
    user = User(organization_id=org.id, team_id=team_id, email=email, full_name=email,
                password_hash=hash_password(PASSWORD), role=role)
    db.add(user)
    await db.commit()
    return user


async def login(client: AsyncClient, email: str, password: str = PASSWORD) -> dict[str, str]:
    r = await client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
async def admin_headers(client: AsyncClient) -> dict[str, str]:
    s = get_settings()
    return await login(client, s.seed_admin_email, s.seed_admin_password)
