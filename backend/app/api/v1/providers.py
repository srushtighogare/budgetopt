"""Provider credential management (admin only). Keys are write-only via the API."""

from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import require_admin
from app.core.security import encrypt_secret
from app.db.models import AuditEvent, LLMModel, Provider, User
from app.db.session import get_db
from app.services.providers.base import ChatMessage
from app.services.providers.registry import supported_providers
from app.services.routing.executor import Attempt, execute_with_fallback

router = APIRouter(tags=["providers"])


class ProviderOut(BaseModel):
    name: str
    is_active: bool
    has_api_key: bool
    adapter_available: bool


class ProviderKeyIn(BaseModel):
    api_key: str = Field(min_length=8, max_length=500)


class ProviderUpdate(BaseModel):
    is_active: bool


class ModelTestOut(BaseModel):
    ok: bool
    text: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float | None
    cost: Decimal
    attempts: list[dict]


def _out(p: Provider) -> ProviderOut:
    return ProviderOut(name=p.name, is_active=p.is_active, has_api_key=bool(p.encrypted_api_key),
                       adapter_available=p.name in supported_providers())


async def _provider(db: AsyncSession, name: str) -> Provider:
    p = await db.scalar(select(Provider).where(Provider.name == name))
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Provider not found")
    return p


@router.get("/providers", response_model=list[ProviderOut])
async def list_providers(
    _: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> list[ProviderOut]:
    return [_out(p) for p in await db.scalars(select(Provider).order_by(Provider.id))]


@router.put("/providers/{name}/key", response_model=ProviderOut)
async def set_provider_key(
    name: str, body: ProviderKeyIn,
    user: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> ProviderOut:
    p = await _provider(db, name)
    p.encrypted_api_key = encrypt_secret(body.api_key.strip())
    db.add(AuditEvent(user_id=user.id, action="set_provider_key", resource="provider",
                      resource_id=name))
    await db.commit()
    return _out(p)


@router.delete("/providers/{name}/key", response_model=ProviderOut)
async def remove_provider_key(
    name: str, user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> ProviderOut:
    p = await _provider(db, name)
    p.encrypted_api_key = None
    db.add(AuditEvent(user_id=user.id, action="remove_provider_key", resource="provider",
                      resource_id=name))
    await db.commit()
    return _out(p)


@router.patch("/providers/{name}", response_model=ProviderOut)
async def update_provider(
    name: str, body: ProviderUpdate,
    user: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> ProviderOut:
    p = await _provider(db, name)
    p.is_active = body.is_active
    db.add(AuditEvent(user_id=user.id, action="update_provider", resource="provider",
                      resource_id=name, details={"is_active": body.is_active}))
    await db.commit()
    return _out(p)


@router.post("/models/{model_id}/test", response_model=ModelTestOut)
async def test_model(
    model_id: int, _: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> ModelTestOut:
    """Connectivity check: one tiny real call to the model (costs a fraction of a cent)."""
    model = await db.scalar(
        select(LLMModel).options(selectinload(LLMModel.provider)).where(LLMModel.id == model_id)
    )
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    res = await execute_with_fallback(
        db, [model], [ChatMessage("user", "Reply with exactly the word: OK")],
        max_tokens=256, timeout=30,
    )
    c = res.completion
    return ModelTestOut(
        ok=res.succeeded, text=c.text if c else None,
        input_tokens=c.input_tokens if c else None, output_tokens=c.output_tokens if c else None,
        latency_ms=c.latency_ms if c else None, cost=res.cost,
        attempts=[_attempt(a) for a in res.attempts],
    )


def _attempt(a: Attempt) -> dict:
    return {k: v for k, v in a.__dict__.items() if v is not None}
