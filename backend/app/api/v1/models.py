"""Model & pricing layer (report architecture item 3)."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import get_current_user, require_admin
from app.db.models import AuditEvent, LLMModel, ModelPerformance, ModelPricing, Provider, User
from app.db.session import get_db
from app.schemas.models import (
    LLMModelCreate,
    LLMModelOut,
    LLMModelUpdate,
    PerformanceOut,
    PricingIn,
    PricingOut,
)

router = APIRouter(prefix="/models", tags=["models"])


async def _load(db: AsyncSession, model_id: int) -> LLMModel:
    model = await db.scalar(
        select(LLMModel)
        .options(selectinload(LLMModel.provider), selectinload(LLMModel.pricing))
        .where(LLMModel.id == model_id)
    )
    if model is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model not found")
    return model


async def _to_out(db: AsyncSession, m: LLMModel) -> LLMModelOut:
    now = datetime.now(UTC)
    current = next((p for p in m.pricing if p.effective_from <= now), None)
    perf = await db.scalars(select(ModelPerformance).where(ModelPerformance.model_id == m.id))
    return LLMModelOut(
        id=m.id, provider=m.provider.name, name=m.name, display_name=m.display_name,
        context_window=m.context_window, max_output_tokens=m.max_output_tokens,
        max_privacy_tier=m.max_privacy_tier, is_active=m.is_active,
        current_pricing=PricingOut.model_validate(current) if current else None,
        performance=[PerformanceOut.model_validate(p) for p in perf],
    )


@router.get("", response_model=list[LLMModelOut])
async def list_models(
    _: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[LLMModelOut]:
    models = await db.scalars(
        select(LLMModel)
        .options(selectinload(LLMModel.provider), selectinload(LLMModel.pricing))
        .order_by(LLMModel.id)
    )
    return [await _to_out(db, m) for m in models]


@router.post("", response_model=LLMModelOut, status_code=status.HTTP_201_CREATED)
async def create_model(
    body: LLMModelCreate, user: User = Depends(require_admin), db: AsyncSession = Depends(get_db)
) -> LLMModelOut:
    provider = await db.scalar(select(Provider).where(Provider.name == body.provider))
    if provider is None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Unknown provider")
    dup = await db.scalar(
        select(LLMModel.id).where(LLMModel.provider_id == provider.id, LLMModel.name == body.name)
    )
    if dup:
        raise HTTPException(status.HTTP_409_CONFLICT, "Model already registered")

    model = LLMModel(
        provider_id=provider.id, name=body.name, display_name=body.display_name,
        context_window=body.context_window, max_output_tokens=body.max_output_tokens,
        max_privacy_tier=body.max_privacy_tier,
    )
    db.add(model)
    await db.flush()
    db.add(ModelPricing(
        model_id=model.id, input_per_1m=body.pricing.input_per_1m,
        output_per_1m=body.pricing.output_per_1m,
        effective_from=body.pricing.effective_from or datetime.now(UTC),
    ))
    db.add(AuditEvent(user_id=user.id, action="create_model", resource="llm_model",
                      resource_id=str(model.id)))
    await db.commit()
    return await _to_out(db, await _load(db, model.id))


@router.patch("/{model_id}", response_model=LLMModelOut)
async def update_model(
    model_id: int, body: LLMModelUpdate,
    user: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> LLMModelOut:
    model = await _load(db, model_id)
    for field, value in body.model_dump(exclude_unset=True).items():
        setattr(model, field, value)
    db.add(AuditEvent(user_id=user.id, action="update_model", resource="llm_model",
                      resource_id=str(model_id)))
    await db.commit()
    return await _to_out(db, await _load(db, model_id))


@router.post("/{model_id}/pricing", response_model=PricingOut,
             status_code=status.HTTP_201_CREATED)
async def add_pricing(
    model_id: int, body: PricingIn,
    user: User = Depends(require_admin), db: AsyncSession = Depends(get_db),
) -> ModelPricing:
    """Add a new price version. Older versions are kept so past costs stay correct."""
    await _load(db, model_id)
    row = ModelPricing(
        model_id=model_id, input_per_1m=body.input_per_1m, output_per_1m=body.output_per_1m,
        effective_from=body.effective_from or datetime.now(UTC),
    )
    db.add(row)
    db.add(AuditEvent(user_id=user.id, action="add_pricing", resource="llm_model",
                      resource_id=str(model_id),
                      details={"input_per_1m": str(body.input_per_1m),
                               "output_per_1m": str(body.output_per_1m)}))
    await db.commit()
    return row


@router.get("/{model_id}/pricing", response_model=list[PricingOut])
async def pricing_history(
    model_id: int, _: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[ModelPricing]:
    return (await _load(db, model_id)).pricing
