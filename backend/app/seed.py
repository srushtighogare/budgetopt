"""Idempotent seed: organization, teams, admin user, providers, models and pricing.

Run with:  python -m app.seed
Prices are USD per 1M tokens; verify them against the providers' pricing pages
before running benchmarks, and add new versions via POST /models/{id}/pricing.
"""

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import encrypt_secret, hash_password
from app.db.models import (
    LLMModel,
    ModelPricing,
    Organization,
    PrivacyTier,
    Provider,
    Role,
    Team,
    User,
)
from app.db.session import SessionLocal

ORG_NAME = "Demo Organization"

TEAMS = [
    # name, monthly budget (USD), alert threshold
    ("Engineering", Decimal("5.00"), 0.8),
    ("Marketing", Decimal("0.50"), 0.8),
    ("Support", Decimal("2.00"), 0.75),
]

# provider, name, display name, context window, max output, input $/1M, output $/1M
MODELS = [
    ("groq", "openai/gpt-oss-20b", "GPT-OSS 20B (Groq)", 131_072, 32_768, "0.075", "0.30"),
    ("groq", "openai/gpt-oss-120b", "GPT-OSS 120B (Groq)", 131_072, 32_768, "0.15", "0.60"),
    ("google", "gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", 1_048_576, 65_536, "0.25", "1.50"),
    ("google", "gemini-3.5-flash", "Gemini 3.5 Flash", 1_048_576, 65_536, "1.50", "9.00"),
]

PROVIDER_KEYS = {
    "groq": "groq_api_key",
    "google": "google_api_key",
    "openai": "openai_api_key",
    "anthropic": "anthropic_api_key",
}


async def _get_or_create(db: AsyncSession, model, defaults=None, **filters):
    obj = await db.scalar(select(model).filter_by(**filters))
    if obj is None:
        obj = model(**filters, **(defaults or {}))
        db.add(obj)
        await db.flush()
    return obj


async def seed(db: AsyncSession) -> None:
    settings = get_settings()

    org = await _get_or_create(db, Organization, name=ORG_NAME)
    for name, budget, threshold in TEAMS:
        await _get_or_create(
            db, Team, organization_id=org.id, name=name,
            defaults={"monthly_budget": budget, "alert_threshold": threshold},
        )

    await _get_or_create(
        db, User, email=settings.seed_admin_email.lower(),
        defaults={
            "organization_id": org.id, "full_name": "Org Admin", "role": Role.ORG_ADMIN,
            "password_hash": hash_password(settings.seed_admin_password),
        },
    )

    providers: dict[str, Provider] = {}
    for pname, setting in PROVIDER_KEYS.items():
        provider = await _get_or_create(db, Provider, name=pname)
        secret = getattr(settings, setting)
        if secret and settings.fernet_key and not provider.encrypted_api_key:
            provider.encrypted_api_key = encrypt_secret(secret)
        providers[pname] = provider

    now = datetime.now(UTC)
    for pname, name, display, ctx, max_out, p_in, p_out in MODELS:
        model = await _get_or_create(
            db, LLMModel, provider_id=providers[pname].id, name=name,
            defaults={"display_name": display, "context_window": ctx, "max_output_tokens": max_out,
                      "max_privacy_tier": PrivacyTier.PUBLIC},
        )
        has_price = await db.scalar(
            select(ModelPricing.id).where(ModelPricing.model_id == model.id)
        )
        if not has_price:
            db.add(ModelPricing(model_id=model.id, input_per_1m=Decimal(p_in),
                                output_per_1m=Decimal(p_out), effective_from=now))

    await db.commit()


async def main() -> None:
    async with SessionLocal() as db:
        await seed(db)
    print("Seed complete.")


if __name__ == "__main__":
    asyncio.run(main())
