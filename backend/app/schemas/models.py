from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import PrivacyTier, TaskType


class PricingIn(BaseModel):
    input_per_1m: Decimal = Field(ge=0, max_digits=18, decimal_places=8)
    output_per_1m: Decimal = Field(ge=0, max_digits=18, decimal_places=8)
    effective_from: datetime | None = None  # defaults to now


class PricingOut(PricingIn):
    model_config = ConfigDict(from_attributes=True)

    id: int
    effective_from: datetime


class PerformanceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    task_type: TaskType
    quality_score: float | None
    quality_samples: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None


class LLMModelCreate(BaseModel):
    provider: str
    name: str = Field(min_length=1, max_length=100)
    display_name: str = Field(min_length=1, max_length=100)
    context_window: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    max_privacy_tier: PrivacyTier = PrivacyTier.PUBLIC
    pricing: PricingIn


class LLMModelUpdate(BaseModel):
    display_name: str | None = None
    context_window: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_privacy_tier: PrivacyTier | None = None
    is_active: bool | None = None


class LLMModelOut(BaseModel):
    id: int
    provider: str
    name: str
    display_name: str
    context_window: int
    max_output_tokens: int
    max_privacy_tier: PrivacyTier
    is_active: bool
    current_pricing: PricingOut | None
    performance: list[PerformanceOut] = []
