from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from app.db.models import PrivacyTier, Role

# --- Auth ---


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshIn(BaseModel):
    refresh_token: str


# --- Users ---


class UserCreate(BaseModel):
    email: EmailStr
    full_name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=10, max_length=128)
    role: Role = Role.MEMBER
    team_id: int | None = None

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if not (any(c.isdigit() for c in v) and any(c.isalpha() for c in v)):
            raise ValueError("password must contain letters and digits")
        return v


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str
    role: Role
    team_id: int | None
    is_active: bool


# --- Teams ---


class TeamCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    monthly_budget: Decimal = Field(gt=0, max_digits=18, decimal_places=8)
    alert_threshold: float = Field(default=0.8, gt=0, lt=1)
    privacy_tier: PrivacyTier = PrivacyTier.PUBLIC
    store_prompts: bool = False


class TeamUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    monthly_budget: Decimal | None = Field(default=None, gt=0, max_digits=18, decimal_places=8)
    alert_threshold: float | None = Field(default=None, gt=0, lt=1)
    privacy_tier: PrivacyTier | None = None
    store_prompts: bool | None = None


class TeamOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    monthly_budget: Decimal
    alert_threshold: float
    privacy_tier: PrivacyTier
    store_prompts: bool
    policy_id: int | None
    created_at: datetime


class TeamSpendOut(BaseModel):
    team_id: int
    period_start: datetime
    monthly_budget: Decimal
    spend: Decimal
    usage_ratio: float
    budget_state: str


# --- API keys ---


class ApiKeyCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class ApiKeyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    team_id: int
    name: str
    prefix: str
    is_active: bool
    created_at: datetime
    last_used_at: datetime | None


class ApiKeyCreated(ApiKeyOut):
    key: str  # plaintext, returned exactly once
