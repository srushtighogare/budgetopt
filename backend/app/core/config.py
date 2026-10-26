from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, read from environment variables (and `.env`)."""

    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    app_name: str = "BudgetOpt"
    environment: str = "development"

    database_url: str = "postgresql+asyncpg://budgetopt:budgetopt@localhost:5432/budgetopt"
    redis_url: str = "redis://localhost:6379/0"

    # Security
    jwt_secret: str = Field(default="dev-only-secret-change-me-please", min_length=16)
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_days: int = 7
    # Fernet key used to encrypt provider API keys at rest. Generate with:
    # python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    fernet_key: str = ""
    cors_origins: list[str] = ["http://localhost:5173"]

    # Gateway
    max_prompt_chars: int = 32_000
    default_max_tokens: int = 1024
    rate_limit_per_minute: int = 60
    cache_ttl_s: int = 86_400
    provider_timeout_s: float = 60.0

    # Background monitoring (forecasts, anomaly scans) inside the API process
    enable_scheduler: bool = True
    forecast_interval_s: int = 3600
    anomaly_interval_s: int = 900

    # Provider credentials (copied, encrypted, into `providers` by the seed script)
    groq_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Initial org admin created by the seed script
    seed_admin_email: str = "admin@example.com"
    seed_admin_password: str = "ChangeMe!2026"


@lru_cache
def get_settings() -> Settings:
    return Settings()
