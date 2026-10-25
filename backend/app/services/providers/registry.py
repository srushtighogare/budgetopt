"""Builds adapters for stored providers, decrypting their API keys on demand."""

from collections.abc import Callable

from app.core.security import decrypt_secret
from app.db.models import Provider
from app.services.providers.adapters import ADAPTERS
from app.services.providers.base import ProviderAdapter

AdapterFactory = Callable[[Provider], ProviderAdapter]


def build_adapter(provider: Provider) -> ProviderAdapter:
    cls = ADAPTERS.get(provider.name)
    if cls is None:
        raise LookupError(f"no adapter implemented for provider '{provider.name}'")
    key = decrypt_secret(provider.encrypted_api_key) if provider.encrypted_api_key else None
    return cls(api_key=key)


def supported_providers() -> list[str]:
    return sorted(ADAPTERS)
