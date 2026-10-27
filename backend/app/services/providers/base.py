"""Provider adapter interface.

Every LLM provider is wrapped in an adapter that speaks one common shape:
chat messages in, `Completion` (text + exact billed token counts + latency) out.
Provider-specific HTTP errors are mapped onto a small error hierarchy so the
fallback executor can decide whether trying another model makes sense.
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import httpx

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int  # everything billed as output, including hidden reasoning tokens
    latency_ms: float
    finish_reason: str | None = None


# --- Errors ---

class ProviderError(Exception):
    """Base class. `retryable` = a different attempt/model may well succeed."""

    retryable = True

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider
        self.status_code = status_code


class ProviderTimeout(ProviderError):
    pass


class RateLimited(ProviderError):
    """429 / quota exhausted. `retry_after_s` is the provider's suggested wait, if given."""

    def __init__(self, provider: str, message: str, status_code: int | None = None,
                 retry_after_s: float | None = None):
        super().__init__(provider, message, status_code)
        self.retry_after_s = retry_after_s


class ProviderUnavailable(ProviderError):
    """5xx or network failure."""


class AuthenticationFailed(ProviderError):
    """Bad or missing API key. Retrying this provider is pointless, another may work."""


class InvalidRequest(ProviderError):
    """4xx caused by the request itself (e.g. context too long, unknown model)."""


class MissingCredentials(AuthenticationFailed):
    pass


def _retry_after(headers, body: str) -> float | None:
    """Seconds to wait, from a Retry-After header or Gemini's RetryInfo in the body."""
    value = (headers or {}).get("retry-after")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', body)
    return float(m.group(1)) if m else None


def error_from_status(provider: str, status: int, body: str, headers=None) -> ProviderError:
    snippet = body[:300]
    if status == 429:
        return RateLimited(provider, f"rate limited: {snippet}", status,
                           retry_after_s=_retry_after(headers, body))
    if status in (401, 403):
        return AuthenticationFailed(provider, f"authentication failed: {snippet}", status)
    if status >= 500:
        return ProviderUnavailable(provider, f"server error {status}: {snippet}", status)
    return InvalidRequest(provider, f"request rejected {status}: {snippet}", status)


# --- Adapter base ---

class ProviderAdapter(ABC):
    name: str

    def __init__(self, api_key: str | None, client: httpx.AsyncClient | None = None,
                 base_url: str | None = None):
        self.api_key = api_key
        self._client = client
        if base_url:
            self.base_url = base_url

    async def generate(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 1024,
        temperature: float | None = None,
        timeout: float = 60.0,
    ) -> Completion:
        if not self.api_key:
            raise MissingCredentials(self.name, "no API key configured")
        if not messages:
            raise InvalidRequest(self.name, "messages must not be empty")

        url, headers, payload = self.build_request(model, messages, max_tokens, temperature)
        start = time.perf_counter()
        try:
            client = self._client or _shared_client()
            resp = await client.post(url, headers=headers, json=payload, timeout=timeout)
        except httpx.TimeoutException as e:
            raise ProviderTimeout(self.name, f"timed out after {timeout}s") from e
        except httpx.HTTPError as e:
            raise ProviderUnavailable(self.name, f"network error: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            raise error_from_status(self.name, resp.status_code, resp.text, resp.headers)
        try:
            return self.parse_response(resp.json(), latency_ms)
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise ProviderUnavailable(self.name, f"unexpected response shape: {e}") from e

    @abstractmethod
    def build_request(
        self, model: str, messages: list[ChatMessage], max_tokens: int,
        temperature: float | None,
    ) -> tuple[str, dict[str, str], dict]:
        """Return (url, headers, json_payload)."""

    @abstractmethod
    def parse_response(self, data: dict, latency_ms: float) -> Completion: ...


_client: httpx.AsyncClient | None = None


def _shared_client() -> httpx.AsyncClient:
    """One pooled HTTP client for all adapters (connection reuse lowers latency)."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(limits=httpx.Limits(max_connections=100))
    return _client
