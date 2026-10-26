"""OpenAI Chat Completions compatible request/response shapes (the subset we support)."""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ContentPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatMessageIn(BaseModel):
    role: Literal["system", "user", "assistant", "developer"]
    content: str | list[ContentPart]

    def text(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(p.text for p in self.content)


class ChatCompletionIn(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerate unsupported OpenAI params

    model: str = "auto"
    messages: list[ChatMessageIn] = Field(min_length=1, max_length=200)
    max_tokens: int | None = Field(default=None, gt=0, le=65_536)
    max_completion_tokens: int | None = Field(default=None, gt=0, le=65_536)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def needs_user_message(cls, v: list[ChatMessageIn]) -> list[ChatMessageIn]:
        if not any(m.role == "user" for m in v):
            raise ValueError("at least one user message is required")
        return v

    @property
    def output_limit(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class PlaygroundIn(ChatCompletionIn):
    team_id: int | None = None  # admins may route on behalf of any team
    use_cache: bool = True


class RouteResultOut(BaseModel):
    request_id: int
    status: str
    text: str | None
    model: str | None
    input_tokens: int
    output_tokens: int
    cost: Decimal
    baseline_cost: Decimal | None
    savings: Decimal | None
    latency_ms: float | None
    budget_state: str
    explanation: str
    attempts: list[dict]
    workload: dict | None = None


class RequestLogOut(BaseModel):
    id: int
    team_id: int
    user_id: int | None
    api_key_id: int | None
    status: str
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    cost: Decimal
    latency_ms: float | None
    created_at: datetime
    explanation: str | None
    savings: Decimal | None


class RequestDetailOut(RequestLogOut):
    prompt_text: str | None
    est_input_tokens: int | None
    est_output_tokens: int | None
    budget_state: str | None
    baseline_cost: Decimal | None
    fallback_used: bool | None
    candidates: dict | None
