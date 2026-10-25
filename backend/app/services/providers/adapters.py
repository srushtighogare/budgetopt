"""Concrete adapters. Each one only knows its provider's wire format."""

from app.services.providers.base import (
    Completion,
    InvalidRequest,
    ProviderAdapter,
)


class OpenAICompatibleAdapter(ProviderAdapter):
    """OpenAI Chat Completions format, also served by Groq."""

    name = "openai"
    base_url = "https://api.openai.com/v1"

    def build_request(self, model, messages, max_tokens, temperature):
        payload: dict = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_completion_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return f"{self.base_url}/chat/completions", headers, payload

    def parse_response(self, data, latency_ms):
        choice = data["choices"][0]
        usage = data["usage"]
        return Completion(
            text=choice["message"].get("content") or "",
            input_tokens=int(usage["prompt_tokens"]),
            # completion_tokens already includes reasoning tokens for reasoning models
            output_tokens=int(usage["completion_tokens"]),
            latency_ms=latency_ms,
            finish_reason=choice.get("finish_reason"),
        )


class GroqAdapter(OpenAICompatibleAdapter):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


class GeminiAdapter(ProviderAdapter):
    """Google Gemini `generateContent` REST API."""

    name = "google"
    base_url = "https://generativelanguage.googleapis.com/v1beta"

    def build_request(self, model, messages, max_tokens, temperature):
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        contents = [
            {"role": "model" if m.role == "assistant" else "user", "parts": [{"text": m.content}]}
            for m in messages
            if m.role != "system"
        ]
        config: dict = {"maxOutputTokens": max_tokens}
        if temperature is not None:
            config["temperature"] = temperature
        payload: dict = {"contents": contents, "generationConfig": config}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        headers = {"x-goog-api-key": self.api_key or ""}
        return f"{self.base_url}/models/{model}:generateContent", headers, payload

    def parse_response(self, data, latency_ms):
        candidates = data.get("candidates") or []
        if not candidates:
            reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
            raise InvalidRequest(self.name, f"prompt blocked: {reason}")
        cand = candidates[0]
        parts = cand.get("content", {}).get("parts", [])
        usage = data["usageMetadata"]
        return Completion(
            text="".join(p.get("text", "") for p in parts if not p.get("thought")),
            input_tokens=int(usage["promptTokenCount"]),
            # "thinking" tokens are billed at the output rate
            output_tokens=int(usage.get("candidatesTokenCount", 0))
            + int(usage.get("thoughtsTokenCount", 0)),
            latency_ms=latency_ms,
            finish_reason=cand.get("finishReason"),
        )


class AnthropicAdapter(ProviderAdapter):
    """Anthropic Messages API."""

    name = "anthropic"
    base_url = "https://api.anthropic.com/v1"
    api_version = "2023-06-01"

    def build_request(self, model, messages, max_tokens, temperature):
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": m.role, "content": m.content} for m in messages if m.role != "system"
            ],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        headers = {"x-api-key": self.api_key or "", "anthropic-version": self.api_version}
        return f"{self.base_url}/messages", headers, payload

    def parse_response(self, data, latency_ms):
        usage = data["usage"]
        return Completion(
            text="".join(b.get("text", "") for b in data["content"] if b.get("type") == "text"),
            input_tokens=int(usage["input_tokens"]),
            output_tokens=int(usage["output_tokens"]),
            latency_ms=latency_ms,
            finish_reason=data.get("stop_reason"),
        )


ADAPTERS: dict[str, type[ProviderAdapter]] = {
    a.name: a for a in (GroqAdapter, OpenAICompatibleAdapter, GeminiAdapter, AnthropicAdapter)
}

