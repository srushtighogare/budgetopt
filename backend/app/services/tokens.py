"""Input-token estimation before a call is made.

Uses the o200k_base BPE vocabulary (GPT-4o / gpt-oss). Other providers use
different tokenizers, so for them this is an estimate; it is only used for
pre-call decisions (context-window checks, cost estimates). Billing always uses
the provider-reported counts.
"""

from functools import lru_cache

import tiktoken

from app.services.providers.base import ChatMessage

# Chat formats add a few framing tokens per message (role markers etc.)
PER_MESSAGE_OVERHEAD = 4
PER_REQUEST_OVERHEAD = 3


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_text_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


def estimate_input_tokens(messages: list[ChatMessage]) -> int:
    return PER_REQUEST_OVERHEAD + sum(
        PER_MESSAGE_OVERHEAD + count_text_tokens(m.content) for m in messages
    )
