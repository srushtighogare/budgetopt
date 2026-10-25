"""PII redaction applied before a prompt is ever written to the database.

Only used when a team has opted in to storing prompt text; otherwise only a
SHA-256 hash is kept. Patterns cover common identifiers in an Indian enterprise
context (Aadhaar, PAN) plus emails, phone numbers, card numbers and API keys.
"""

import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("API_KEY", re.compile(r"\b(?:sk|gsk|bo|AIza|sk-ant)[-_A-Za-z0-9]{16,}\b")),
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    ("AADHAAR", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b")),
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d{1,3}[ -]?)?\d{5}[ -]?\d{5}\b")),
]


def redact(text: str) -> str:
    for label, pattern in _PATTERNS:
        text = pattern.sub(f"[{label}]", text)
    return text
