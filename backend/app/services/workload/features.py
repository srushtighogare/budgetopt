"""Hand-crafted prompt features shared by the complexity model and token estimator.

Kept deliberately small and interpretable so every number in a routing
explanation can be traced back to something visible in the prompt.
"""

import re

import numpy as np

from app.services.tokens import count_text_tokens

_CODE = re.compile(r"```|\bdef |\bclass |\bfunction\b|\breturn\b|[{};]\s*$|\bSELECT\b|=>|::",
                   re.MULTILINE)
_MATH = re.compile(r"\\frac|\\sqrt|\$[^$]+\$|\d+\s*[-+*/^=]\s*\d+|\bequation\b|\bprove\b",
                   re.IGNORECASE)
_CONSTRAINT = re.compile(r"\b(must|exactly|at least|at most|no more than|without|only|"
                         r"step[- ]by[- ]step|in \d+ words|format|bullet|json|table)\b",
                         re.IGNORECASE)
_ENUM = re.compile(r"^\s*(\d+[.)]|[-*•])\s+", re.MULTILINE)
_SENT = re.compile(r"[.!?](\s|$)")
_NUM = re.compile(r"\d+(?:\.\d+)?")

FEATURE_NAMES = [
    "log_tokens", "log_chars", "n_sentences", "n_questions", "n_list_items",
    "n_constraints", "code_signals", "math_signals", "n_numbers", "has_context",
    "log_context_tokens", "instruction_tokens",
]


def prompt_features(text: str) -> np.ndarray:
    tokens = count_text_tokens(text)
    # Dolly/Alpaca style: instruction, blank line, then supporting context/passage
    instruction, _, context = text.partition("\n\n")
    ctx_tokens = count_text_tokens(context) if context else 0
    return np.array([
        np.log1p(tokens),
        np.log1p(len(text)),
        len(_SENT.findall(text)),
        text.count("?"),
        len(_ENUM.findall(text)),
        len(_CONSTRAINT.findall(text)),
        len(_CODE.findall(text)),
        len(_MATH.findall(text)),
        len(_NUM.findall(text)),
        float(bool(context.strip())),
        np.log1p(ctx_tokens),
        count_text_tokens(instruction),
    ], dtype=np.float64)
