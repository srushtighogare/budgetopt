"""Objective answer checks and judge-output parsing shared by profiling and labelling."""

import json
import re


def last_boxed(s: str) -> str | None:
    """Content of the last \\boxed{...} in a LaTeX solution (handles nested braces)."""
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j + 1:k]
    return None


def norm_answer(a: str) -> str:
    a = a.replace(",", "").replace("$", "").replace("\\!", "").replace(" ", "")
    a = re.sub(r"\\text\{[^}]*\}", "", a)
    return a.rstrip(".").lower()


def contains_answer(text: str, gold: str) -> bool:
    """True if the answer's final result matches the gold answer.

    Uses the last \\boxed{} if present, else the last number in the text.
    """
    g = norm_answer(gold)
    boxed = last_boxed(text)
    if boxed is not None:
        return norm_answer(boxed) == g
    nums = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return bool(nums) and norm_answer(nums[-1]) == g


def parse_json(text: str):
    """First JSON object/array in an LLM reply (tolerates prose or code fences)."""
    text = text.strip()
    m = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    return json.loads(m.group(1) if m else text)
