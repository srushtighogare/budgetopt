"""Semantic cache: reuse an earlier answer when a new prompt *means* the same thing.

The exact cache (services/cache.py) only catches byte-identical requests. Here the
last user message is embedded locally (BAAI/bge-small-en-v1.5 via fastembed, 384-d,
~2 ms on CPU) and the nearest stored prompt for the same team is found with a
pgvector HNSW cosine index.

A wrong cached answer is worse than a cache miss, so a hit requires ALL of:
  - two-stage matching. Embeddings alone place opposites close together ("is coffee
    good / bad for you" score 0.947, see ml/eval/semantic_cache_report.md), so:
      similarity >= ACCEPT (0.97)        serve from cache
      VERIFY (0.85) <= similarity < 0.97 ask a cheap verifier model "would one answer
                                         correctly answer both?"; serve only on YES
      below VERIFY                       miss
    The verifier costs a fraction of a cent and is charged to the request; any
    verifier error counts as NO.
  - the same context: system prompt, earlier turns, route and generation settings
  - the same numbers in the prompt ("price for 3 users" must never match "for 30")
  - a task type where a near-identical question has the same answer (by default
    simple_qa and analysis only: never coding, math, translation, extraction or
    summarization, where small input changes change the output)
  - a team whose privacy tier is not confidential
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import cache
from pathlib import Path

import numpy as np
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.security import decrypt_secret, encrypt_secret
from app.db.models import PrivacyTier, SemanticCacheEntry, TaskType, Team
from app.services.providers.base import ChatMessage

log = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "fastembed"
_NUM = re.compile(r"\d+(?:[.,]\d+)?")


@dataclass(frozen=True)
class SemanticHit:
    entry_id: int
    similarity: float
    response_text: str
    model_id: int
    model_name: str
    original_cost: Decimal
    original_request_id: int
    verified_by: str | None = None  # verifier model, if the grey-zone check was used


@dataclass
class LookupResult:
    hit: SemanticHit | None
    verifier_cost: Decimal = Decimal(0)
    note: str | None = None  # why a close candidate was rejected


VERIFIER_SYSTEM = """You check whether two user requests to an AI assistant can be fully and
correctly answered by exactly the same response. Answer NO if they differ in any way that
changes the right answer: a different person, place, product or thing; different numbers
or dates; opposite or different direction (good/bad, advantages/disadvantages, first/
second, largest/smallest); a different audience, length, format or level of detail; or a
related but different question. Answer YES only if one answer serves both perfectly.
Reply with exactly one word: YES or NO."""


@cache
def _model():
    from fastembed import TextEmbedding  # heavy import, only when the cache is used

    return TextEmbedding(MODEL_NAME, cache_dir=str(CACHE_DIR))


def embed(text: str) -> list[float]:
    v = np.asarray(next(iter(_model().embed([text[:4000]]))), dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def warm_up() -> None:
    if get_settings().semantic_cache_enabled:
        try:
            embed("warm up")
        except Exception:  # model files missing/unavailable: cache simply stays off
            log.exception("semantic cache model failed to load; semantic cache disabled")


def numbers_in(text: str) -> str:
    return " ".join(sorted(n.replace(",", "") for n in _NUM.findall(text)))


def context_key(team_id: int, route: str, messages: list[ChatMessage], max_tokens: int,
                temperature: float | None) -> str:
    """Everything that must match exactly: all but the last user message, plus settings."""
    last_user = max(i for i, m in enumerate(messages) if m.role == "user")
    context = [[m.role, m.content] for i, m in enumerate(messages) if i != last_user]
    material = json.dumps({"team": team_id, "route": route, "max_tokens": max_tokens,
                           "temperature": temperature, "context": context}, sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def eligible(team: Team, task: TaskType) -> bool:
    s = get_settings()
    return (s.semantic_cache_enabled and team.privacy_tier != PrivacyTier.CONFIDENTIAL
            and task.value in s.semantic_cache_tasks)


async def verify(a: str, b: str, verifier) -> tuple[bool, Decimal]:
    """`verifier(messages) -> (text, cost)`; any failure is treated as NO."""
    try:
        text, cost = await verifier([ChatMessage("system", VERIFIER_SYSTEM),
                                     ChatMessage("user", f"Request A: {a}\n\nRequest B: {b}")])
    except Exception as e:
        log.warning("semantic cache verifier failed: %s", e)
        return False, Decimal(0)
    return text.strip().upper().startswith("YES"), cost


async def lookup(db: AsyncSession, team: Team, ctx_key: str, embedding: list[float],
                 numbers: str, prompt: str = "", verifier=None,
                 now: datetime | None = None) -> LookupResult:
    now = now or datetime.now(UTC)
    s = get_settings()
    distance = SemanticCacheEntry.embedding.cosine_distance(embedding)
    row = (await db.execute(
        select(SemanticCacheEntry, distance.label("d"))
        .where(SemanticCacheEntry.team_id == team.id,
               SemanticCacheEntry.context_key == ctx_key,
               SemanticCacheEntry.expires_at > now)
        .order_by(distance).limit(1)
    )).first()
    if row is None:
        return LookupResult(None)
    entry, d = row
    similarity = round(1.0 - float(d), 4)
    if similarity < s.semantic_cache_verify:
        return LookupResult(None)
    if entry.numbers != numbers:
        return LookupResult(None, note=f"closest cached prompt ({similarity:.3f}) has "
                                       f"different numbers")
    verified_by, cost = None, Decimal(0)
    if similarity < s.semantic_cache_accept:
        if verifier is None:
            return LookupResult(None)
        ok, cost = await verify(prompt, decrypt_secret(entry.prompt_encrypted), verifier)
        if not ok:
            return LookupResult(None, cost, note=f"closest cached prompt ({similarity:.3f}) "
                                                 f"rejected by the verifier")
        verified_by = s.semantic_cache_verifier_model
    await db.execute(update(SemanticCacheEntry).where(SemanticCacheEntry.id == entry.id)
                     .values(hits=SemanticCacheEntry.hits + 1))
    return LookupResult(SemanticHit(entry.id, similarity, entry.response_text, entry.model_id,
                                    entry.model_name, entry.original_cost,
                                    entry.original_request_id, verified_by), cost)


async def store(db: AsyncSession, team: Team, ctx_key: str, task: TaskType,
                embedding: list[float], numbers: str, *, prompt: str, response_text: str,
                model_id: int, model_name: str, cost: Decimal, request_id: int) -> None:
    db.add(SemanticCacheEntry(
        team_id=team.id, context_key=ctx_key, task_type=task, embedding=embedding,
        numbers=numbers[:500], prompt_encrypted=encrypt_secret(prompt[:4000]),
        response_text=response_text, model_id=model_id,
        model_name=model_name, original_cost=cost, original_request_id=request_id,
        expires_at=datetime.now(UTC) + timedelta(seconds=get_settings().cache_ttl_s)))
    await db.commit()


async def purge_expired(db: AsyncSession, now: datetime | None = None) -> int:
    """Delete expired entries (and with them the encrypted prompts)."""
    res = await db.execute(delete(SemanticCacheEntry).where(
        SemanticCacheEntry.expires_at <= (now or datetime.now(UTC))))
    await db.commit()
    return res.rowcount or 0
