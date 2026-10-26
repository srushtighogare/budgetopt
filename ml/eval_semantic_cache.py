"""Choose and justify the semantic cache's similarity threshold.

ml/datasets/semantic_cache_pairs.jsonl holds 100 curated prompt pairs:
  50 "same"      rewordings of one question (a cache hit is correct)
  50 "different" near-misses that look alike but need a different answer: another
                 entity, opposite direction, other audience/length, related question
                 (a cache hit would return a WRONG answer)
For each threshold, precision = share of hits that were correct, recall = share of
"same" pairs served from cache. A wrong answer is far worse than a missed saving.

Part 2 evaluates the deployed two-stage design: accept at >= ACCEPT, ask the
verifier model for pairs in [VERIFY, ACCEPT), reject below.

Usage: python ml/eval_semantic_cache.py   -> ml/eval/semantic_cache_report.md
"""

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))

import httpx  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.providers.adapters import ADAPTERS  # noqa: E402
from app.services.providers.base import ProviderError, RateLimited  # noqa: E402
from app.services.semantic_cache import embed, verify  # noqa: E402

VERIFIERS = {"gemini-3.5-flash-lite": "google", "openai/gpt-oss-20b": "groq"}


async def verifier_decisions(pairs: list[dict], idx: list[int], model: str) -> dict[int, bool]:
    s = get_settings()
    key = s.google_api_key if VERIFIERS[model] == "google" else s.groq_api_key
    out: dict[int, bool] = {}
    async with httpx.AsyncClient(timeout=60) as client:
        adapter = ADAPTERS[VERIFIERS[model]](key, client)

        async def call(messages):
            for attempt in range(5):
                try:
                    c = await adapter.generate(model, messages, max_tokens=512, timeout=60)
                    return c.text, 0
                except RateLimited:
                    await asyncio.sleep(15 * (attempt + 1))
            raise ProviderError(model, "rate limited")

        for i in idx:
            ok, _ = await verify(pairs[i]["a"], pairs[i]["b"], call)
            out[i] = ok
    return out


def main() -> None:
    with open(ROOT / "datasets" / "semantic_cache_pairs.jsonl") as f:
        pairs = [json.loads(line) for line in f]
    sims = np.array([float(np.dot(embed(p["a"]), embed(p["b"]))) for p in pairs])
    same = np.array([p["same"] for p in pairs])

    rows, chosen = [], None
    for t in np.round(np.arange(0.80, 0.991, 0.01), 2):
        hit = sims >= t
        tp, fp = int((hit & same).sum()), int((hit & ~same).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / same.sum()
        rows.append((t, tp, fp, precision, recall))
        if chosen is None and fp == 0:
            chosen = (t, recall)

    worst_near_miss = max((sims[i], pairs[i]) for i in np.nonzero(~same)[0])
    weakest_para = min((sims[i], pairs[i]) for i in np.nonzero(same)[0])
    table = "\n".join(f"| {t:.2f} | {tp} | {fp} | {p:.3f} | {r:.2f} |"
                      for t, tp, fp, p, r in rows)
    report = f"""# Semantic cache — threshold selection

100 curated pairs (50 rewordings, 50 near-misses), embedded with BAAI/bge-small-en-v1.5.

- Similarity of rewordings: median {np.median(sims[same]):.3f}, min {sims[same].min():.3f}
- Similarity of near-misses: median {np.median(sims[~same]):.3f}, max {sims[~same].max():.3f}
- Closest near-miss ({worst_near_miss[0]:.3f}): "{worst_near_miss[1]['a']}" vs
  "{worst_near_miss[1]['b']}"
- Least similar rewording ({weakest_para[0]:.3f}): "{weakest_para[1]['a']}" vs
  "{weakest_para[1]['b']}"

| Threshold | Correct hits | Wrong hits | Precision | Recall |
|---|---|---|---|---|
{table}

**Lowest threshold with no wrong hits: {chosen[0]:.2f}** (recall {chosen[1]:.0%}).
A threshold alone forces a bad trade-off: embeddings put opposites ("good"/"bad") almost as
close as true rewordings, so a safe threshold misses most rewordings. Hence the two-stage
design below. The numbers guard and the task-type allow-list add further protection.
"""
    s = get_settings()
    accept, low = s.semantic_cache_accept, s.semantic_cache_verify
    grey = [i for i, x in enumerate(sims) if low <= x < accept]
    model = s.semantic_cache_verifier_model
    decisions = asyncio.run(verifier_decisions(pairs, grey, model))
    hit = np.array([sims[i] >= accept or decisions.get(i, False) for i in range(len(pairs))])
    tp, fp = int((hit & same).sum()), int((hit & ~same).sum())
    g_same = [i for i in grey if same[i]]
    g_diff = [i for i in grey if not same[i]]
    wrong = [pairs[i] for i in range(len(pairs)) if hit[i] and not same[i]]
    report += f"""
## Two-stage design (deployed)

Accept at similarity >= {accept}; ask the verifier (`{model}`) when {low} <= similarity <
{accept}; reject below {low}.

- Pairs needing the verifier: {len(grey)} of {len(pairs)}
- Verifier said YES to {sum(decisions[i] for i in g_same)}/{len(g_same)} grey-zone rewordings
  and to {sum(decisions[i] for i in g_diff)}/{len(g_diff)} grey-zone near-misses
- **End-to-end: {tp} correct hits, {fp} wrong hits; precision {tp / max(1, tp + fp):.3f},
  recall {tp / same.sum():.0%}** (threshold-only at 0.95: precision 1.000, recall
  {rows[[r[0] for r in rows].index(0.95)][4]:.0%})
- Wrong hits: {[(w['a'], w['b']) for w in wrong] or 'none'}
"""
    (ROOT / "eval").mkdir(exist_ok=True)
    (ROOT / "eval" / "semantic_cache_report.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
