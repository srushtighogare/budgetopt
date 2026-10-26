"""Benchmark data collection: the response matrix.

For a fixed set of held-out prompts (never used to train the classifier, not in the
profiling set), every benchmark model answers every prompt once, and every answer
is scored once:
  - GSM8K/MATH prompts: 10 if the final answer matches the gold answer, else 0
  - all others: graded 1-10 by a judge model that is NOT one of the benchmark
    models (the same rubric as model profiling), so every answer is judged alike

Routing strategies are then evaluated offline by looking up, for each prompt, the
cost and score of the model a strategy would pick (benchmark_eval.py). This is the
standard methodology for evaluating LLM routers (e.g. RouterBench, RouteLLM): it
costs one answer per (prompt, model) instead of one per (prompt, strategy).

Free-tier friendly and resumable, like labeling_run.py.

Usage: backend/.venv/bin/python -I ml/benchmark_collect.py [--status]
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "backend"))

from app.core.config import get_settings  # noqa: E402
from app.services.profiling.answers import contains_answer, parse_json  # noqa: E402
from app.services.profiling.quality import JUDGE_SYSTEM  # noqa: E402
from app.services.providers.adapters import GeminiAdapter  # noqa: E402
from app.services.providers.base import ChatMessage  # noqa: E402

import labeling_run as lr  # noqa: E402
import train_classifier as tc  # noqa: E402

OUT = ROOT / "datasets" / "benchmark"
PER_TASK = 20
MAX_CHARS = 1500
SEED = 2027
ANSWER_MAX_TOKENS = 1024

MODELS = ["openai/gpt-oss-20b", "gemini-3.5-flash-lite", "openai/gpt-oss-120b",
          "gemini-3.5-flash"]
JUDGES = ["qwen/qwen3.8-27b", "gemini-3-flash-preview"]  # deliberately outside MODELS
lr.PRICES.setdefault("gemini-3.5-flash-lite", (0.10, 0.40))


def read(name: str) -> list[dict]:
    p = OUT / f"{name}.jsonl"
    with open(p) if p.exists() else open("/dev/null") as f:
        return [json.loads(line) for line in f]


def append(name: str, row: dict) -> None:
    with open(OUT / f"{name}.jsonl", "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt_set() -> list[dict]:
    """Held-out prompts: the classifier's test split minus the profiling set."""
    path = OUT / "prompts.jsonl"
    if path.exists():
        return read("prompts")
    test_texts = set(tc.splits()["test"][0])
    with open(lr.LABELS / "annotations.jsonl") as f:
        rows = [json.loads(line) for line in f]
    prof = ROOT.parent / "backend" / "app" / "services" / "profiling" / "profiling_set.jsonl"
    with open(prof) as f:
        excluded = {json.loads(line)["text"] for line in f}
    pool = [r for r in rows if r["text"] in test_texts and r["text"] not in excluded
            and len(r["text"]) <= MAX_CHARS]
    rng = random.Random(SEED)
    items = []
    for task in lr.TASKS:
        group = [r for r in pool if r["label"] == task]
        if task == "reasoning_math":
            group = [r for r in group if r["source"] in ("gsm8k", "math")]
        rng.shuffle(group)
        for r in group[:PER_TASK]:
            gold = None
            if r["source"] == "gsm8k":
                gold = lr._gold_index("gsm8k").get(r["text"])
            elif r["source"] == "math":
                for sub in ("algebra", "counting_and_probability", "geometry",
                            "intermediate_algebra", "number_theory", "prealgebra",
                            "precalculus"):
                    gold = gold or lr._gold_index(f"math:{sub}").get(r["text"])
            items.append({"bid": f"b{len(items):03d}", "task_type": task, "text": r["text"],
                          "gold": gold, "source": r["source"]})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return items


async def collect(r: lr.Runner, items: list[dict]) -> None:
    done = {(a["bid"], a["model"]) for a in read("answers")}
    # interleave models so a quota pause leaves a balanced partial matrix
    for it in items:
        for model in MODELS:
            if (it["bid"], model) in done or model in r.exhausted:
                continue
            try:
                c = await r.call(model, [ChatMessage("user", it["text"])], ANSWER_MAX_TOKENS)
            except lr.DailyQuotaExhausted:
                print(f"  daily quota exhausted for {model}; continuing with the others")
                continue
            append("answers", {"bid": it["bid"], "model": model, "text": c.text,
                               "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                               "latency_ms": round(c.latency_ms, 1),
                               "finish_reason": c.finish_reason})
        if len(r.exhausted) == len(MODELS):
            raise lr.StageBlocked("all benchmark models exhausted today")


async def judge(r: lr.Runner, items: list[dict]) -> None:
    by_id = {it["bid"]: it for it in items}
    done = {(s["bid"], s["model"]) for s in read("scores")}
    for a in read("answers"):
        key = (a["bid"], a["model"])
        if key in done:
            continue
        it = by_id[a["bid"]]
        if it["gold"]:
            correct = contains_answer(a["text"], it["gold"])
            append("scores", {"bid": a["bid"], "model": a["model"], "score": 10.0 if correct
                              else 0.0, "judge": "gold-answer", "correct": correct})
            continue
        msgs = [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", (
            f"### REQUEST\n{it['text'][:6000]}\n\n### ANSWER\n{a['text'][:8000]}"))]
        c = await r.first_available(JUDGES, msgs, 1024)
        try:
            v = parse_json(c.text)
            score = float(v["score"])
            if not 1 <= score <= 10:
                raise ValueError(score)
        except (ValueError, KeyError, TypeError):
            print(f"  unparseable judgment for {key}, retry next run")
            continue
        append("scores", {"bid": a["bid"], "model": a["model"], "score": score,
                          "judge": c.model, "reason": str(v.get("reason", ""))[:300]})


def status(items: list[dict]) -> None:
    a, s = read("answers"), read("scores")
    print(f"prompts {len(items)} | answers {len(a)}/{len(items) * len(MODELS)} | "
          f"scores {len(s)}/{len(items) * len(MODELS)}")
    for m in MODELS:
        print(f"  {m:24} answers {sum(x['model'] == m for x in a):4}  "
              f"scores {sum(x['model'] == m for x in s):4}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-usd", type=float, default=5.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = build_prompt_set()
    if args.status:
        return status(items)
    r = lr.Runner(args.max_usd)
    key = get_settings().google_api_key
    r.adapters["gemini-3.5-flash-lite"] = GeminiAdapter(key, r.client)
    try:
        for name, stage in (("answers", collect), ("judge", judge)):
            print(f"== {name}")
            try:
                await stage(r, items)
            except (lr.StageBlocked, lr.DailyQuotaExhausted) as e:
                print(f"  {name} paused: {e}")
    except lr.CostCapReached as e:
        print(f"STOPPED: {e}")
    finally:
        await r.client.aclose()
        status(items)


if __name__ == "__main__":
    asyncio.run(main())
