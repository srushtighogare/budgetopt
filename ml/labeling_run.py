"""Free-tier labelling run: produces complexity labels, real output-token counts and
an independently labelled out-of-distribution (OOD) test set for the classifier.

Stages (all resumable; results are appended to ml/datasets/labels/*.jsonl):
  ood      ~320 natural Alpaca prompts the dataset builder did NOT label, labelled
           into the 8 task types by an LLM annotator (batched) -> classifier OOD test
  answers  for 800 training prompts (100 per task type): a cheap-model answer
           (GPT-OSS 20B) and a strong reference answer (Gemini 3.5 Flash while its
           free quota lasts, then GPT-OSS 120B), with billed token counts
  judge    a judge model (different from that row's reference model) scores the
           cheap answer against the reference; GSM8K/MATH rows are also checked
           against the gold final answer

Free-tier behaviour: per-minute limits are waited out; when a model's daily
quota is exhausted it is marked exhausted for this run and the next model in
that stage's list is used; when a stage has no model left the run stops and
prints how to resume. A paid-rate cost estimate is tracked and the run aborts
if it would exceed --max-usd (safety net in case billing is enabled).

Usage (from repo root):
  backend/.venv/bin/python -I ml/labeling_run.py            # run / resume
  backend/.venv/bin/python -I ml/labeling_run.py --status   # progress only
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from functools import cache
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))

import httpx  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.services.profiling.answers import (  # noqa: E402
    contains_answer,
    last_boxed,
    parse_json,
)
from app.services.providers.adapters import GeminiAdapter, GroqAdapter  # noqa: E402
from app.services.providers.base import ChatMessage, ProviderAdapter  # noqa: E402

DATA = ROOT / "datasets"
LABELS = DATA / "labels"
SEED = 7
PER_CLASS = 100
OOD_SIZE = 320
OOD_BATCH = 16
ANSWER_MAX_TOKENS = 1024
# Gemini's billed thinking tokens share the output budget; give references headroom
REFERENCE_MAX_TOKENS = 2048
JUDGE_MAX_TOKENS = 1024

TASKS = ["translation", "summarization", "simple_qa", "extraction", "creative_writing",
         "coding", "reasoning_math", "analysis"]

# USD per 1M tokens (input, output) — same as app/seed.py; used for the safety cap
PRICES = {"openai/gpt-oss-20b": (0.075, 0.30), "openai/gpt-oss-120b": (0.15, 0.60),
          "gemini-3.5-flash": (1.50, 9.00), "gemini-3.1-flash-lite": (0.25, 1.50),
          # approximate list prices; only used for the paid-rate safety cap
          "gemini-3-flash-preview": (0.50, 3.00), "qwen/qwen3.8-27b": (0.29, 0.59)}

# Training-set annotation: a different annotator from the OOD test set's (gpt-oss-120b)
TRAIN_ANNOTATOR_ORDER = ["gemini-3.1-flash-lite", "openai/gpt-oss-120b"]
ANNOTATE_PLAN = {  # source -> number of prompts to annotate
    "alpaca_natural": 2400, "alpaca_translate": 246, "dolly": 700, "codealpaca": 350,
    "mbpp": 100, "gsm8k": 150, "math": 150,
}

CHEAP = "openai/gpt-oss-20b"
# Any strong model with free quota left can serve as the reference; each row records
# which one did. (Models are interchangeable here; the pipeline is what's evaluated.)
REFERENCE_ORDER = ["gemini-3.5-flash", "gemini-3-flash-preview", "openai/gpt-oss-120b",
                   "qwen/qwen3.8-27b"]
# Flash-Lite is the last-resort judge so free-tier judging isn't throttled to ~150/day;
# every row records its judge, and judges are validated against gold math answers.
JUDGE_ORDER = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "gemini-3-flash-preview",
               "gemini-3.1-flash-lite"]
ANNOTATOR_ORDER = ["openai/gpt-oss-120b", "gemini-3.5-flash"]


class DailyQuotaExhausted(Exception):
    pass


class StageBlocked(Exception):
    """No model left for a stage today."""


class CostCapReached(Exception):
    pass


@dataclass
class Call:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str | None
    model: str


class Runner:
    def __init__(self, max_usd: float):
        s = get_settings()
        self.client = httpx.AsyncClient(timeout=120)
        self.adapters: dict[str, ProviderAdapter] = {
            CHEAP: GroqAdapter(s.groq_api_key, self.client),
            "openai/gpt-oss-120b": GroqAdapter(s.groq_api_key, self.client),
            "gemini-3.5-flash": GeminiAdapter(s.google_api_key, self.client),
            "gemini-3.1-flash-lite": GeminiAdapter(s.google_api_key, self.client),
            "gemini-3-flash-preview": GeminiAdapter(s.google_api_key, self.client),
            "qwen/qwen3.8-27b": GroqAdapter(s.groq_api_key, self.client),
        }
        self.exhausted: set[str] = set()
        self.max_usd = max_usd
        self.spent = self._spent_so_far()

    def _spent_so_far(self) -> float:
        total = 0.0
        for name in ("answers", "judgments", "ood", "annotations"):
            for row in read(name):
                total += row.get("paid_rate_usd", 0.0)
        return total

    def _cost(self, model: str, tin: int, tout: int) -> float:
        p_in, p_out = PRICES[model]
        return (tin * p_in + tout * p_out) / 1e6

    async def call(self, model: str, messages: list[ChatMessage], max_tokens: int) -> Call:
        if model in self.exhausted:
            raise DailyQuotaExhausted(model)
        adapter = self.adapters[model]
        url, headers, payload = adapter.build_request(model, messages, max_tokens, None)
        last = "no attempt"
        for attempt in range(8):
            t0 = time.perf_counter()
            try:
                resp = await self.client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as e:
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
                last = f"network: {e}"
                continue
            latency = (time.perf_counter() - t0) * 1000
            if resp.status_code == 200:
                c = adapter.parse_response(resp.json(), latency)
                cost = self._cost(model, c.input_tokens, c.output_tokens)
                if self.spent + cost > self.max_usd:
                    raise CostCapReached(f"paid-rate estimate would exceed ${self.max_usd}")
                self.spent += cost
                return Call(c.text, c.input_tokens, c.output_tokens, latency,
                            c.finish_reason, model)
            if resp.status_code == 429:
                wait = self._retry_after(resp)
                if self._is_daily(resp, wait):
                    self.exhausted.add(model)
                    raise DailyQuotaExhausted(model)
                await asyncio.sleep(wait)
                last = "429 per-minute limit"
                continue
            if resp.status_code >= 500:
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
                last = f"{resp.status_code}"
                continue
            raise RuntimeError(f"{model} {resp.status_code}: {resp.text[:300]}")
        raise RuntimeError(f"{model} kept failing ({last})")

    @staticmethod
    def _retry_after(resp: httpx.Response) -> float:
        if ra := resp.headers.get("retry-after"):
            try:
                return float(ra) + 0.5
            except ValueError:
                pass
        m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', resp.text)
        return float(m.group(1)) + 0.5 if m else 20.0

    @staticmethod
    def _is_daily(resp: httpx.Response, wait: float) -> bool:
        body = resp.text
        if "PerDay" in body or "per day" in body.lower() or "RPD" in body or "TPD" in body:
            return True
        reset = resp.headers.get("x-ratelimit-reset-requests", "")
        remaining = resp.headers.get("x-ratelimit-remaining-requests")
        return (remaining == "0" and ("h" in reset)) or wait > 600

    async def first_available(self, order: list[str], messages, max_tokens,
                              avoid: str | None = None) -> Call:
        prefs = [m for m in order if m != avoid] + ([avoid] if avoid in order else [])
        for model in prefs:
            if model in self.exhausted:
                continue
            try:
                return await self.call(model, messages, max_tokens)
            except DailyQuotaExhausted:
                print(f"  daily quota exhausted for {model}; switching")
        raise StageBlocked(f"no model left among {order}")


# --- persistence ---

def path(name: str) -> Path:
    return LABELS / f"{name}.jsonl"


def read(name: str) -> list[dict]:
    p = path(name)
    return [json.loads(line) for line in open(p)] if p.exists() else []


def append(name: str, row: dict) -> None:
    with open(path(name), "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --- sampling ---

LIMIT: int | None = None  # --limit, for smoke tests
SHARD: tuple[int, int] = (0, 1)  # --shard k/n: this process handles ids with id % n == k


def answer_sample() -> list[dict]:
    with open(DATA / "train.jsonl") as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(SEED)
    groups = []
    for t in TASKS:
        group = [r for r in rows if r["label"] == t]
        rng.shuffle(group)
        groups.append(group[:PER_CLASS])
    # Round-robin across task types so a partial (quota-limited) run is still balanced
    out = [g[i] for i in range(PER_CLASS) for g in groups if i < len(g)]
    return out[:LIMIT] if LIMIT else out


def ood_sample() -> list[dict]:
    """Natural Alpaca prompts that build_dataset.py did not keyword-label."""
    sys.path.insert(0, str(ROOT))
    import build_dataset as bd

    used = set()
    for n in ("train", "val", "test"):
        with open(DATA / f"{n}.jsonl") as f:
            used |= {re.sub(r"\W+", " ", r["text"].lower()).strip() for r in map(json.loads, f)}
    with open(bd.RAW / "alpaca" / "alpaca.json") as f:
        alpaca = json.load(f)
    pool = []
    for x in alpaca:
        text = bd.join(x["instruction"], x["input"])
        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key not in used and len(text) < 1500:
            pool.append(text)
    rng = random.Random(SEED)
    rng.shuffle(pool)
    return [{"ood_id": i, "text": t} for i, t in enumerate(pool[:LIMIT or OOD_SIZE])]


def annotation_pool() -> list[dict]:
    """Prompts from every source, so each source contributes several task types and the
    classifier cannot learn 'source style -> label' shortcuts. Excludes the OOD set."""
    sys.path.insert(0, str(ROOT))
    import pandas as pd

    import build_dataset as bd

    def key(t: str) -> str:
        return re.sub(r"\W+", " ", t.lower()).strip()

    ood = {key(x["text"]) for x in ood_sample()}
    rng = random.Random(SEED + 1)
    pools: dict[str, list[tuple[str, str | None]]] = {k: [] for k in ANNOTATE_PLAN}
    with open(bd.RAW / "alpaca" / "alpaca.json") as f:
        for x in json.load(f):
            text = bd.join(x["instruction"], x["input"])
            if len(text) >= 1500 or key(text) in ood:
                continue
            is_tr = bool(bd.TRANSLATE.search(x["instruction"])) and not bd.CODE.search(text)
            pools["alpaca_translate" if is_tr else "alpaca_natural"].append((text, None))
    with open(bd.RAW / "dolly" / "dolly.jsonl") as f:
        for line in f:
            x = json.loads(line)
            pools["dolly"].append((bd.join(x["instruction"], x["context"]),
                                   bd.DOLLY_MAP[x["category"]]))
    with open(bd.RAW / "codealpaca" / "codealpaca.json") as f:
        pools["codealpaca"] = [(bd.join(x["instruction"], x["input"]), "coding")
                               for x in json.load(f)]
    mbpp = pd.read_parquet(bd.RAW / "mbpp" / "train.parquet")
    pools["mbpp"] = [(t, "coding") for t in mbpp["text"]]
    pools["gsm8k"] = [(q, "reasoning_math")
                      for q in pd.read_parquet(bd.RAW / "gsm8k" / "train.parquet")["question"]]
    for f in sorted((bd.RAW / "math").glob("*.parquet")):
        pools["math"] += [(p, "reasoning_math") for p in pd.read_parquet(f)["problem"]]

    out = []
    for src, n in ANNOTATE_PLAN.items():
        items = pools[src]
        rng.shuffle(items)
        for text, source_label in items[:n]:
            out.append({"ann_id": len(out), "text": text.strip()[:bd.MAX_CHARS], "source": src,
                        "source_label": source_label})
    return out[:LIMIT] if LIMIT else out


# --- prompts ---

ANNOTATE_SYSTEM = """You label user requests to an AI assistant with exactly one task type:
translation: convert text between human languages
summarization: condense/shorten given text or produce a summary/headline of it
simple_qa: short factual or general-knowledge question, or answer from a given passage
extraction: pull out, list, classify, sort or tag information from given text/items
creative_writing: stories, poems, slogans, names, brainstorming ideas, open-ended writing
coding: write, explain, fix or convert program code, SQL, regex, markup
reasoning_math: math problems, calculations, logic puzzles needing step-by-step reasoning
analysis: compare, evaluate, argue, explain causes/implications, critique, recommend
Return ONLY a JSON array of objects {"i": <index>, "label": "<task type>"}."""

JUDGE_SYSTEM = """You are a strict evaluator. Given a user request, a REFERENCE answer from a
strong model and a CANDIDATE answer, rate the CANDIDATE from 1 to 10 for correctness,
completeness and following the request's instructions. The reference is a guide, not
ground truth; if the candidate is correct in a different way, judge it on its merits.
Then decide ACCEPTABLE: would a typical user be as satisfied with the candidate as with
the reference for this request? Reply ONLY with JSON:
{"score": <1-10>, "acceptable": <true|false>, "reason": "<one short sentence>"}"""


def judge_messages(prompt: str, reference: str, candidate: str) -> list[ChatMessage]:
    return [ChatMessage("system", JUDGE_SYSTEM), ChatMessage("user", (
        f"### REQUEST\n{prompt[:6000]}\n\n### REFERENCE ANSWER\n{reference[:6000]}\n\n"
        f"### CANDIDATE ANSWER\n{candidate[:6000]}"))]


@cache
def _gold_index(source: str) -> dict[str, str]:
    import pandas as pd

    if source == "gsm8k":
        df = pd.read_parquet(DATA / "raw" / "gsm8k" / "train.parquet")
        return {q.strip(): a.split("####")[-1].strip()
                for q, a in zip(df["question"], df["answer"], strict=True)}
    df = pd.read_parquet(DATA / "raw" / "math" / f"{source.split(':')[1]}.parquet")
    return {p.strip(): last_boxed(sol)
            for p, sol in zip(df["problem"], df["solution"], strict=True)}


def gold_answer(row: dict) -> str | None:
    """Final answer for GSM8K/MATH rows, used for an objective correctness check."""
    src = row["source"]
    if src == "gsm8k" or src.startswith("math:"):
        return _gold_index(src).get(row["text"])
    return None


# --- stages ---

async def stage_ood(r: Runner) -> None:
    done = {row["ood_id"] for row in read("ood")}
    todo = [x for x in ood_sample() if x["ood_id"] not in done]
    for b in range(0, len(todo), OOD_BATCH):
        batch = todo[b:b + OOD_BATCH]
        listing = "\n\n".join(f"[{i}] {x['text'][:700]}" for i, x in enumerate(batch))
        c = await r.first_available(ANNOTATOR_ORDER, [
            ChatMessage("system", ANNOTATE_SYSTEM), ChatMessage("user", listing)], 2048)
        try:
            labels = {int(o["i"]): o["label"] for o in parse_json(c.text)}
        except (ValueError, KeyError, TypeError):
            print("  unparseable annotation batch, will retry next run")
            continue
        share = r._cost(c.model, c.input_tokens, c.output_tokens) / len(batch)
        for i, x in enumerate(batch):
            if labels.get(i) in TASKS:
                append("ood", {**x, "label": labels[i], "annotator": c.model,
                               "paid_rate_usd": share})
        print(f"  ood {len(done) + b + len(batch)}/{OOD_SIZE} via {c.model}")


async def stage_annotate(r: Runner) -> None:
    done = {row["ann_id"] for row in read("annotations")}
    pool = annotation_pool()
    todo = [x for x in pool if x["ann_id"] not in done]
    for b in range(0, len(todo), OOD_BATCH):
        batch = todo[b:b + OOD_BATCH]
        listing = "\n\n".join(f"[{i}] {x['text'][:700]}" for i, x in enumerate(batch))
        c = await r.first_available(TRAIN_ANNOTATOR_ORDER, [
            ChatMessage("system", ANNOTATE_SYSTEM), ChatMessage("user", listing)], 2048)
        try:
            labels = {int(o["i"]): o["label"] for o in parse_json(c.text)}
        except (ValueError, KeyError, TypeError):
            print("  unparseable annotation batch, will retry next run")
            continue
        share = r._cost(c.model, c.input_tokens, c.output_tokens) / len(batch)
        for i, x in enumerate(batch):
            if labels.get(i) in TASKS:
                append("annotations", {**x, "label": labels[i], "annotator": c.model,
                                       "paid_rate_usd": share})
        if (b // OOD_BATCH) % 10 == 0:
            print(f"  annotated {len(done) + b + len(batch)}/{len(pool)} via {c.model}")


async def stage_answers(r: Runner) -> None:
    done = {(row["id"], row["role"]) for row in read("answers")}
    sample = answer_sample()
    for n, row in enumerate(sample, 1):
        msgs = [ChatMessage("user", row["text"])]
        for role in ("cheap", "reference"):
            if (row["id"], role) in done:
                continue
            if role == "cheap":
                c = await r.call(CHEAP, msgs, ANSWER_MAX_TOKENS)
            else:
                c = await r.first_available(REFERENCE_ORDER, msgs, REFERENCE_MAX_TOKENS)
            append("answers", {
                "id": row["id"], "role": role, "label": row["label"], "source": row["source"],
                "math_level": row.get("math_level"), "model": c.model, "text": c.text,
                "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "latency_ms": round(c.latency_ms, 1), "finish_reason": c.finish_reason,
                "paid_rate_usd": r._cost(c.model, c.input_tokens, c.output_tokens),
            })
        if n % 10 == 0:
            print(f"  answers {n}/{len(sample)}  paid-rate est ${r.spent:.3f}")


async def stage_judge(r: Runner) -> None:
    answers: dict[int, dict[str, dict]] = {}
    for a in read("answers"):
        answers.setdefault(a["id"], {})[a["role"]] = a
    done = {j["id"] for j in read("judgments")}
    texts = {row["id"]: row for row in answer_sample()}
    k, n_shards = SHARD
    todo = [i for i, v in answers.items()
            if "cheap" in v and "reference" in v and i not in done and i % n_shards == k]
    for n, i in enumerate(todo, 1):
        cheap, ref, row = answers[i]["cheap"], answers[i]["reference"], texts[i]
        c = await r.first_available(JUDGE_ORDER, judge_messages(row["text"], ref["text"],
                                    cheap["text"]), JUDGE_MAX_TOKENS, avoid=ref["model"])
        try:
            v = parse_json(c.text)
            score, ok = int(v["score"]), bool(v["acceptable"])
        except (ValueError, KeyError, TypeError):
            print(f"  unparseable judgment for {i}, will retry next run")
            continue
        gold = gold_answer(row)
        append("judgments", {
            "id": i, "label": row["label"], "judge": c.model, "reference_model": ref["model"],
            "score": score, "acceptable": ok, "reason": v.get("reason"),
            "gold": gold, "cheap_correct": contains_answer(cheap["text"], gold) if gold else None,
            "reference_correct": contains_answer(ref["text"], gold) if gold else None,
            "paid_rate_usd": r._cost(c.model, c.input_tokens, c.output_tokens),
        })
        if n % 10 == 0:
            print(f"  judged {len(done) + n}/{len(answers)}  paid-rate est ${r.spent:.3f}")


def status() -> None:
    a, j, o, n = read("answers"), read("judgments"), read("ood"), read("annotations")
    cheap = sum(x["role"] == "cheap" for x in a)
    ref = sum(x["role"] == "reference" for x in a)
    spent = sum(x.get("paid_rate_usd", 0) for x in a + j + o + n)
    print(f"ood labels {len(o)}/{OOD_SIZE} | train annotations {len(n)}/"
          f"{sum(ANNOTATE_PLAN.values())} | cheap answers {cheap}/{PER_CLASS * 8} | "
          f"reference answers {ref}/{PER_CLASS * 8} | judgments {len(j)}/{PER_CLASS * 8} | "
          f"paid-rate equivalent ${spent:.3f}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-usd", type=float, default=5.0)
    ap.add_argument("--stages", default="ood,answers,judge")
    ap.add_argument("--limit", type=int, help="only the first N items (smoke test)")
    ap.add_argument("--labels-dir", help="write to another directory (smoke test)")
    ap.add_argument("--shard", default="0/1", help="k/n: judge only ids with id %% n == k")
    ap.add_argument("--judge-order", help="comma-separated judge models, overrides default")
    args = ap.parse_args()
    global LIMIT, LABELS, SHARD, JUDGE_ORDER
    LIMIT = args.limit
    k, n = args.shard.split("/")
    SHARD = (int(k), int(n))
    if args.judge_order:
        JUDGE_ORDER = args.judge_order.split(",")
    if args.labels_dir:
        LABELS = Path(args.labels_dir)
    LABELS.mkdir(parents=True, exist_ok=True)
    if args.status:
        return status()

    r = Runner(args.max_usd)
    stages = {"ood": stage_ood, "annotate": stage_annotate, "answers": stage_answers,
              "judge": stage_judge}
    try:
        for name in args.stages.split(","):
            print(f"== stage {name}")
            try:
                await stages[name](r)
            except (StageBlocked, DailyQuotaExhausted) as e:
                print(f"  stage {name} paused: {e}")
    except CostCapReached as e:
        print(f"STOPPED: {e}")
    finally:
        await r.client.aclose()
        status()
        if r.exhausted:
            print(f"Daily free quota used up for: {sorted(r.exhausted)}. "
                  "Re-run the same command tomorrow to resume.")


if __name__ == "__main__":
    asyncio.run(main())
