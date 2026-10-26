"""Build the labelled prompt dataset for the task classifier.

Labels come from each source dataset's own metadata (no LLM labelling), so the
data is free to build and fully reproducible:

  translation       Alpaca-cleaned instructions mentioning translation (non-code)
  summarization     Dolly `summarization` + Alpaca "summar*" instructions
  simple_qa         Dolly `open_qa`, `general_qa`, `closed_qa`
  extraction        Dolly `information_extraction`, `classification`
  creative_writing  Dolly `creative_writing`, `brainstorming`
  coding            CodeAlpaca-20k + MBPP
  reasoning_math    GSM8K + MATH (all subjects; MATH difficulty level kept)
  analysis          Alpaca instructions led by analytic verbs (compare, evaluate, ...)

Keyword-derived classes (translation, summarization-from-Alpaca, analysis) can
make an in-distribution test look easier than reality, so the classifier is
additionally evaluated on an independently labelled test set
(see label_ood_testset.py).

Usage: python ml/build_dataset.py   (reads ml/datasets/raw, writes ml/datasets/*.jsonl)
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
RAW = ROOT / "datasets" / "raw"
OUT = ROOT / "datasets"
SEED = 42
CAP_PER_CLASS = 700
MAX_CHARS = 6000  # long Dolly contexts are truncated; the task signal is up front

CODE = re.compile(r"\b(code|python|java|javascript|sql|function|program|script|algorithm|html|"
                  r"c\+\+|binary|regex)\b", re.I)
TRANSLATE = re.compile(r"\btranslat", re.I)
SUMMARIZE = re.compile(r"\bsummar", re.I)
ANALYSIS = re.compile(
    r"^\s*(compare|contrast|analy[sz]e|evaluate|assess|critique|critically|discuss|"
    r"explain (why|how)|what are the (advantages|disadvantages|pros|benefits|drawbacks|"
    r"implications|consequences|differences)|argue|justify|examine|weigh)\b", re.I)

DOLLY_MAP = {
    "open_qa": "simple_qa", "general_qa": "simple_qa", "closed_qa": "simple_qa",
    "information_extraction": "extraction", "classification": "extraction",
    "summarization": "summarization",
    "creative_writing": "creative_writing", "brainstorming": "creative_writing",
}


def join(instruction: str, extra: str | None) -> str:
    text = instruction.strip()
    if extra and extra.strip():
        text += "\n\n" + extra.strip()
    return text[:MAX_CHARS]


def load() -> list[dict]:
    rows: list[dict] = []

    with open(RAW / "dolly" / "dolly.jsonl") as f:
        dolly = [json.loads(line) for line in f]
    with open(RAW / "alpaca" / "alpaca.json") as f:
        alpaca = json.load(f)
    with open(RAW / "codealpaca" / "codealpaca.json") as f:
        codealpaca = json.load(f)

    for x in dolly:
        rows.append({"text": join(x["instruction"], x["context"]),
                     "label": DOLLY_MAP[x["category"]], "source": f"dolly:{x['category']}"})

    for x in alpaca:
        ins = x["instruction"]
        if CODE.search(ins + " " + x["input"]):
            continue
        label = ("translation" if TRANSLATE.search(ins) else
                 "analysis" if ANALYSIS.search(ins) else
                 "summarization" if SUMMARIZE.search(ins) else None)
        if label:
            rows.append({"text": join(ins, x["input"]), "label": label, "source": "alpaca"})

    for x in codealpaca:
        rows.append({"text": join(x["instruction"], x["input"]), "label": "coding",
                     "source": "codealpaca"})
    for split in ("train", "validation", "test"):
        for t in pd.read_parquet(RAW / "mbpp" / f"{split}.parquet")["text"]:
            rows.append({"text": t.strip(), "label": "coding", "source": "mbpp"})

    for q in pd.read_parquet(RAW / "gsm8k" / "train.parquet")["question"]:
        rows.append({"text": q.strip(), "label": "reasoning_math", "source": "gsm8k"})
    for f in sorted((RAW / "math").glob("*.parquet")):
        for _, r in pd.read_parquet(f).iterrows():
            level = int(r["level"].split()[-1]) if r["level"][-1].isdigit() else None
            rows.append({"text": r["problem"].strip(), "label": "reasoning_math",
                         "source": f"math:{f.stem}", "math_level": level})
    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for r in rows:
        key = re.sub(r"\W+", " ", r["text"].lower()).strip()
        if key and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def balanced_sample(rows: list[dict], rng: random.Random) -> list[dict]:
    """Cap each class, sampling evenly across its sources so no source dominates."""
    out = []
    for label in sorted({r["label"] for r in rows}):
        by_src: dict[str, list[dict]] = {}
        for r in rows:
            if r["label"] == label:
                by_src.setdefault(r["source"].split(":")[0], []).append(r)
        for v in by_src.values():
            rng.shuffle(v)
        picked: list[dict] = []
        while len(picked) < CAP_PER_CLASS and any(by_src.values()):
            for v in by_src.values():
                if v and len(picked) < CAP_PER_CLASS:
                    picked.append(v.pop())
        out.extend(picked)
    return out


def split(rows: list[dict], rng: random.Random) -> dict[str, list[dict]]:
    """Stratified 70/15/15 split."""
    parts = {"train": [], "val": [], "test": []}
    for label in sorted({r["label"] for r in rows}):
        group = [r for r in rows if r["label"] == label]
        rng.shuffle(group)
        n = len(group)
        a, b = int(n * 0.70), int(n * 0.85)
        parts["train"] += group[:a]
        parts["val"] += group[a:b]
        parts["test"] += group[b:]
    return parts


def main() -> None:
    rng = random.Random(SEED)
    rows = balanced_sample(dedupe(load()), rng)
    for i, r in enumerate(rows):
        r["id"] = i
    parts = split(rows, rng)
    for name, part in parts.items():
        with open(OUT / f"{name}.jsonl", "w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:5} {len(part):5}  {dict(sorted(Counter(r['label'] for r in part).items()))}")


if __name__ == "__main__":
    main()
