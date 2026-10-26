"""Build the fixed prompt set used to profile model quality per task type.

8 prompts per task type, drawn from the single-annotator labelled pool
(ml/datasets/labels/annotations.jsonl). Short prompts are preferred to keep
profiling inside free-tier quotas. GSM8K / MATH items carry their gold final
answer so math quality is measured objectively, not only by a judge.

The set is committed and never changes between runs, so quality numbers for
different models and dates are comparable.

Usage: python ml/build_profiling_set.py
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import labeling_run as lr  # noqa: E402

PER_TASK = 8
MAX_CHARS = 1200
SEED = 11
OUT = ROOT.parent / "backend" / "app" / "services" / "profiling" / "profiling_set.jsonl"


def main() -> None:
    with open(lr.LABELS / "annotations.jsonl") as f:
        rows = [r for r in map(json.loads, f) if len(r["text"]) <= MAX_CHARS]
    rng = random.Random(SEED)
    items = []
    for task in lr.TASKS:
        group = [r for r in rows if r["label"] == task]
        if task == "reasoning_math":
            # only items whose gold answer we can check
            group = [r for r in group if r["source"] in ("gsm8k", "math")]
        rng.shuffle(group)
        for r in group[:PER_TASK]:
            gold = None
            if r["source"] == "gsm8k":
                gold = lr._gold_index("gsm8k").get(r["text"])
            elif r["source"] == "math":
                for sub in ("algebra", "counting_and_probability", "geometry",
                            "intermediate_algebra", "number_theory", "prealgebra", "precalculus"):
                    gold = gold or lr._gold_index(f"math:{sub}").get(r["text"])
            items.append({"item_id": f"{task[:4]}-{len(items):03d}", "task_type": task,
                          "text": r["text"], "gold": gold, "source": r["source"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with_gold = sum(1 for it in items if it["gold"])
    print(f"{len(items)} items ({with_gold} with gold answers) -> {OUT}")


if __name__ == "__main__":
    main()
