"""Evaluate the task classifier on the out-of-distribution (OOD) test set.

The in-distribution test set shares its labelling rules with the training data
(source metadata / keyword rules). The OOD set is natural Alpaca prompts those
rules did NOT capture, labelled independently by an LLM annotator
(ml/labeling_run.py, stage "ood"), so it estimates real-world accuracy.

Usage: python ml/eval_ood.py   (appends a section to ml/eval/classifier_report.md)
"""

import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

ROOT = Path(__file__).parent
EVAL = ROOT / "eval"
ARTIFACT = ROOT.parent / "backend" / "app" / "services" / "workload" / "artifacts"
MARKER = "\n## Out-of-distribution test set"


def main() -> None:
    with open(ROOT / "datasets" / "labels" / "ood.jsonl") as f:
        rows = [json.loads(line) for line in f]
    model = joblib.load(ARTIFACT / "task_classifier.joblib")["pipeline"]
    texts, gold = [r["text"] for r in rows], [r["label"] for r in rows]

    probs = model.predict_proba(texts)
    classes = np.array(model.classes_)
    pred = classes[probs.argmax(axis=1)]
    top2 = classes[np.argsort(probs, axis=1)[:, -2:]]
    acc = accuracy_score(gold, pred)
    macro = f1_score(gold, pred, average="macro")
    top2_acc = float(np.mean([g in t for g, t in zip(gold, top2, strict=True)]))
    labels = sorted(set(gold) | set(pred))

    # Ceiling: how often two different LLM annotators agree on these same prompts
    second_path = ROOT / "datasets" / "labels" / "ood_second_annotator.jsonl"
    agreement = None
    if second_path.exists():
        with open(second_path) as f:
            second = {r["ood_id"]: r["label"] for r in map(json.loads, f) if r["label"]}
        pairs = [(r["label"], second[r["ood_id"]]) for r in rows if r["ood_id"] in second]
        agreement = sum(a == b for a, b in pairs) / len(pairs)

    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(gold, pred, labels=labels, ax=ax,
                                            xticks_rotation=45, colorbar=False, cmap="Oranges")
    ax.set_title(f"Task classifier — OOD test set, n={len(rows)}")
    fig.tight_layout()
    fig.savefig(EVAL / "classifier_confusion_ood.png", dpi=140)

    metrics_path = EVAL / "classifier_metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["ood"] = {"n": len(rows), "accuracy": round(acc, 4), "macro_f1": round(macro, 4),
                      "top2_accuracy": round(top2_acc, 4),
                      "inter_annotator_agreement": round(agreement, 4) if agreement else None,
                      "label_distribution": dict(Counter(gold))}
    metrics_path.write_text(json.dumps(metrics, indent=2))

    report_path = EVAL / "classifier_report.md"
    report = report_path.read_text().split(MARKER)[0]
    annotators = Counter(r["annotator"] for r in rows)
    report_path.write_text(report + f"""{MARKER}

{len(rows)} natural Alpaca prompts that the dataset rules did not label, annotated
independently by an LLM ({dict(annotators)}). This is the more honest estimate of
real-world accuracy.

- Accuracy: **{acc:.3f}**
- Macro-F1: **{macro:.3f}**
- Top-2 accuracy: **{top2_acc:.3f}**
- Inter-annotator agreement (a second LLM annotator on the same prompts):
  **{agreement:.3f}** — the practical ceiling for any classifier on these labels, since
  the task types genuinely overlap (e.g. "edit this text", "explain this code")
- Label distribution: `{dict(sorted(Counter(gold).items()))}`

```
{classification_report(gold, pred, labels=labels, digits=3, zero_division=0)}
```

![OOD confusion matrix](classifier_confusion_ood.png)
""")
    print(f"OOD n={len(rows)} acc={acc:.4f} macro-F1={macro:.4f} top2={top2_acc:.4f}")
    print(classification_report(gold, pred, labels=labels, digits=3, zero_division=0))


if __name__ == "__main__":
    main()
