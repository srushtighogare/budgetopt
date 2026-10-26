"""Train and evaluate the task-type classifier.

Training data: ml/datasets/labels/annotations.jsonl — ~4k prompts drawn from all
six source datasets and labelled by ONE annotator with ONE rubric
(ml/labeling_run.py, stage "annotate"). A first version trained on per-source
labels scored 0.86 in-distribution but only 0.44 on independently labelled
prompts: it had learned each dataset's writing style instead of the task. Mixing
sources under a single labelling standard removes that shortcut. The final check
is the out-of-distribution set (ml/eval_ood.py), labelled by a different model.

Compares three classic text classifiers on TF-IDF features, selects the best by
validation macro-F1, reports held-out test metrics, and exports the winner for
the backend. Classic linear models are chosen deliberately: sub-millisecond CPU
inference (the router's overhead budget is <50 ms), no GPU, small artifacts and
inspectable feature weights for explainability.

Usage: python ml/train_classifier.py
Outputs:
  backend/app/services/workload/artifacts/task_classifier.joblib
  ml/eval/classifier_report.md, ml/eval/classifier_metrics.json,
  ml/eval/classifier_confusion.png
"""

import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

ROOT = Path(__file__).parent
DATA = ROOT / "datasets"
EVAL = ROOT / "eval"
ARTIFACT = ROOT.parent / "backend" / "app" / "services" / "workload" / "artifacts"
SEED = 42


def splits() -> dict[str, tuple[list[str], list[str]]]:
    """Stratified 80/10/10 split of the annotated set."""
    import random

    with open(DATA / "labels" / "annotations.jsonl") as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(SEED)
    parts: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for label in sorted({r["label"] for r in rows}):
        group = [r for r in rows if r["label"] == label]
        rng.shuffle(group)
        a, b = int(len(group) * 0.8), int(len(group) * 0.9)
        parts["train"] += group[:a]
        parts["val"] += group[a:b]
        parts["test"] += group[b:]
    return {k: ([r["text"] for r in v], [r["label"] for r in v]) for k, v in parts.items()}


def source_agreement() -> dict[str, float]:
    """How often the annotator agrees with a dataset's own label (annotator sanity check)."""
    with open(DATA / "labels" / "annotations.jsonl") as f:
        rows = [json.loads(line) for line in f]
    out = {}
    for src in sorted({r["source"] for r in rows if r["source_label"]}):
        sub = [r for r in rows if r["source"] == src and r["source_label"]]
        out[src] = round(sum(r["label"] == r["source_label"] for r in sub) / len(sub), 3)
    return out


def features() -> FeatureUnion:
    # word n-grams capture task verbs ("translate", "summarize"); char n-grams
    # capture code/math syntax ("def ", "$\\frac", "==") and are typo-robust
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=10_000,
                                 sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=15_000, sublinear_tf=True)),
    ])


def candidates() -> dict[str, list[Pipeline]]:
    grid: dict[str, list[Pipeline]] = {"logreg": [], "linear_svm": [], "complement_nb": []}
    for c in (1.0, 4.0, 16.0):
        grid["logreg"].append(Pipeline([
            ("features", features()),
            ("clf", LogisticRegression(C=c, max_iter=3000, class_weight="balanced")),
        ]))
        grid["linear_svm"].append(Pipeline([
            ("features", features()),
            # calibration gives probabilities, used as routing confidence; a single
            # calibrated model (ensemble=False) keeps the artifact ~1/3 the size
            ("clf", CalibratedClassifierCV(LinearSVC(C=c / 4, class_weight="balanced"),
                                           cv=3, ensemble=False)),
        ]))
    for a in (0.1, 0.3, 1.0):
        grid["complement_nb"].append(Pipeline([("features", features()),
                                               ("clf", ComplementNB(alpha=a))]))
    return grid


def main() -> None:
    EVAL.mkdir(exist_ok=True)
    ARTIFACT.mkdir(parents=True, exist_ok=True)
    data = splits()
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    Xte, yte = data["test"]
    agreement = source_agreement()

    results = []
    best = (None, -1.0, "")
    for family, pipes in candidates().items():
        for pipe in pipes:
            t0 = time.perf_counter()
            pipe.fit(Xtr, ytr)
            fit_s = time.perf_counter() - t0
            f1 = f1_score(yva, pipe.predict(Xva), average="macro")
            params = {k: v for k, v in pipe.named_steps["clf"].get_params().items()
                      if k in ("C", "alpha")} or {
                "C": pipe.named_steps["clf"].estimator.C}
            results.append({"family": family, "params": params, "val_macro_f1": round(f1, 4),
                            "fit_s": round(fit_s, 1)})
            print(f"{family:14} {params} val macro-F1={f1:.4f} ({fit_s:.1f}s)")
            if f1 > best[1]:
                best = (pipe, f1, family)

    model, val_f1, family = best
    pred = model.predict(Xte)
    labels = sorted(set(ytr))
    acc = accuracy_score(yte, pred)
    macro = f1_score(yte, pred, average="macro")

    # Single-prompt latency, the number that matters for routing overhead
    sample = Xte[:200]
    model.predict(sample[:5])
    t0 = time.perf_counter()
    for x in sample:
        model.predict_proba([x])
    latency_ms = (time.perf_counter() - t0) / len(sample) * 1000

    probs = model.predict_proba(Xte).max(axis=1)
    correct = np.array(pred) == np.array(yte)
    calib = {f"conf>={t}": {"share": round(float((probs >= t).mean()), 3),
                            "accuracy": round(float(correct[probs >= t].mean()), 4)}
             for t in (0.5, 0.7, 0.9) if (probs >= t).any()}

    # `stop_words_` holds every pruned term and only exists for introspection;
    # dropping it shrinks the artifact several-fold without changing predictions
    for _, vec in model.named_steps["features"].transformer_list:
        vec.stop_words_ = None
    joblib.dump({"pipeline": model, "labels": labels, "family": family,
                 "trained_on": len(Xtr)}, ARTIFACT / "task_classifier.joblib", compress=3)

    cm = confusion_matrix(yte, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=45,
                                                           colorbar=False, cmap="Blues")
    ax.set_title(f"Task classifier ({family}) — in-distribution test, n={len(yte)}")
    fig.tight_layout()
    fig.savefig(EVAL / "classifier_confusion.png", dpi=140)

    report = classification_report(yte, pred, labels=labels, digits=3)
    metrics = {"selected": family, "val_macro_f1": round(val_f1, 4),
               "test_accuracy": round(acc, 4), "test_macro_f1": round(macro, 4),
               "latency_ms_per_prompt": round(latency_ms, 3), "confidence": calib,
               "n_train": len(Xtr), "n_val": len(Xva), "n_test": len(Xte),
               "annotator_vs_source_label_agreement": agreement,
               "model_selection": results}
    (EVAL / "classifier_metrics.json").write_text(json.dumps(metrics, indent=2))

    rows = "\n".join(f"| {r['family']} | {r['params']} | {r['val_macro_f1']} |" for r in results)
    (EVAL / "classifier_report.md").write_text(f"""# Task classifier — evaluation

Data: {len(Xtr)} train / {len(Xva)} validation / {len(Xte)} test prompts across
{len(labels)} task types: prompts from six public datasets, all labelled by one LLM
annotator with one rubric (`ml/labeling_run.py`, stage `annotate`).

Annotator agreement with each dataset's own labels (sanity check):
`{json.dumps(agreement)}`

## Model selection (validation macro-F1)

| Model | Hyper-parameters | Val macro-F1 |
|---|---|---|
{rows}

**Selected:** {family}

## Held-out test set

- Accuracy: **{acc:.3f}**
- Macro-F1: **{macro:.3f}**
- Inference latency: **{latency_ms:.2f} ms** per prompt (single-threaded CPU)

```
{report}
```

Confidence vs accuracy: `{json.dumps(calib)}`

![confusion matrix](classifier_confusion.png)
""")
    print(f"\nselected {family}: test acc={acc:.4f} macro-F1={macro:.4f} "
          f"latency={latency_ms:.2f}ms")
    print(report)


if __name__ == "__main__":
    main()
