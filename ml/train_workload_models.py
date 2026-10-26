"""Train the complexity model and the output-token estimator from the labelling run.

Complexity label (per prompt): 1 if the cheap model's answer was NOT good enough,
i.e. the judge marked it unacceptable, or (GSM8K/MATH) its final answer was wrong.
The model therefore predicts P(cheap model fails), which is exactly the quantity
the router needs.

Token estimator target: log1p(output tokens billed by the cheap model).

Both use the same features the backend computes at request time
(app.services.workload.analyzer.model_input), with the task type taken from the
deployed classifier, so training and serving see identical inputs. With ~800
rows, models are compared with stratified 5-fold cross-validation and then
refit on all data.

Usage: python ml/train_workload_models.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parent / "backend"))

from app.services.workload.analyzer import (  # noqa: E402
    ARTIFACTS,
    DEFAULT_OUTPUT_TOKENS,
    model_input,
)

LABELS = ROOT / "datasets" / "labels"
EVAL = ROOT / "eval"
SEED = 42


def read(name: str) -> list[dict]:
    with open(LABELS / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f]


def build() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[dict]]:
    with open(ROOT / "datasets" / "train.jsonl") as f:
        texts = {r["id"]: r["text"] for r in map(json.loads, f)}
    cheap = {a["id"]: a for a in read("answers") if a["role"] == "cheap"}
    judged = {j["id"]: j for j in read("judgments")}
    clf = joblib.load(ARTIFACTS / "task_classifier.joblib")["pipeline"]

    ids = [i for i in judged if i in cheap]
    pred_tasks = clf.predict([texts[i] for i in ids])
    X = np.vstack([model_input(texts[i], t) for i, t in zip(ids, pred_tasks, strict=True)])
    fail = np.array([
        int(not judged[i]["acceptable"] or judged[i].get("cheap_correct") is False)
        for i in ids
    ])
    out_tokens = np.array([cheap[i]["output_tokens"] for i in ids], dtype=float)
    meta = [{"id": i, "label": judged[i]["label"], "task_pred": t}
            for i, t in zip(ids, pred_tasks, strict=True)]
    return X, fail, out_tokens, list(pred_tasks), meta


def eval_complexity(X: np.ndarray, y: np.ndarray, tasks: list[str]) -> tuple[object, dict]:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    candidates = {
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
            random_state=SEED),
        "logistic_regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(C=0.5, max_iter=2000)),
    }
    # Baseline: per-task failure rate (what a task-only rule would know)
    task_arr = np.array(tasks)
    base = np.zeros(len(y))
    for tr, te in cv.split(X, y):
        rates = {t: y[tr][task_arr[tr] == t].mean() if (task_arr[tr] == t).any() else y[tr].mean()
                 for t in set(tasks)}
        base[te] = [rates[t] for t in task_arr[te]]
    results = {"task_prior_baseline": {"roc_auc": round(roc_auc_score(y, base), 4),
                                       "brier": round(brier_score_loss(y, base), 4)}}
    best, best_auc = None, -1.0
    for name, model in candidates.items():
        p = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        auc = roc_auc_score(y, p)
        results[name] = {"roc_auc": round(auc, 4), "brier": round(brier_score_loss(y, p), 4)}
        if auc > best_auc:
            best, best_auc, best_p = (name, model), auc, p
    name, model = best
    model.fit(X, y)

    # Routing utility: send prompts below a threshold to the cheap model
    utility = []
    for thr in (0.2, 0.33, 0.5, 0.66):
        to_cheap = best_p < thr
        utility.append({
            "threshold": thr, "share_to_cheap": round(float(to_cheap.mean()), 3),
            "cheap_failure_rate_when_routed": round(float(y[to_cheap].mean()), 3)
            if to_cheap.any() else None,
        })
    return model, {"selected": name, "positive_rate": round(float(y.mean()), 3),
                   "cv": results, "routing_utility": utility}


def eval_tokens(X: np.ndarray, out_tokens: np.ndarray, tasks: list[str]) -> tuple[object, dict]:
    cv = KFold(5, shuffle=True, random_state=SEED)
    y = np.log1p(out_tokens)
    candidates = {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
            random_state=SEED),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }
    results = {}
    median_pred = np.array([DEFAULT_OUTPUT_TOKENS[t] for t in tasks], dtype=float)
    results["task_median_baseline"] = {"mae_tokens": round(mean_absolute_error(out_tokens,
                                                                               median_pred), 1)}
    best, best_mae = None, float("inf")
    for name, model in candidates.items():
        pred = np.expm1(cross_val_predict(model, X, y, cv=cv))
        mae = mean_absolute_error(out_tokens, pred)
        within = float(np.mean(np.abs(pred - out_tokens) <= 0.5 * out_tokens))
        results[name] = {"mae_tokens": round(mae, 1), "within_50pct": round(within, 3)}
        if mae < best_mae:
            best, best_mae = (name, model), mae
    name, model = best
    model.fit(X, y)
    return model, {"selected": name, "median_output_tokens": float(np.median(out_tokens)),
                   "cv": results}


def main() -> None:
    X, fail, out_tokens, tasks, _ = build()
    print(f"rows: {len(X)}  cheap-model failure rate: {fail.mean():.3f}")

    cx_model, cx_metrics = eval_complexity(X, fail, tasks)
    te_model, te_metrics = eval_tokens(X, out_tokens, tasks)
    joblib.dump({"model": cx_model, "trained_on": len(X), "metrics": cx_metrics},
                ARTIFACTS / "complexity.joblib", compress=3)
    joblib.dump({"model": te_model, "trained_on": len(X), "metrics": te_metrics},
                ARTIFACTS / "token_estimator.joblib", compress=3)

    per_task = {t: {"n": int(np.sum(np.array(tasks) == t)),
                    "cheap_failure_rate": round(float(fail[np.array(tasks) == t].mean()), 3),
                    "median_output_tokens": float(np.median(out_tokens[np.array(tasks) == t]))}
                for t in sorted(set(tasks))}
    metrics = {"n": len(X), "complexity": cx_metrics, "token_estimator": te_metrics,
               "per_task": per_task}
    EVAL.mkdir(exist_ok=True)
    (EVAL / "workload_models_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
