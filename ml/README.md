# Workload analysis — data and models

Scripts that build the data and train the three models the backend uses for
workload analysis (`backend/app/services/workload/`). Run them from the repo root;
the `Makefile` has a target for each step.

| Step | Command | Output |
|---|---|---|
| Download public datasets (~70 MB) | `make data` | `ml/datasets/raw/` (not committed) |
| Source-labelled splits | `make dataset` | `ml/datasets/{train,val,test}.jsonl` |
| Free-tier LLM labelling (resumable) | `make label` | `ml/datasets/labels/*.jsonl` |
| Task classifier | `make train-classifier` | `artifacts/task_classifier.joblib`, `ml/eval/classifier_*` |
| Out-of-distribution check | `make eval-ood` | appends to `ml/eval/classifier_report.md` |
| Complexity model + token estimator | `make train-workload` | `artifacts/complexity.joblib`, `artifacts/token_estimator.joblib` |

## Labelling run (`labeling_run.py`)

All labelling uses free-tier API quotas and records which model produced each label.
A paid-rate cost estimate is tracked and the run stops at `--max-usd` (default $5) as a
safety net.

- **annotate**: about 4k prompts from all six datasets, labelled into 8 task types by one
  annotator with one rubric. This is the classifier's training data.
- **ood**: 320 natural Alpaca prompts labelled by a *different* annotator, used as an
  out-of-distribution test set. A second annotator's labels measure inter-annotator agreement.
- **answers / judge**: 800 prompts (100 per task type, interleaved). Each gets an answer
  from a cheap model (GPT-OSS 20B) and from a strong reference model, and a judge scores
  the cheap answer. Math rows are also checked against gold answers. These give the complexity
  label ("cheap model failed") and real output-token counts.

## Datasets and licences

| Dataset | Source | Licence |
|---|---|---|
| databricks-dolly-15k | huggingface.co/datasets/databricks/databricks-dolly-15k | CC BY-SA 3.0 |
| Alpaca (cleaned) | huggingface.co/datasets/yahma/alpaca-cleaned | CC BY 4.0 |
| CodeAlpaca-20k | huggingface.co/datasets/sahil2801/CodeAlpaca-20k | CC BY 4.0 |
| GSM8K | huggingface.co/datasets/openai/gsm8k | MIT |
| MBPP | huggingface.co/datasets/google-research-datasets/mbpp | CC BY 4.0 |
| MATH | huggingface.co/datasets/EleutherAI/hendrycks_math | MIT |

The JSONL files in `ml/datasets/` contain prompts derived from these datasets, and are
redistributed under their original licences for research and education.
