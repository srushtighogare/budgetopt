PY := backend/.venv/bin/python
RAW := ml/datasets/raw
HF := https://huggingface.co/datasets

.PHONY: up test lint data dataset label label-status train-classifier eval-ood train-workload ml

up:              ## start Postgres and Redis
	docker compose up -d postgres redis

test:            ## run backend tests
	cd backend && .venv/bin/pytest -q

lint:
	backend/.venv/bin/ruff check backend ml

data:            ## download the public datasets (~70 MB)
	mkdir -p $(RAW)/dolly $(RAW)/alpaca $(RAW)/codealpaca $(RAW)/gsm8k $(RAW)/mbpp $(RAW)/math
	curl -sfL -o $(RAW)/dolly/dolly.jsonl $(HF)/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
	curl -sfL -o $(RAW)/alpaca/alpaca.json $(HF)/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json
	curl -sfL -o $(RAW)/codealpaca/codealpaca.json $(HF)/sahil2801/CodeAlpaca-20k/resolve/main/code_alpaca_20k.json
	curl -sfL -o $(RAW)/gsm8k/train.parquet $(HF)/openai/gsm8k/resolve/main/main/train-00000-of-00001.parquet
	for s in train test validation; do curl -sfL -o $(RAW)/mbpp/$$s.parquet $(HF)/google-research-datasets/mbpp/resolve/main/full/$$s-00000-of-00001.parquet; done
	for t in algebra counting_and_probability geometry intermediate_algebra number_theory prealgebra precalculus; do \
	  curl -sfL -o $(RAW)/math/$$t.parquet $(HF)/EleutherAI/hendrycks_math/resolve/main/$$t/train-00000-of-00001.parquet; done

dataset:         ## build source-labelled splits (used to sample prompts for labelling)
	$(PY) -I ml/build_dataset.py

label:           ## run/resume the free-tier labelling job
	$(PY) -I -u ml/labeling_run.py

label-status:
	$(PY) -I ml/labeling_run.py --status

train-classifier:
	$(PY) -I ml/train_classifier.py

eval-ood:
	$(PY) -I ml/eval_ood.py

train-workload:  ## complexity model + output-token estimator
	$(PY) -I ml/train_workload_models.py

ml: train-classifier eval-ood train-workload
