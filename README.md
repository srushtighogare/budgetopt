# BudgetOpt

**AI based Cost-Optimal LLM Selection and Token Budget Optimization Platform**
Final-year project, Department of Computer Engineering, Vishwakarma Institute of Technology (Group BE-A5).

BudgetOpt helps organizations control LLM spending. It classifies each AI workload by task type and complexity, estimates token usage, and allocates work across LLMs. Allocation uses multi-objective optimization (NSGA-II) over cost, quality and latency. BudgetOpt also enforces team budgets, forecasts spend and supports what-if analysis, and it explains every routing decision.

The full design and the build roadmap are in [docs/plan.md](docs/plan.md).

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Foundation: FastAPI, PostgreSQL schema + Alembic, JWT auth, RBAC, teams, API keys, model and versioned pricing registry, audit log, CI | **Done** |
| 2 | Provider adapters (Groq, Gemini, OpenAI, Anthropic), fallback executor, circuit breaker | **Done** |
| 3 | OpenAI-compatible gateway, request and decision logging, exact cache, rate limits, PII redaction | **Done** |
| 4 | Workload analysis ML: task classifier, complexity model, token estimator | Classifier done; complexity & token models training on labelled data |
| 5 | Model profiling: per-task quality via LLM judge + gold answers, latency percentiles | **Done** (profiles filling on free tiers) |
| 6 | Optimizer: NSGA-II and ILP allocation, constraint filter, explanations | Next |
| 7–9 | Forecasting, anomaly detection, what-if simulator, semantic cache | Planned |
| 10 | React dashboard | Planned |
| 11 | Benchmark and evaluation | Planned |

## Models

Seven models from two providers are registered (Groq: GPT-OSS 20B/120B, Qwen 3.8 27B;
Google: Gemini 3.5 Flash, 3.5 Flash-Lite, 3.1 Flash-Lite, 3 Flash preview). They are
interchangeable, and any model can be added through the API with its pricing. The development
runs use free-tier quotas only.

## Tech stack

FastAPI · SQLAlchemy 2 (async) · Alembic · PostgreSQL 16 (pgvector) · Redis · scikit-learn · pymoo · OR-Tools · statsmodels · React + TypeScript + Vite

## Running locally

```bash
cp .env.example .env          # fill in JWT_SECRET, FERNET_KEY and provider keys
docker compose up -d postgres redis
cd backend
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/alembic upgrade head
.venv/bin/python -m app.seed
.venv/bin/uvicorn app.main:app --reload
```

The API docs are at http://localhost:8000/docs. Log in with `SEED_ADMIN_EMAIL` / `SEED_ADMIN_PASSWORD`.

## Using the gateway from an application

Create a team API key (`POST /api/v1/teams/{id}/api-keys`), then point any OpenAI SDK at BudgetOpt:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="bo_...")
r = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hi"}])
print(r.model_extra["budgetopt"])  # cost, savings vs baseline, budget state, explanation
```

`model="auto"` lets BudgetOpt choose the model. A registered model name requests that model, and the other eligible models serve as fallbacks. Over-budget teams get `402`, rate-limited callers get `429`, and every request appears in `GET /api/v1/requests` with an explanation of the routing decision.

## Model profiling

```bash
cd backend
.venv/bin/python -m app.services.profiling.cli quality --per-task 4   # resumable
.venv/bin/python -m app.services.profiling.cli summary                # quality/latency matrix
```

Each model answers a fixed set of 64 prompts (8 per task type). Math answers are checked
against gold answers. Other answers are graded 1–10 by a judge model, and a model never
judges itself. Results are stored per answer and aggregated into per-task quality scores.
Latency p50/p95 comes from live traffic plus profiling calls.

## Tests

Tests run against a real PostgreSQL database (`budgetopt_test`) and Redis (db 15):

```bash
docker compose exec postgres psql -U budgetopt -c "CREATE DATABASE budgetopt_test"
cd backend && .venv/bin/pytest
```

## Project structure

```
backend/
  app/
    api/v1/      REST endpoints (auth, users, teams, models, providers, requests,
                 gateway incl. OpenAI-compatible /v1/chat/completions)
    core/        settings, security (argon2, JWT, Fernet), RBAC dependencies
    db/          ORM schema and session
    schemas/     request/response models
    services/    pricing, budget, cache, rate_limit, privacy, tokens, providers/ (LLM
                 adapters), routing/ (router, pipeline, fallback executor, breaker),
                 workload/ (classifier, complexity, token estimates), profiling/
    seed.py      demo org, teams, admin, providers, models
  migrations/    Alembic migrations
  tests/
ml/              datasets, labelling job, training and evaluation scripts (see ml/README.md)
docs/            design plan, and later evaluation results
docker-compose.yml
```

## Team

Srushti Ghogare · Aditya Sonone · Kirtiraj Nagargoje. Guide: Prof. Sheetal Phatangare.
