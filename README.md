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
| 3 | OpenAI-compatible gateway, request and decision logging, exact cache, rate limits | Next |
| 4 | Workload analysis ML: task classifier, complexity model, token estimator | Planned |
| 5 | Model profiling: LLM-as-judge quality, latency tracking | Planned |
| 6 | Optimizer: NSGA-II and ILP allocation, constraint filter, explanations | Planned |
| 7–9 | Forecasting, anomaly detection, what-if simulator, semantic cache | Planned |
| 10 | React dashboard | Planned |
| 11 | Benchmark and evaluation | Planned |

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

## Tests

Tests run against a real PostgreSQL database, `budgetopt_test`:

```bash
docker compose exec postgres psql -U budgetopt -c "CREATE DATABASE budgetopt_test"
cd backend && .venv/bin/pytest
```

## Project structure

```
backend/
  app/
    api/v1/      REST endpoints (auth, users, teams, models, providers)
    core/        settings, security (argon2, JWT, Fernet), RBAC dependencies
    db/          ORM schema and session
    schemas/     request/response models
    services/    pricing, budget, providers/ (LLM adapters), routing/ (fallback executor,
                 circuit breaker); more modules land here per phase
    seed.py      demo org, teams, admin, providers, models
  migrations/    Alembic migrations
  tests/
docs/            design plan, and later evaluation results
docker-compose.yml
```

## Team

Srushti Ghogare · Aditya Sonone · Kirtiraj Nagargoje. Guide: Prof. Sheetal Phatangare.
