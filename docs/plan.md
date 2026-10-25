# BudgetOpt — Build Plan (from the submitted project report)

## Context
The FF-180 report (VIT, BE-A5, guide Sheetal Phatangare) commits to: **"AI based Cost-Optimal LLM Selection and Token Budget Optimization Platform"**. The existing repo is a small prototype (Streamlit, SQLite, keyword-only routing between a cheap and a premium tier) that does not match the report. The user wants it treated as a **new project**, built on the report's stack, and technically sound enough for a final-year evaluation. Previous code is only a reference. One useful idea to reuse is the try-primary-then-fallback pattern in `backend/executor.py`.

### What the report promises (it is the acceptance checklist)
| # | Report commitment | How we deliver it |
|---|---|---|
| O1 | Platform for LLM cost and token optimization | FastAPI backend, React dashboard, PostgreSQL |
| O2 | Classify workloads by **task type and complexity** | ML task classifier (7–8 classes) and a complexity regressor |
| O3 | Estimate **tokens, quality need, latency** per workload | Output-token predictor, per-task minimum quality, latency profiles |
| O4 | Select and **allocate** workloads across LLMs | Real-time router driven by an allocation policy table |
| O5 | **Multi-objective optimization** of cost, quality and latency | **NSGA-II** (pymoo) Pareto front, plus an **ILP** (OR-Tools/PuLP) for budget-constrained allocation |
| O6 | Monitoring, **forecasting**, **what-if** | Spend forecasting (Holt-Winters/ARIMA), overrun alerts, a simulator that replays history |
| Arch 3 | Model and pricing layer: cost, quality, latency, token limits | `models` registry with pricing history and measured performance |
| Arch 6 | Optional LLM integration layer | OpenAI-compatible `/v1/chat/completions` gateway with per-app API keys |
| Env | Caching of repeated requests | Exact-match cache plus semantic cache (Redis) |
| Safety | Auth, RBAC, API-key protection, validation, encryption, abuse monitoring | JWT and argon2, roles, Fernet-encrypted provider keys, rate limits, anomaly detection |
| Ethics | Never pick a model just because it is cheaper; explainability | Hard quality, latency and privacy constraints; a "why this model / savings" explanation on every decision |
| Econ | Compare allocations, quantify savings before changing production | What-if module and a savings report |

Two problems in the report to flag to the user (not code): the Environment section mentions "digital vehicle damage assessment", which looks copy-pasted from another project. The Sustainability bullet list also has broken formatting. Both should be fixed before Review 1.

## Target stack
- **Backend:** Python 3.12, FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2
- **DB:** PostgreSQL 16, with **pgvector** for semantic-cache embeddings; **Redis** for the cache, rate limits and the job queue
- **ML/Opt:** scikit-learn, sentence-transformers (`all-MiniLM-L6-v2`), **pymoo** (NSGA-II), **OR-Tools** (CP-SAT/ILP), statsmodels (forecasting)
- **Background jobs:** ARQ or Celery workers for profiling, re-optimization and forecast refresh
- **Frontend:** React 18, Vite, TypeScript, Tailwind, shadcn/ui, TanStack Query, Recharts, React Router
- **Providers (adapter interface):** Groq, Google Gemini, OpenAI, Anthropic. Each one is a class implementing `generate()`, `count_tokens()` and `pricing`
- **Infra:** Docker Compose (api, worker, postgres, redis, frontend), GitHub Actions CI (ruff, mypy, pytest, eslint, vitest)

## Repository layout (new)
```
budgetopt/
  backend/app/
    api/v1/          auth, teams, users, models, requests, gateway, optimization, forecast, whatif, analytics
    core/            config, security (jwt, hashing, fernet), rbac, rate_limit, logging
    db/              session, base, models/ (ORM), alembic/
    schemas/         pydantic DTOs
    services/
      providers/     base.py, groq.py, gemini.py, openai.py, anthropic.py, registry.py
      workload/      classifier.py, complexity.py, token_estimator.py, features.py
      profiling/     quality_eval.py (LLM-as-judge), latency_tracker.py (EWMA)
      optimizer/     problem.py (NSGA-II), allocator.py (ILP), policy.py
      routing/       router.py, constraints.py, explainer.py, fallback.py
      cache/         exact.py, semantic.py
      budget/        enforcer.py, forecaster.py, anomaly.py
      whatif/        simulator.py
    workers/         tasks.py (profiling, reoptimize, forecast refresh)
  ml/                datasets/, train_classifier.py, train_token_estimator.py, eval/ (reports, plots)
  frontend/src/      pages/, components/, api/, hooks/
  benchmarks/        run_benchmark.py, results/
  docs/              architecture, API, evaluation results (feeds the final report)
  docker-compose.yml, .env.example, Makefile
```

## Data model (PostgreSQL)
- `organizations`, `teams` (monthly_budget, alert_threshold, policy_id), `users` (role: org_admin | team_manager | member, team_id), `api_keys` (stored as a hash, scoped to a team)
- `providers` (Fernet-encrypted API key), `llm_models` (context_window, max_output, supports_privacy_tier, active)
- `model_pricing` (input/output price per 1M tokens, effective_from): **versioned**, so old costs stay correct when prices change
- `model_performance` (model_id, task_type, quality_score, p50/p95 latency, sample_n, updated_at)
- `requests` (team, user, task_type, complexity, est_tokens, actual in/out tokens, model, cost, latency_ms, cache_hit, status). Prompts are stored **hashed by default**; raw text only when a team opts in (privacy)
- `routing_decisions` (request_id, candidates JSON, constraints applied, chosen model, baseline_model, baseline_cost, savings, explanation)
- `optimization_runs` (inputs, Pareto front JSON, selected solution), `allocation_policies` (task_type × complexity_band → ordered model list)
- `forecasts` (team, horizon, predicted spend, CI, overrun_date), `alerts` (budget/anomaly), `audit_events` (who viewed or changed what)

## Core algorithms (the technical heart, to explain in the viva)

**1. Workload analysis (O2, O3)**
- *Task classifier:* MiniLM sentence embedding, then logistic regression (try LightGBM as a comparison). Classes: translation, summarization, simple_qa, extraction, creative_writing, coding, reasoning_math, analysis. Training data is roughly 3–5k labelled prompts assembled from public sets (e.g. Dolly-15k categories, GSM8K, HumanEval/MBPP, CNN-DM, WMT, LMSYS samples), with a held-out test set. Report accuracy, macro-F1 and a confusion matrix.
- *Complexity:* a regressor on the embedding plus handcrafted features (length, number of sub-questions, code blocks, constraints). Labels come from "did the cheap model pass the judge?" on the profiling set, so complexity means *probability a cheap model fails*. This is better grounded than keyword weights.
- *Token estimator:* input tokens via tiktoken or the provider tokenizer (exact). Output tokens via gradient-boosted regression on task type, input length and complexity. Report MAE.

**2. Model profiling (Arch 3)**
- An offline job runs a fixed evaluation set per task type through every model. An **LLM-as-judge** (strongest model, rubric 1–10, plus reference answers where they exist: exact match for GSM8K, unit tests for code) produces `quality[model][task]`.
- Latency comes from live traffic as an EWMA of p50/p95. Pricing comes from the versioned table.

**3. Multi-objective optimization (O5)**
- *Decision variables:* fraction x[t][m] of workload class t (task × complexity band) sent to model m, with Σ_m x[t][m] = 1.
- *Objectives (NSGA-II, pymoo):* minimize expected cost = Σ volume_t · x · (in_tok·p_in + out_tok·p_out); maximize volume-weighted quality; minimize volume-weighted p95 latency.
- *Constraints:* quality ≥ q_min[t], latency ≤ L_max[t], tokens ≤ context window, privacy tier, total cost ≤ team budget (from the forecast volume).
- *Output:* a Pareto front shown in React as a 3-D/2-D scatter. The admin picks a point or a preset ("cost-saver", "balanced", "quality-first"), and that becomes the `allocation_policy`.
- *ILP (OR-Tools):* a single-objective, exact solve, used for a hard requirement such as "maximize quality subject to budget ≤ ₹X". It also serves as a baseline to validate NSGA-II (hypervolume and gap to the ILP optimum).

**4. Real-time routing (O4, Ethics)**
Per request (target under 50 ms overhead):
1. Validate and authenticate, check the rate limit
2. Cache lookup (exact hash, then semantic cosine above 0.95 within the team)
3. Classify, score complexity and estimate tokens
4. Budget gate. Hard block at ≥100%; at the soft threshold, switch to the team's "cost-saver" policy instead of blocking
5. Pick candidates from the policy table and drop any that violate quality, latency, context or privacy limits
6. Call the provider, falling back on error or timeout to the next candidate
7. Log the decision with an explanation, e.g. *"coding/high → gpt-4o-mini rejected (quality 6.1 < 7.5); chose gemini-pro; est. ₹0.42 vs ₹1.10 baseline, saved 62%"*

**5. Budget, forecasting and anomaly (O6, Safety)**
- Daily spend per team. Holt-Winters for weekly seasonality, with ARIMA as a comparison. Month-end projection with a confidence interval and a predicted overrun date. Report MAPE on held-out days.
- Anomaly detection: rolling z-score or IsolationForest on per-user hourly tokens and spend, which raises an alert.

**6. What-if analysis (O6, Econ)**
Replay the last N days of logged requests (task type, tokens) under any candidate policy, a price change or a volume growth %. Compare cost, quality and latency against the current setup and the all-premium baseline before deploying.

## Security (report pp. 5–6)
- JWT access and refresh tokens, argon2 password hashing, RBAC enforced in a FastAPI dependency. Members see only their own usage; managers see their team; org admins see everything. Tests cover the report's "access another team's spend" example.
- App API keys stored as SHA-256 with a prefix and shown only once. Provider keys are Fernet-encrypted with the key from env. No secrets in the repo (`.env.example` only).
- Pydantic validation, maximum prompt size, Redis rate limiting per key and user, CORS allow-list, HTTPS behind a reverse proxy, parameterized ORM queries.
- Privacy: prompt hashing or opt-in storage, a PII redaction pass (regex for emails, phones, Aadhaar-like IDs) before logging, and a data-retention setting.
- `audit_events` for access to financial data.

## React dashboard pages
Login · Overview (spend, savings, requests, cache hit %, forecast vs budget) · Teams and Budgets (CRUD, thresholds) · Models and Pricing (registry, price edits, quality/latency profiles heatmap) · Optimization (run NSGA-II, Pareto chart, pick and apply a policy) · Forecast (chart with CI, overrun warnings) · What-if simulator · Request log with explanations · Alerts · Playground/chat (send a prompt, see the routing explanation live) · API keys.

## Build phases (any order is allowed; this order reduces rework)
1. **Foundation:** Docker Compose, FastAPI skeleton, config, Postgres and Alembic schema, auth and RBAC, seed org/teams/users, CI.
2. **Provider and model layer:** adapter interface plus 4 providers, model registry, versioned pricing, cost calculator, fallback executor (port the idea from `backend/executor.py`).
3. **Gateway and logging:** `/v1/chat/completions` and an internal `/route`, request and decision logging, exact cache, rate limits.
4. **Workload analysis ML:** build the dataset, train and evaluate the classifier, complexity model and token estimator. Save models plus evaluation reports in `ml/eval/`.
5. **Profiling:** quality evaluation harness (LLM-as-judge plus exact checks), latency tracker, `model_performance`.
6. **Optimizer:** NSGA-II problem, ILP allocator, policy generation and apply, constraint filter, explainer, then wire into the router.
7. **Budget and forecasting:** enforcer with policy downgrade, forecaster, anomaly detector, alerts, workers.
8. **What-if simulator and savings report.**
9. **Semantic cache** (pgvector).
10. **React frontend:** all pages above.
11. **Evaluation and docs:** full benchmark, plots, architecture/API docs for the final report.

## Evaluation the final report needs (and how we verify)
- **Benchmark:** at least 500 held-out prompts across all task types. Compare (a) all-premium, (b) all-cheap, (c) random, (d) keyword router (old prototype), (e) BudgetOpt policies. Metrics: total cost, **cost reduction %**, judge quality (**% quality retained** vs premium), p95 latency, plus the cost–quality curve (as in the RouteLLM and FrugalGPT papers).
- **ML metrics:** classifier accuracy and macro-F1, token-estimator MAE, forecast MAPE, NSGA-II hypervolume vs the ILP optimum.
- **System checks:** pytest unit tests (cost math, constraint filter, RBAC, budget states, cache), API integration tests against a test Postgres in Docker, a load test with Locust for routing overhead p95 under 50 ms, and Playwright smoke tests for the React flows.
- **End-to-end manual check:** run `docker compose up`, log in as admin, create a team with a small budget, send prompts from the playground, then check the routing explanation, the spend increase, the soft-alert downgrade, the hard block, an applied NSGA-II policy changing routing, the forecast chart, and a what-if result.

## Prerequisites to confirm during build
- API keys: Groq and Gemini (free tier) at minimum. OpenAI/Anthropic keys are optional; without them those adapters are tested with mocks.
- Docker Desktop installed locally.
- The old prototype is kept in git history (or moved to `legacy/`). It is not deleted without the user's OK.
