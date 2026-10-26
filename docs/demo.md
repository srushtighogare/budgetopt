# Demo script for the project review

A 12–15 minute walkthrough that shows every objective from the project report working, in
an order that tells one story: *requests come in → BudgetOpt understands them → routes them
cheaply without losing quality → keeps the budget under control → proves its savings.*

Screenshots of each page are in [`screenshots/`](screenshots/).

## Before the review (10 minutes, once)

```bash
cd budgetopt
docker compose up -d --build                    # PostgreSQL, Redis, API, dashboard
docker compose exec api python -m app.demo_traffic   # 6 weeks of synthetic history
```

Open http://localhost:8080 and sign in as the seeded admin. In the header, switch the team to
**Demo (synthetic traffic)**. Check the Playground answers one prompt (API keys are in `.env`).

Keep a terminal ready for step 3 and the API docs (http://localhost:8000/docs) open in a tab.

## 1. The problem (1 min), Overview page

> "Teams send every request to expensive models. This team's last 30 days would have cost
> **$19.99** on the premium model; BudgetOpt routed it for **$1.59**, 92% less."

Point at: budget meter (91% used, past the 80% alert line), month-end forecast with its
interval and overrun probability, open alerts, spend by model and by task type.

## 2. One request, end to end (3 min), Playground  → *Objectives 2, 3, 4*

Send *"What is the capital of Australia?"*, then *"Write a Python function that checks whether
a string is a palindrome."*

For each, read the **Why this model** box aloud:
- **Workload analysis**: task type with confidence, complexity band, token estimates (about 1 ms).
- **Routing**: the team's optimized policy for that task type and complexity band, and the
  model it drew.
- **Price**: what it cost, what the premium model would have cost, the saving.

Send the first prompt again: **cache hit**, no model call, cost ≈ 0.
Then send a reworded version (*"Which city is Australia's capital?"*): the **semantic cache**
recognises the same meaning.

Open **Request log** and click the request: every candidate model, which were rejected and
why (quality floor, privacy, context window), and the fallback chain.

## 3. Any application can use it (1 min), terminal  → *Architecture item 6*

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="bo_...")   # key from Teams page
r = client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "Hi"}])
print(r.choices[0].message.content, r.model_extra["budgetopt"]["explanation"])
```

> "Existing apps change only their base URL and key."

## 4. Model and pricing layer (1 min), Models page  → *Architecture item 3*

Registry with versioned prices (old request costs stay correct when prices change) and the
**quality heatmap**: each model's measured quality per task type, from a fixed profiling set
graded against gold answers or by an independent judge.

## 5. Multi-objective optimization (3 min), Optimization page  → *Objective 5*

Run **NSGA-II**. Explain the chart: each blue point is a complete allocation (what share of
each kind of request goes to which model); together they form the **Pareto front** of monthly
cost vs quality. Orange diamonds are "no optimization" baselines.

- Click **Cost saver**, **Balanced**, **Quality first**: the traffic-split table changes.
- Point at the note: *"Checked against the exact LP optimum: largest gap 1.0%."*
- Point at the quality floors: a model below 8/10 is never used for hard requests (the
  report's ethics requirement: never cheaper at the cost of quality).
- Click **Apply**, go back to the Playground and send a prompt: the explanation names the
  new policy.

## 6. Budget control and forecasting (2 min), Forecast and Alerts  → *Objective 6*

- Forecast: cumulative spend, dashed projection with its 90% band crossing the budget line,
  and the accuracy check (re-predicting the last 7 days vs a naive average).
- Alerts: budget threshold, forecast overrun, and **anomaly** alerts (one API key suddenly
  made 300 requests in an hour against a usual 4).
- Teams page: at the alert threshold routing switches to the cost-saver policy; at 100% the
  team is blocked before any model is called.

## 7. What-if before changing production (1 min), What-if page  → *Economic feasibility*

Simulate: actual vs cheapest-meeting-the-floor vs +50% traffic. The table shows monthly
cost, budget fit, quality and the saving against premium. Nothing is called; real past
requests are re-priced.

## 8. Security (1 min)  → *Safety & Security section*

Sign in as a team member (create one on the Teams page) and try the Marketing team's spend in
the API docs: **403**, recorded in the audit log, the report's exact example. Mention: argon2
passwords, encrypted provider keys, hashed API keys, login throttling, PII redaction, prompts
stored only as hashes by default. See [`security.md`](security.md).

## 9. Evidence (1 min)

Open [`evaluation.md`](evaluation.md): classifier accuracy (including the dataset-shortcut
finding), optimizer vs exact optimum, forecast benchmark, semantic-cache precision/recall,
routing overhead (p95 ≈ 12–25 ms), and the end-to-end routing benchmark on held-out prompts.

## Likely questions

| Question | Short answer |
|---|---|
| Why NSGA-II and not just the ILP? | The ILP gives one optimum for one objective; NSGA-II returns the whole cost–quality–latency trade-off in one run so a manager can choose. The ILP/LP is used to check NSGA-II (median gap 0.8%). |
| How do you know quality is kept? | Per-task quality profiles with gold answers and an independent judge; hard quality floors per complexity band; the held-out benchmark compares quality against the premium model. |
| Isn't an LLM judge biased? | Judges never grade their own answers, the benchmark judge is outside the evaluated models, math uses gold answers, and the classifier's labels were checked against a second annotator. |
| What if a provider goes down? | Fallback to the next eligible model, and a circuit breaker skips a failing model for 60 s. Both happened for real during development (Gemini 503s). |
| How much latency does BudgetOpt add? | About 12 ms at p95 (25 ms with the semantic cache), measured with real PostgreSQL and Redis. |
| Why is classifier accuracy 62%, not 90%? | Two strong LLMs agree only 72% on these labels, because task types overlap. 62% against a 72% ceiling, top-2 accuracy 80%, and wrong-but-adjacent classes route similarly. |
