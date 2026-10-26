# Routing overhead

BudgetOpt's own added latency per request (the LLM call is replaced by an instant answer;
real PostgreSQL, Redis, classifier, routing and logging). Target from the design: p95 < 50 ms.

| Mode | Scenario | p50 ms | p95 ms | p99 ms | Throughput |
|---|---|---|---|---|---|
| without semantic cache | cache miss, sequential | 9.07 | 11.93 | 16.11 |  |
| without semantic cache | cache miss, 20 concurrent | 81.16 | 131.56 | 145.65 | 226.3 req/s |
| without semantic cache | exact cache hit, sequential | 6.13 | 8.82 | 9.35 |  |
| with semantic cache | cache miss, sequential | 11.69 | 24.54 | 27.09 |  |
| with semantic cache | cache miss, 20 concurrent | 124.07 | 181.56 | 202.9 | 149.3 req/s |
| with semantic cache | exact cache hit, sequential | 6.16 | 7.78 | 8.28 |  |

Workload analysis alone (classifier + complexity + token estimates):
p50 0.86 ms, p95 0.99 ms.
Measured in-process on a development laptop, single API worker.
