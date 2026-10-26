# Semantic cache — threshold selection

100 curated pairs (50 rewordings, 50 near-misses), embedded with BAAI/bge-small-en-v1.5.

- Similarity of rewordings: median 0.939, min 0.842
- Similarity of near-misses: median 0.780, max 0.947
- Closest near-miss (0.947): "Is coffee good for your health?" vs
  "Is coffee bad for your health?"
- Least similar rewording (0.842): "What is the speed of light?" vs
  "How fast does light travel?"

| Threshold | Correct hits | Wrong hits | Precision | Recall |
|---|---|---|---|---|
| 0.80 | 50 | 19 | 0.725 | 1.00 |
| 0.81 | 50 | 17 | 0.746 | 1.00 |
| 0.82 | 50 | 17 | 0.746 | 1.00 |
| 0.83 | 50 | 13 | 0.794 | 1.00 |
| 0.84 | 50 | 11 | 0.820 | 1.00 |
| 0.85 | 48 | 10 | 0.828 | 0.96 |
| 0.86 | 48 | 9 | 0.842 | 0.96 |
| 0.87 | 48 | 8 | 0.857 | 0.96 |
| 0.88 | 48 | 7 | 0.873 | 0.96 |
| 0.89 | 45 | 5 | 0.900 | 0.90 |
| 0.90 | 41 | 4 | 0.911 | 0.82 |
| 0.91 | 40 | 3 | 0.930 | 0.80 |
| 0.92 | 38 | 2 | 0.950 | 0.76 |
| 0.93 | 34 | 2 | 0.944 | 0.68 |
| 0.94 | 24 | 2 | 0.923 | 0.48 |
| 0.95 | 19 | 0 | 1.000 | 0.38 |
| 0.96 | 10 | 0 | 1.000 | 0.20 |
| 0.97 | 6 | 0 | 1.000 | 0.12 |
| 0.98 | 3 | 0 | 1.000 | 0.06 |
| 0.99 | 0 | 0 | 1.000 | 0.00 |

**Lowest threshold with no wrong hits: 0.95** (recall 38%).
A threshold alone forces a bad trade-off: embeddings put opposites ("good"/"bad") almost as
close as true rewordings, so a safe threshold misses most rewordings. Hence the two-stage
design below. The numbers guard and the task-type allow-list add further protection.

## Two-stage design (deployed)

Accept at similarity >= 0.97; ask the verifier (`gemini-3.5-flash-lite`) when 0.85 <= similarity <
0.97; reject below 0.85.

- Pairs needing the verifier: 52 of 100
- Verifier said YES to 42/42 grey-zone rewordings
  and to 0/10 grey-zone near-misses
- **End-to-end: 48 correct hits, 0 wrong hits; precision 1.000,
  recall 96%** (threshold-only at 0.95: precision 1.000, recall
  38%)
- Wrong hits: none
