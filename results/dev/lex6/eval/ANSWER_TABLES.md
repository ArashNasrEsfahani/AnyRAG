## Answer screen: idea_test 24 (GLM z-ai/glm-5.3-flash pinned together; judge v2 strict, gpt-oss-120b)

HYBRID and HYBRID+rerank: REUSED records and verdicts (idea_test for the 12 new questions, lex5_pilot for the 12 pilot questions; same reader, endpoint, settings and judge). Their latencies were measured earlier the same day (not interleaved with LEX6): the latency ratios carry that caveat.

| arm | answered | judged | correct /24 | fa | en | T1 | T2 | T3 | shown retention | 100% | p50 s | p95 s | LLM tokens/q | router tokens/q | $/q | exact no-change |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 24 | 24 | **12** | 5 | 7 | 6 | 2 | 4 | 0.728 | 14 | 1.62 | 3.69 | 2169 | 0 | 0.00035 | - |
| HYBRID+rerank | 24 | 24 | **13** | 6 | 7 | 6 | 1 | 6 | 0.828 | 17 | 2.74 | 5.59 | 2215 | 0 | 0.00037 | - |
| LEX6 | 24 | 24 | **11** | 5 | 6 | 5 | 1 | 5 | 0.754 | 15 | 1.46 | 2.52 | 2141 | 0 | 0.00036 | 2 |

### Gate ratios on the 24 (unchanged thresholds; screen only: the gate itself is on 54+6)

| arm | vs | net correct | tokens ratio | p50 ratio | p95 ratio | accuracy-path resources (all <= 1.5x) | efficiency-path resources (p50, p95 <= 0.7x, tokens <= 1x) |
|---|---|---:|---:|---:|---:|---|---|
| LEX6 | HYBRID | -1 | 0.99 | 0.90 | 0.68 | ok | FAIL |
| LEX6 | HYBRID+rerank | -2 | 0.97 | 0.53 | 0.45 | ok | ok |

Per question (1 = correct; shown retention):

| id | HYBRID | HYBRID+rerank | LEX6 |
|---|---|---|---|
| RPT-fa-T1-002 | 1 (1.00) | 1 (1.00) | 0 (0.00) |
| RPT-fa-T2-002 | 0 (0.50) | 0 (1.00) | 0 (0.50) |
| RPT-fa-T3-002 | 0 (0.50) | 1 (1.00) | 1 (1.00) |
| RPT-en-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T2-002 | 1 (1.00) | 0 (1.00) | 0 (1.00) |
| RPT-en-T3-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-fa-T1-002 | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| LAW-fa-T2-002 | 0 (0.33) | 0 (0.33) | 0 (0.33) |
| BOOK-fa-T3-002 | 0 (0.50) | 0 (0.50) | 0 (0.50) |
| LAW-en-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-en-T2-002 | 0 (0.60) | 1 (0.80) | 0 (0.60) |
| LAW-en-T3-002 | 0 (0.00) | 0 (0.00) | 0 (0.25) |
| RPT-fa-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T2-001 | 0 (0.00) | 0 (0.50) | 0 (0.00) |
| RPT-fa-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T2-001 | 0 (1.00) | 0 (1.00) | 1 (1.00) |
| RPT-en-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-fa-T2-001 | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| LAW-fa-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| LAW-fa-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-en-T1-001 | 0 (0.00) | 0 (1.00) | 0 (1.00) |
| LAW-en-T2-001 | 1 (0.78) | 0 (0.00) | 0 (0.67) |
| BOOK-en-T3-001 | 0 (0.25) | 1 (0.75) | 0 (0.25) |
