## LEX7.1 answer screen: idea_test 24 (GLM z-ai/glm-5.3-flash pinned; judge v2 strict, gpt-oss-120b)

HYBRID and HYBRID+rerank: REUSED records and verdicts (idea_test / lex5_pilot; same reader, pin, settings, judge). Their latencies were measured earlier the same day, not interleaved with LEX7.1.

| arm | answered | judged | correct /24 | fa | en | T1 | T2 | T3 | shown retention | 100% | p50 s | p95 s | LLM tokens/q | fixer tokens/q | $/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 24 | 24 | **12** | 5 | 7 | 6 | 2 | 4 | 0.728 | 14 | 1.62 | 3.69 | 2169 | 0 | 0.00035 |
| HYBRID+rerank | 24 | 24 | **13** | 6 | 7 | 6 | 1 | 6 | 0.828 | 17 | 2.74 | 5.59 | 2215 | 0 | 0.00037 |
| LEX7.1 | 24 | 24 | **13** | 6 | 7 | 7 | 2 | 4 | 0.781 | 15 | 2.30 | 4.38 | 3152 | 945 | 0.00045 |

Gate ratios on the 24 (unchanged thresholds: net >= +3/54 at <= 1.5x tokens and p50/p95 vs EACH control; a 24-question screen, not the gate itself):

| arm | vs | net correct | tokens ratio | p50 ratio | p95 ratio | resources <= 1.5x |
|---|---|---:|---:|---:|---:|---|
| LEX7.1 | HYBRID | +1 | 1.45 | 1.42 | 1.19 | ok |
| LEX7.1 | HYBRID+rerank | +0 | 1.42 | 0.84 | 0.78 | ok |

Stage p50 ms (LEX7.1): {"t_encode_ms": 23.5, "t_channels_ms": 6.9, "t_mmr_ms": 13.5, "t_fix_ms": 680.1, "t_retrieval_ms": 792.4, "t_answer_ms": 1476.9}
Fixer status: {"ok": 16, "none": 7, "fail": 1}

Per question (1 = correct; shown retention):

| id | HYBRID | HYBRID+rerank | LEX7.1 |
|---|---|---|---|
| RPT-fa-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T2-002 | 0 (0.50) | 0 (1.00) | 1 (1.00) |
| RPT-fa-T3-002 | 0 (0.50) | 1 (1.00) | 0 (0.50) |
| RPT-en-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T2-002 | 1 (1.00) | 0 (1.00) | 0 (1.00) |
| RPT-en-T3-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-fa-T1-002 | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| LAW-fa-T2-002 | 0 (0.33) | 0 (0.33) | 0 (0.33) |
| BOOK-fa-T3-002 | 0 (0.50) | 0 (0.50) | 0 (0.50) |
| LAW-en-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-en-T2-002 | 0 (0.60) | 1 (0.80) | 0 (0.60) |
| LAW-en-T3-002 | 0 (0.00) | 0 (0.00) | 0 (0.00) |
| RPT-fa-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T2-001 | 0 (0.00) | 0 (0.50) | 0 (0.00) |
| RPT-fa-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T2-001 | 0 (1.00) | 0 (1.00) | 0 (0.67) |
| RPT-en-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-fa-T2-001 | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| LAW-fa-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| LAW-fa-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-en-T1-001 | 0 (0.00) | 0 (1.00) | 1 (1.00) |
| LAW-en-T2-001 | 1 (0.78) | 0 (0.00) | 1 (0.89) |
| BOOK-en-T3-001 | 0 (0.25) | 1 (0.75) | 0 (0.25) |
