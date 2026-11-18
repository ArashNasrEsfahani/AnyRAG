## LEX5b answer screen (arm S-c): idea_test 24 (GLM z-ai/glm-5.3-flash pinned; judge v2 strict, gpt-oss-120b)

HYBRID, HYBRID+rerank, LEX7.1: REUSED records and verdicts (idea_test / lex5_pilot / lex7/v71; same reader, pin, settings, judge). Their latencies were measured earlier the same day, not interleaved with LEX5B.

| arm | answered | judged | correct /24 | fa | en | T1 | T2 | T3 | shown retention | 100% | p50 s | p95 s | LLM tokens/q | selector/fixer tokens/q | $/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 24 | 24 | **12** | 5 | 7 | 6 | 2 | 4 | 0.728 | 14 | 1.62 | 3.69 | 2169 | 0 | 0.00035 |
| HYBRID+rerank | 24 | 24 | **13** | 6 | 7 | 6 | 1 | 6 | 0.828 | 17 | 2.74 | 5.59 | 2215 | 0 | 0.00037 |
| LEX7.1 | 24 | 24 | **13** | 6 | 7 | 7 | 2 | 4 | 0.781 | 15 | 2.30 | 4.38 | 3152 | 945 | 0.00045 |
| LEX5B | 24 | 24 | **17** | 6 | 11 | 7 | 5 | 5 | 0.852 | 17 | 4.38 | 7.74 | 4315 | 2165 | 0.00069 |

Gate ratios on the 24 (tokens <= 2.5x per the user amendment 2026-09-26 (was 1.5x); p50/p95 <= 1.5x vs EACH control, unchanged; accuracy target +3 vs each control; a 24-question screen, not the gate itself):

| arm | vs | net correct | tokens ratio (<= 2.5) | p50 ratio (<= 1.5) | p95 ratio (<= 1.5) | tokens gate | latency gate |
|---|---|---:|---:|---:|---:|---|---|
| LEX5B | HYBRID | +5 | 1.99 | 2.70 | 2.10 | ok | FAIL |
| LEX5B | HYBRID+rerank | +4 | 1.95 | 1.60 | 1.39 | ok | FAIL |
| LEX5B | LEX7.1 | +4 | 1.37 | 1.90 | 1.77 | ok | FAIL |

Stage p50 ms (LEX5B; selector call p50 1615 / p95 3054 ms): {"t_encode_ms": 20.5, "t_f2_ms": 4.5, "t_package_ms": 19.3, "t_view_ms": 402.6, "t_pack_ms": 1.6, "t_retrieval_ms": 2136.1, "t_answer_ms": 2165.1}
Selector status: {"ok": 19, "none": 0, "all_none": 0, "fail": 5}

Per question (1 = correct; shown retention):

| id | HYBRID | HYBRID+rerank | LEX7.1 | LEX5B |
|---|---|---|---|---|
| RPT-fa-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T2-002 | 0 (0.50) | 0 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T3-002 | 0 (0.50) | 1 (1.00) | 0 (0.50) | 0 (0.50) |
| RPT-en-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T2-002 | 1 (1.00) | 0 (1.00) | 0 (1.00) | 1 (1.00) |
| RPT-en-T3-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-fa-T1-002 | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| LAW-fa-T2-002 | 0 (0.33) | 0 (0.33) | 0 (0.33) | 0 (0.33) |
| BOOK-fa-T3-002 | 0 (0.50) | 0 (0.50) | 0 (0.50) | 0 (0.50) |
| LAW-en-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-en-T2-002 | 0 (0.60) | 1 (0.80) | 0 (0.60) | 1 (0.80) |
| LAW-en-T3-002 | 0 (0.00) | 0 (0.00) | 0 (0.00) | 0 (0.25) |
| RPT-fa-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T2-001 | 0 (0.00) | 0 (0.50) | 0 (0.00) | 0 (0.50) |
| RPT-fa-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T2-001 | 0 (1.00) | 0 (1.00) | 0 (0.67) | 1 (1.00) |
| RPT-en-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-fa-T2-001 | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| LAW-fa-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| LAW-fa-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-en-T1-001 | 0 (0.00) | 0 (1.00) | 1 (1.00) | 1 (1.00) |
| LAW-en-T2-001 | 1 (0.78) | 0 (0.00) | 1 (0.89) | 1 (0.56) |
| BOOK-en-T3-001 | 0 (0.25) | 1 (0.75) | 0 (0.25) | 1 (1.00) |
