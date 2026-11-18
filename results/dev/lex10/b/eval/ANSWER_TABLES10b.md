## LEX10b answer screen: idea_test 24 (GLM z-ai/glm-5.3-flash pinned; judge v2 strict, gpt-oss-120b)

HYBRID, HYBRID+rerank and LEX7.1: REUSED records and verdicts (same reader, pin, settings, judge), measured earlier the same day, not interleaved with LEX10b.

| arm | answered | judged | correct /24 | fa | en | T1 | T2 | T3 | shown retention | 100% | p50 s | p95 s | LLM tokens/q | prompt/q | completion/q | $/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 24 | 24 | **12** | 5 | 7 | 6 | 2 | 4 | 0.728 | 14 | 1.62 | 3.69 | 2169 | 2053 | 116 | 0.00035 |
| HYBRID+rerank | 24 | 24 | **13** | 6 | 7 | 6 | 1 | 6 | 0.828 | 17 | 2.74 | 5.59 | 2215 | 2089 | 126 | 0.00037 |
| LEX7.1 | 24 | 24 | **13** | 6 | 7 | 7 | 2 | 4 | 0.781 | 15 | 2.30 | 4.38 | 3152 | 2922 | 230 | 0.00045 |
| LEX10b | 24 | 24 | **15** | 7 | 8 | 7 | 3 | 5 | 0.898 | 19 | 1.41 | 2.80 | 5150 | 5010 | 140 | 0.00082 |

Gate ratios on the 24 (amended token gate 2.5x HYBRID; latency <= 1.5x p50 / p95 and accuracy +3 vs each control unchanged; a 24-question screen, the accuracy gate itself is on 54):

| vs | net correct | tokens ratio | p50 ratio | p95 ratio | tokens <= 2.5x | latency <= 1.5x | net >= +3 |
|---|---:|---:|---:|---:|---|---|---|
| HYBRID | +3 | 2.37 | 0.87 | 0.76 | ok | ok | yes |
| HYBRID+rerank | +2 | 2.32 | 0.51 | 0.50 | ok | ok | no |
| LEX7.1 | +2 | 1.63 | 0.61 | 0.64 | ok | ok | no |

Actual tokens/q 5150 (fa 6433, en 3868) vs predicted 5144 on the same 24 (mean abs error 2.3%); stage p50 ms {"t_encode_ms": 21.0, "t_channels_ms": 3.9, "t_pack_ms": 0.3, "t_retrieval_ms": 26.3, "t_answer_ms": 1385.8}; encoder device ['cuda']

Distractor check vs HR (both shown packs full retention): 17 questions; LEX10b wrong where HR right: none; LEX10b right where HR wrong: RPT-fa-T2-002, RPT-en-T2-002, BOOK-en-T1-001

Per question (1 = correct; shown retention):

| id | HYBRID | HYBRID+rerank | LEX7.1 | LEX10b |
|---|---|---|---|---|
| RPT-fa-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T2-002 | 0 (0.50) | 0 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T3-002 | 0 (0.50) | 1 (1.00) | 0 (0.50) | 1 (1.00) |
| RPT-en-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T2-002 | 1 (1.00) | 0 (1.00) | 0 (1.00) | 1 (1.00) |
| RPT-en-T3-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-fa-T1-002 | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| LAW-fa-T2-002 | 0 (0.33) | 0 (0.33) | 0 (0.33) | 0 (1.00) |
| BOOK-fa-T3-002 | 0 (0.50) | 0 (0.50) | 0 (0.50) | 0 (0.50) |
| LAW-en-T1-002 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-en-T2-002 | 0 (0.60) | 1 (0.80) | 0 (0.60) | 0 (0.80) |
| LAW-en-T3-002 | 0 (0.00) | 0 (0.00) | 0 (0.00) | 0 (0.25) |
| RPT-fa-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-fa-T2-001 | 0 (0.00) | 0 (0.50) | 0 (0.00) | 0 (0.50) |
| RPT-fa-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| RPT-en-T2-001 | 0 (1.00) | 0 (1.00) | 0 (0.67) | 0 (1.00) |
| RPT-en-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-fa-T2-001 | 0 (1.00) | 0 (1.00) | 0 (1.00) | 0 (1.00) |
| LAW-fa-T1-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| LAW-fa-T3-001 | 1 (1.00) | 1 (1.00) | 1 (1.00) | 1 (1.00) |
| BOOK-en-T1-001 | 0 (0.00) | 0 (1.00) | 1 (1.00) | 1 (1.00) |
| LAW-en-T2-001 | 1 (0.78) | 0 (0.00) | 1 (0.89) | 1 (1.00) |
| BOOK-en-T3-001 | 0 (0.25) | 1 (0.75) | 0 (0.25) | 0 (0.50) |
