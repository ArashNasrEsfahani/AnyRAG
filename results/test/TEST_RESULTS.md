# TRUE TEST RESULTS (one-shot, 96 test questions; frozen protocol PROTOCOL.md section 9, code 77924d4)

Machine scores of the pinned judge (gpt-oss-120b @ deepinfra/bf16, strict). `a-b` = bounds from needs_adjudication cells (never counted as 0). 90 answerable + 6 unanswerable (AB = correct abstention).

## 1. Primary gate (reader GLM-5.3-flash)

| check | result |
|---|---|
| accuracy: LEX5B_6000 >= +5/90 vs HYBRID_6000 AND vs HR_6000 | **FAILED** (net vs HYBRID +5..+7, vs HR -3..+0) |
| tokens: LEX5B_6000 <= 2.5x contemporaneous HYBRID_6000 (96 q) | **PASSED** (ratio 2.04-2.32; vs HR 2.02-2.29) |
| superiority (exact sign test, Holm over 2) | vs HYBRID p=0.118 (Holm 0.237), vs HR p=1.000 (Holm 1.000) -> not supported |

- **LEX5B_6000 vs HYBRID_6000** (GLM, 90 answerable): net +6, better 11 / worse 4 (sign p 0.118), 95% bootstrap CI of the net [-2, +14], family-cluster CI [-3, +15]. Better: RPT-fa-T2-002, RPT-en-T2-001, RPT-en-T2-004, RPT-en-T3-004, RPT-en-T3-005, BOOK-fa-T1-005, BOOK-fa-T2-001, BOOK-fa-T2-003, LAW-en-T1-004, LAW-en-T2-004, LAW-en-T2-005. Worse: BOOK-fa-T2-005, BOOK-en-T2-002, LAW-fa-T1-002, LAW-en-T2-002. Unresolved: RPT-fa-T2-003, LAW-fa-T2-002.
- **LEX5B_6000 vs HR_6000** (GLM, 90 answerable): net -1, better 6 / worse 7 (sign p 1.000), 95% bootstrap CI of the net [-8, +7], family-cluster CI [-8, +5]. Better: RPT-en-T2-003, RPT-en-T3-005, BOOK-fa-T1-005, BOOK-fa-T2-003, LAW-fa-T3-004, LAW-en-T2-004. Worse: RPT-fa-T2-003, RPT-en-T3-002, BOOK-fa-T2-005, BOOK-en-T2-002, LAW-en-T3-001, LAW-en-T3-003, LAW-en-T3-004. Unresolved: RPT-en-T3-001, LAW-fa-T2-002, LAW-en-T2-005.
- **SETR_6000 vs LEX5B_6000** (GLM, 90 answerable): net -5, better 4 / worse 10 (sign p 0.180), 95% bootstrap CI of the net [-13, +3], family-cluster CI [-13, +2]. Better: RPT-en-T3-002, BOOK-en-T2-002, BOOK-en-T3-005, LAW-fa-T3-005. Worse: RPT-fa-T2-002, RPT-en-T2-005, RPT-en-T3-005, BOOK-fa-T1-005, BOOK-fa-T2-003, LAW-fa-T2-003, LAW-fa-T2-004, LAW-en-T1-004, LAW-en-T2-001, LAW-en-T2-005. Unresolved: LAW-fa-T2-002.

## 2. Every reader x arm (correct)

| reader | arm | /96 | /90 ans | AB /6 | fa /45 | en /45 | T1 | T2 | T3 | tokens/q | p50 s | p95 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GLM-5.3-flash | HYBRID_6000 | 68-69 | 62-63 | 6 | 35-36 | 27 | 28 | 20-21 | 14 | 2216 | 1.84 | 5.90 |
| GLM-5.3-flash | HR_6000 | 75-77 | 69-71 | 6 | 36 | 33-35 | 28 | 23-24 | 18-19 | 2236 | 3.21 | 6.01 |
| GLM-5.3-flash | LEX5B_6000 | 74-75 | 68-69 | 6 | 36-37 | 32 | 29 | 23-24 | 16 | 4511-5130 | 3.97 | 8.62 |
| GLM-5.3-flash | HYBRID_15500 | 74 | 68 | 6 | 35 | 33 | 28 | 22 | 18 | 5235 | 2.56 | 6.15 |
| GLM-5.3-flash | HR_15500 | 75-76 | 69-70 | 6 | 38 | 31-32 | 28 | 21-22 | 20 | 5286 | 3.60 | 8.78 |
| GLM-5.3-flash | LEX5B_15500 | 73 | 67 | 6 | 36 | 31 | 28 | 20 | 19 | 7541-8154 | 4.04 | 8.47 |
| GLM-5.3-flash | SETR_6000 | 69 | 63 | 6 | 33 | 30 | 27 | 18 | 18 | 7502-8127 | 5.53 | 13.12 |
| GLM-5.3-flash | SETR_15500 | 72-73 | 66-67 | 6 | 35 | 31-32 | 28 | 22-23 | 16 | 10881 | 5.78 | 10.80 |
| gpt-oss-20b | HYBRID_6000 | 55 | 50 | 5 | 24 | 26 | 24 | 15 | 11 | 2080 | 1.04 | 2.23 |
| gpt-oss-20b | HR_6000 | 62-64 | 57-59 | 5 | 25-26 | 32-33 | 26 | 18 | 13-15 | 2103 | 2.13 | 3.97 |
| gpt-oss-20b | LEX5B_6000 | 65-66 | 60-61 | 5 | 31 | 29-30 | 25 | 19 | 16-17 | 4386-5006 | 2.81 | 4.48 |
| gpt-oss-20b | HYBRID_15500 | 61-62 | 56-57 | 5 | 27-28 | 29 | 27 | 16 | 13-14 | 4786 | 1.21 | 2.85 |
| gpt-oss-20b | HR_15500 | 65 | 60 | 5 | 30 | 30 | 27 | 17 | 16 | 4851 | 2.43 | 4.38 |
| gpt-oss-20b | LEX5B_15500 | 63-64 | 58-59 | 5 | 30 | 28-29 | 28 | 18-19 | 12 | 7103-7716 | 3.03 | 5.54 |
| gpt-oss-20b | SETR_6000 | 61 | 56 | 5 | 26 | 30 | 26 | 15 | 15 | 7379-8004 | 4.29 | 9.16 |
| gpt-oss-20b | SETR_15500 | 65 | 60 | 5 | 29 | 31 | 25 | 22 | 13 | 10429 | 4.29 | 7.71 |
| Qwen3-235B-2507 | HYBRID_6000 | 64 | 59 | 5 | 33 | 26 | 27 | 20 | 12 | 3325 | 3.50 | 22.50 |
| Qwen3-235B-2507 | HR_6000 | 64-66 | 59-61 | 5 | 29-30 | 30-31 | 27 | 17 | 15-17 | 3389-3455 | 5.39 | 20.62 |
| Qwen3-235B-2507 | LEX5B_6000 | 69 | 64 | 5 | 34 | 30 | 27 | 19 | 18 | 5613-6233 | 6.24 | 32.16 |
| Qwen3-235B-2507 | HYBRID_15500 | 71-73 | 66-68 | 5 | 35-36 | 31-32 | 29 | 20-21 | 17-18 | 8220 | 4.84 | 33.95 |
| Qwen3-235B-2507 | HR_15500 | 69 | 64 | 5 | 33 | 31 | 27 | 21 | 16 | 8050 | 5.97 | 26.85 |
| Qwen3-235B-2507 | LEX5B_15500 | 75-78 | 70-73 | 5 | 36-37 | 34-36 | 28 | 22-23 | 20-22 | 10521-11134 | 7.03 | 22.64 |
| Qwen3-235B-2507 | SETR_6000 | 70 | 65 | 5 | 32 | 33 | 26 | 22 | 17 | 8705-9330 | 7.46 | 19.01 |
| Qwen3-235B-2507 | SETR_15500 | 74 | 69 | 5 | 36 | 33 | 28 | 24 | 17 | 13830 | 7.88 | 26.61 |
| Llama-3.3-70B | HYBRID_6000 | 63 | 57 | 6 | 32 | 25 | 26 | 19 | 12 | 1976 | 4.69 | 17.87 |
| Llama-3.3-70B | HR_6000 | 64-65 | 58-59 | 6 | 29-30 | 29 | 28 | 15 | 15-16 | 1997 | 6.07 | 16.05 |
| Llama-3.3-70B | LEX5B_6000 | 67 | 61 | 6 | 33 | 28 | 29 | 19 | 13 | 4277-4897 | 7.20 | 17.75 |
| Llama-3.3-70B | HYBRID_15500 | 70 | 64 | 6 | 34 | 30 | 28 | 19 | 17 | 4690 | 5.10 | 12.60 |
| Llama-3.3-70B | HR_15500 | 71-72 | 65-66 | 6 | 32 | 33-34 | 27 | 19 | 19-20 | 4743 | 6.23 | 16.60 |
| Llama-3.3-70B | LEX5B_15500 | 72 | 67 | 5 | 32 | 35 | 28 | 23 | 16 | 6997-7610 | 7.51 | 16.84 |
| Llama-3.3-70B | SETR_6000 | 68-69 | 62-63 | 6 | 34-35 | 28 | 27 | 18-19 | 17 | 7269-7894 | 8.98 | 21.77 |
| Llama-3.3-70B | SETR_15500 | 66 | 60 | 6 | 30 | 30 | 28 | 18 | 14 | 10331 | 8.52 | 16.15 |
| Gemma-3-27B | HYBRID_6000 | 56-57 | 51-52 | 5 | 28-29 | 23 | 23 | 17-18 | 11 | 2068-2112 | 9.68 | 81.82 |
| Gemma-3-27B | HR_6000 | 57 | 51 | 6 | 27 | 24 | 24 | 16 | 11 | 2108-2158 | 12.34 | 86.18 |
| Gemma-3-27B | LEX5B_6000 | 63 | 58 | 5 | 32 | 26 | 26 | 17 | 15 | 4372-4992 | 14.53 | 105.14 |
| Gemma-3-27B | HYBRID_15500 | 60-62 | 55-57 | 5 | 29-30 | 26-27 | 24 | 16-18 | 15 | 4884-5094 | 8.97 | 49.84 |
| Gemma-3-27B | HR_15500 | 58-59 | 52-53 | 6 | 26-27 | 26 | 24 | 14 | 14-15 | 4972 | 9.76 | 88.67 |
| Gemma-3-27B | LEX5B_15500 | 63-64 | 57-58 | 6 | 29-30 | 28 | 26 | 15-16 | 16 | 7201-7917 | 12.61 | 101.24 |
| Gemma-3-27B | SETR_6000 | 58-60 | 53-55 | 5 | 26-27 | 27-28 | 27 | 12-13 | 14-15 | 7365-8034 | 14.10 | 87.21 |
| Gemma-3-27B | SETR_15500 | 58 | 53 | 5 | 27 | 26 | 21 | 16 | 16 | 10524-10695 | 14.03 | 86.45 |
| DeepSeek-V4-flash | HYBRID_6000 | 66-67 | 60-61 | 6 | 35 | 25-26 | 28 | 20 | 12-13 | 2658-3183 | 19.99 | 91.35 |
| DeepSeek-V4-flash | HR_6000 | 67-68 | 61-62 | 6 | 35-36 | 26 | 25 | 23 | 13-14 | 2733-3543 | 19.23 | 102.54 |
| DeepSeek-V4-flash | LEX5B_6000 | 72-73 | 66-67 | 6 | 38 | 28-29 | 28 | 23 | 15-16 | 4958-6427 | 19.68 | 103.16 |
| DeepSeek-V4-flash | HYBRID_15500 | 68-70 | 62-64 | 6 | 36-37 | 26-27 | 26 | 20-21 | 16-17 | 5603-6559 | 20.28 | 72.30 |
| DeepSeek-V4-flash | HR_15500 | 78-79 | 72-73 | 6 | 39-40 | 33 | 28 | 24-25 | 20 | 5650-6589 | 17.12 | 105.76 |
| DeepSeek-V4-flash | LEX5B_15500 | 72 | 66 | 6 | 35 | 31 | 27 | 26 | 13 | 7819-10440 | 16.11 | 124.44 |
| DeepSeek-V4-flash | SETR_6000 | 71 | 65 | 6 | 37 | 28 | 28 | 22 | 15 | 7874-9225 | 20.71 | 151.33 |
| DeepSeek-V4-flash | SETR_15500 | 67-68 | 61-62 | 6 | 32-33 | 29 | 27 | 20-21 | 14 | 11268-12365 | 22.72 | 85.26 |

Tier sizes (answerable): {'T1': 30, 'T2': 30, 'T3': 30}. Tokens = query-LLM tokens per question (selector + reader; non-GLM readers: the shared GLM selector calls + their own answer call); latency of non-GLM readers = the shared pre-answer time of the GLM cell + their own answer call.

## 3. Generality across readers (descriptive; 90 answerable, nets with better/worse)

| reader | LEX5B-HYBRID @6000 | LEX5B-HR @6000 | SETR-LEX5B @6000 | SETR-HYBRID @6000 | LEX5B-HYBRID @15500 | LEX5B-HR @15500 | HYBRID 15500-6000 | HR 15500-6000 |
|---|---|---|---|---|---|---|---|---|
| GLM-5.3-flash | +6 (11/4, p 0.12) | -1 (6/7, p 1.00) | -5 (4/10, p 0.18) | +1 (8/7, p 1.00) | -1 (5/6, p 1.00) | -2 (7/9, p 0.80) | +6 (10/4, p 0.18) | +0 (6/5, p 1.00) |
| gpt-oss-20b | +10 (15/5, p 0.04) | +3 (14/11, p 0.69) | -4 (9/13, p 0.52) | +6 (11/5, p 0.21) | +2 (8/6, p 0.79) | -2 (7/8, p 1.00) | +6 (10/4, p 0.18) | +3 (8/6, p 0.79) |
| Qwen3-235B-2507 | +5 (12/7, p 0.36) | +5 (11/7, p 0.48) | +1 (11/10, p 1.00) | +6 (9/3, p 0.15) | +4 (8/5, p 0.58) | +6 (11/4, p 0.12) | +7 (11/3, p 0.06) | +5 (10/6, p 0.45) |
| Llama-3.3-70B | +4 (12/8, p 0.50) | +3 (13/10, p 0.68) | +1 (12/10, p 0.83) | +5 (10/4, p 0.18) | +3 (11/8, p 0.65) | +2 (11/10, p 1.00) | +7 (12/5, p 0.14) | +7 (11/4, p 0.12) |
| Gemma-3-27B | +7 (13/6, p 0.17) | +7 (17/10, p 0.25) | -5 (9/13, p 0.52) | +2 (10/7, p 0.63) | +2 (10/8, p 0.81) | +5 (15/9, p 0.31) | +4 (9/5, p 0.42) | +1 (12/11, p 1.00) |
| DeepSeek-V4-flash | +6 (8/2, p 0.11) | +5 (10/4, p 0.18) | -1 (4/6, p 0.75) | +5 (8/3, p 0.23) | +4 (9/6, p 0.61) | -6 (4/11, p 0.12) | +2 (9/6, p 0.61) | +11 (11/0, p 0.00) |

## 4. Tokens and latency (GLM cells, 96 questions)

- HYBRID_6000: tokens 212764-212764 (0 unknown-usage calls) = 1.00-1.00x HYBRID_6000; latency p50 1.84 s (1.00x), p95 5.90 s (1.00x)
- HR_6000: tokens 214713-214713 (0 unknown-usage calls) = 1.01-1.01x HYBRID_6000; latency p50 3.21 s (1.74x), p95 6.01 s (1.02x)
- LEX5B_6000: tokens 433071-492573 (15 unknown-usage calls) = 2.04-2.32x HYBRID_6000; latency p50 3.97 s (2.16x), p95 8.62 s (1.46x)
- HYBRID_15500: tokens 502609-502609 (0 unknown-usage calls) = 2.36-2.36x HYBRID_6000; latency p50 2.56 s (1.39x), p95 6.15 s (1.04x)
- HR_15500: tokens 507531-507531 (0 unknown-usage calls) = 2.39-2.39x HYBRID_6000; latency p50 3.60 s (1.96x), p95 8.78 s (1.49x)
- LEX5B_15500: tokens 724005-782871 (15 unknown-usage calls) = 3.40-3.68x HYBRID_6000; latency p50 4.04 s (2.20x), p95 8.47 s (1.44x)
- SETR_6000: tokens 720259-780276 (6 unknown-usage calls) = 3.39-3.67x HYBRID_6000; latency p50 5.53 s (3.01x), p95 13.12 s (2.22x)
- SETR_15500: tokens 1044630-1044630 (0 unknown-usage calls) = 4.91-4.91x HYBRID_6000; latency p50 5.78 s (3.14x), p95 10.80 s (1.83x)

Selector behaviour (GLM cells): LEX5B_6000 n 96, fallbacks 19 {'deadline': 15, 'overflow_alternatives': 2, 'duplicate_alternative': 1, 'id_out_of_range': 1}, all-NONE 6, final pack = fallback pack 37 (same set 56); LEX5B_15500 n 96, fallbacks 17 {'deadline': 15, 'overflow_alternatives': 2}, all-NONE 6, final pack = fallback pack 38 (same set 74); SETR_6000 n 96, fallbacks 6 {'deadline': 6}, all-NONE 0, final pack = fallback pack 19 (same set 19); SETR_15500 n 96, fallbacks 1 {'no_marker': 1}, all-NONE 0, final pack = fallback pack 15 (same set 15)

## 5. Budget effect and LEX-by-budget interaction (GLM)

- HYBRID_15500 vs HYBRID_6000: net +6 (10/4, p 0.180)
- HR_15500 vs HR_6000: net +0 (6/5, p 1.000)
- LEX5B_15500 vs LEX5B_6000: net -1 (7/9, p 0.804)
- SETR_15500 vs SETR_6000: net +3 (7/3, p 0.344)
- LEX5B_15500 vs HYBRID_15500: net -1 (5/6, p 1.000)
- LEX5B_15500 vs HR_15500: net -2 (7/9, p 0.804)
- SETR_15500 vs LEX5B_15500: net -1 (6/6, p 1.000)
- Interaction (LEX5B gain at 6,000 minus gain at 15,500): vs HYBRID +7, vs HR +1 (includes independent selector calls per budget).

## 6. needs_adjudication and system failures

- GLM-5.3-flash HYBRID_6000 V2-test-RPT-fa-T2-003 (needs_adjudication)
- GLM-5.3-flash HR_6000 V2-test-RPT-en-T3-001 (needs_adjudication)
- GLM-5.3-flash HR_6000 V2-test-LAW-en-T2-005 (needs_adjudication)
- GLM-5.3-flash LEX5B_6000 V2-test-LAW-fa-T2-002 (needs_adjudication)
- GLM-5.3-flash HR_15500 V2-test-RPT-en-T2-005 (needs_adjudication)
- GLM-5.3-flash SETR_15500 V2-test-LAW-en-T2-004 (needs_adjudication)
- gpt-oss-20b HR_6000 V2-test-RPT-en-T3-005 (needs_adjudication)
- gpt-oss-20b HR_6000 V2-test-BOOK-fa-T3-004 (needs_adjudication)
- gpt-oss-20b LEX5B_6000 V2-test-RPT-en-T3-005 (needs_adjudication)
- gpt-oss-20b HYBRID_15500 V2-test-BOOK-fa-T3-002 (needs_adjudication)
- gpt-oss-20b LEX5B_15500 V2-test-LAW-en-T2-003 (needs_adjudication)
- Qwen3-235B-2507 HR_6000 V2-test-BOOK-en-T3-004 (needs_adjudication)
- Qwen3-235B-2507 HR_6000 V2-test-LAW-fa-T3-002 (needs_adjudication)
- Qwen3-235B-2507 HYBRID_15500 V2-test-BOOK-en-T3-001 (needs_adjudication)
- Qwen3-235B-2507 HYBRID_15500 V2-test-LAW-fa-T2-002 (needs_adjudication)
- Qwen3-235B-2507 LEX5B_15500 V2-test-RPT-fa-T2-003 (needs_adjudication)
- Qwen3-235B-2507 LEX5B_15500 V2-test-RPT-en-T3-001 (needs_adjudication)
- Qwen3-235B-2507 LEX5B_15500 V2-test-BOOK-en-T3-003 (needs_adjudication)
- Llama-3.3-70B HR_6000 V2-test-LAW-fa-T3-005 (needs_adjudication)
- Llama-3.3-70B HR_15500 V2-test-LAW-en-T3-001 (needs_adjudication)
- Llama-3.3-70B SETR_6000 V2-test-LAW-fa-T2-002 (needs_adjudication)
- Gemma-3-27B HYBRID_6000 V2-test-RPT-fa-T2-003 (needs_adjudication)
- Gemma-3-27B HYBRID_15500 V2-test-LAW-fa-T2-003 (needs_adjudication)
- Gemma-3-27B HYBRID_15500 V2-test-LAW-en-T2-003 (needs_adjudication)
- Gemma-3-27B HR_15500 V2-test-BOOK-fa-T3-004 (needs_adjudication)
- Gemma-3-27B LEX5B_15500 V2-test-LAW-fa-T2-002 (needs_adjudication)
- Gemma-3-27B SETR_6000 V2-test-RPT-en-T3-001 (needs_adjudication)
- Gemma-3-27B SETR_6000 V2-test-LAW-fa-T2-002 (needs_adjudication)
- DeepSeek-V4-flash HYBRID_6000 V2-test-RPT-en-T3-001 (needs_adjudication)
- DeepSeek-V4-flash HR_6000 V2-test-LAW-fa-T3-005 (needs_adjudication)
- DeepSeek-V4-flash LEX5B_6000 V2-test-LAW-en-T3-001 (needs_adjudication)
- DeepSeek-V4-flash HYBRID_15500 V2-test-RPT-en-T3-001 (needs_adjudication)
- DeepSeek-V4-flash HYBRID_15500 V2-test-LAW-fa-T2-002 (needs_adjudication)
- DeepSeek-V4-flash HR_15500 V2-test-LAW-fa-T2-002 (needs_adjudication)
- DeepSeek-V4-flash SETR_15500 V2-test-LAW-fa-T2-002 (needs_adjudication)
- system failures: none

## 7. Run notes (operational)

- Spend (shared reader file: answers + selectors + replays; the judge is ledgered separately): start $10.6269, end $13.1343 -> $2.5074.
- The run was interrupted and resumed: started 2026-10-01 23:09, stopped by the user at about primary question 4/96 (GPU needed elsewhere), resumed 2026-10-04 17:53 on the same frozen commit, completed 2026-10-04. Ollama (bge-m3 query embeddings for HYBRID / HR / SETR) died and was restarted twice (~19:40-19:58 and ~20:38; a watchdog restarted it on failure afterwards). Cells that hit those gaps got their infrastructure rerun. **Latency p50/p95 are therefore measured under interrupted / resumed conditions with GPU contention and two Ollama restarts; they are a descriptive trade-off, not a gate.**
- Infra clean-up (PROTOCOL.md section 10, amendment 2): cells whose every record was an HTTP 429 or Ollama-connection error were re-tried at concurrency 1 with back-off (max 5 tries): 61 tries logged; final outcome per reader {'DeepSeek-V4-flash': {'ok': 61}}. Cells still failed are the system failures of section 6.
- Reruns logged in reruns.jsonl: 123 (of which infra-cleanup tries 61).
- The 14 DeepSeek replay cells never attempted before the 2026-10-04 stop were answered as ordinary replay calls at concurrency 1 (`finish_todo.py`).
- Token bounds: the frozen calibration has no ratio for non-primary reader models; failed (usage not reported) answer calls of those readers get an upper bound from a ratio calibrated on the test's own known-usage replay calls (same method and margin, missing keys only: openrouter/deepseek/deepseek-v4-flash|*|*, openrouter/deepseek/deepseek-v4-flash|answer|*, openrouter/deepseek/deepseek-v4-flash|answer|en, openrouter/deepseek/deepseek-v4-flash|answer|fa, openrouter/google/gemma-3-27b-it|*|*, openrouter/google/gemma-3-27b-it|answer|*, openrouter/google/gemma-3-27b-it|answer|en, openrouter/google/gemma-3-27b-it|answer|fa, openrouter/meta-llama/llama-3.3-70b-instruct|*|*, openrouter/meta-llama/llama-3.3-70b-instruct|answer|*, openrouter/meta-llama/llama-3.3-70b-instruct|answer|en, openrouter/meta-llama/llama-3.3-70b-instruct|answer|fa, openrouter/openai/gpt-oss-20b|answer|*, openrouter/openai/gpt-oss-20b|answer|en, openrouter/openai/gpt-oss-20b|answer|fa, openrouter/qwen/qwen3-235b-a22b-2507|*|*, openrouter/qwen/qwen3-235b-a22b-2507|answer|*, openrouter/qwen/qwen3-235b-a22b-2507|answer|en, openrouter/qwen/qwen3-235b-a22b-2507|answer|fa). Frozen keys and every GLM / gate number are unaffected.

One-shot execution: run-to-run model variance is not estimated. Subgroups, abstention, budget and reader-generality results are descriptive. Human review of the GLM 6,000 discordant answers: `human_check/review_test.html` (key: `human_check/key_test.json`).
