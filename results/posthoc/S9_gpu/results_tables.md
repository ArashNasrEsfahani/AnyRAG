## Accuracy (machine, strict, GLM-5.3-flash; bounds over unresolved judge cells)

| arm | all /90 | T1 /30 | T2 /30 | T3 /30 | fa /45 | en /45 | without closed-book-correct /64 | abstention /6 |
|---|---|---|---|---|---|---|---|---|
| HYBRID_AGENT | 77 | 28 | 24 | 25 | 39 | 38 | 55 | 6 |
| ITER_BM25 | 73 | 27 | 23 | 23 | 34 | 39 | 50 | 6 |
| HYBRID_6000 | 62..63 | 28 | 20..21 | 14 | 35..36 | 27 | 44..45 | 6 |
| HR_6000 | 69..71 | 28 | 23..24 | 18..19 | 36 | 33..35 | 50..51 | 6 |
| LEX5B_6000 | 68..69 | 29 | 23..24 | 16 | 36..37 | 32 | 51..52 | 6 |
| SETR_6000 | 63 | 27 | 18 | 18 | 33 | 30 | 47 | 6 |
| HYBRID_15500 | 68 | 28 | 22 | 18 | 35 | 33 | 49 | 6 |
| HR_15500 | 69..70 | 28 | 21..22 | 20 | 38 | 31..32 | 50..51 | 6 |
| LEX5B_15500 | 67 | 28 | 20 | 19 | 36 | 31 | 50 | 6 |
| SETR_15500 | 66..67 | 28 | 22..23 | 16 | 35 | 31..32 | 47..48 | 6 |

## Cost and time (all 96 questions; latency and GPU time descriptive)

| arm | query LLM tokens / q | x HYBRID_6000 | LLM calls / q | GPU ms / q mean (median) | latency s median (mean) |
|---|---|---|---|---|---|
| HYBRID_AGENT | 9070 | 4.09 | 3.62 | 1512 (1137) | 10.5 (13.0) |
| ITER_BM25 | 9285 | 4.19 | 3.64 | 0 (0) | 8.8 (9.9) |
| HYBRID_6000 | 2216 | 1.00 | 1.00 | 205 (157) | 1.8 (2.8) |
| HR_6000 | 2237 | 1.01 | 1.00 | 1331 (1205) | 3.2 (4.0) |
| LEX5B_6000 | 4511..5131 | 2.04..2.32 | 2.00 | 38 (24) | 4.0 (4.5) |
| SETR_6000 | 7503..8128 | 3.39..3.67 | 2.00 | 294 (164) | 5.5 (6.8) |
| HYBRID_15500 | 5236 | 2.36 | 1.00 | 164 (162) | 2.6 (3.1) |
| HR_15500 | 5287 | 2.39 | 1.00 | 1328 (1187) | 3.6 (4.7) |
| LEX5B_15500 | 7542..8155 | 3.40..3.68 | 2.00 | 38 (21) | 4.0 (4.6) |
| SETR_15500 | 10882 | 4.91 | 2.00 | 158 (158) | 5.8 (6.2) |

## Agent behaviour (test)

- **HYBRID_AGENT**: tool calls/q {'search': 2.27, 'finish': 1.0}; rounds mean 2.50, histogram {'2': 72, '3': 12, '5': 12}; evidence source {'finish': 96}; evidence chars median 2830.0 mean 3123; agent calls/q 2.62; system failures 0; truncated answers 0; test cost $0.1097
- **ITER_BM25**: tool calls/q {'search': 2.41, 'finish': 1.0}; rounds mean 2.54, histogram {'2': 68, '3': 15, '4': 2, '5': 11}; evidence source {'finish': 96}; evidence chars median 2735.0 mean 3014; agent calls/q 2.64; system failures 0; truncated answers 0; test cost $0.1151
- **HYBRID_AGENT GPU**: 218 searches; per search mean Ollama embed 77 ms, reranker 589 ms (50 pairs); devices ['cuda/fp16=True']; CPU fallbacks 0

## Paired: HYBRID_AGENT minus comparison arm, 90 answerable questions

| comparison | net | wins/losses | sign p | 95% CI pp (questions) | 95% CI pp (components) | T1 net | T2 net | T3 net | fa net | en net |
|---|---|---|---|---|---|---|---|---|---|---|
| HYBRID_AGENT vs ITER_BM25 | +4 | 7/3 | 0.344 | [-2.2, 11.1] | [-6.0, 14.9] | +1 | +1 | +2 | +5 | -1 |
| HYBRID_AGENT vs HR_6000 | +6..+8 | 15/8; 15/9; 16/8; 16/9 | 0.152-0.307 | [-3.3, 18.9] | [-5.4, 15.9] | +0 | +0..+1 | +6..+7 | +3 | +3..+5 |
| HYBRID_AGENT vs HR_15500 | +7..+8 | 14/6; 14/7 | 0.115-0.189 | [-2.2, 18.9] | [0.0, 16.0] | +0 | +2..+3 | +5 | +1 | +6..+7 |
| HYBRID_AGENT vs HYBRID_6000 | +14..+15 | 20/5; 20/6 | 0.004-0.009 | [5.6, 26.7] | [3.5, 24.5] | +0 | +3..+4 | +11 | +3..+4 | +11 |
| HYBRID_AGENT vs LEX5B_6000 | +8..+9 | 15/7; 16/7 | 0.093-0.134 | [-1.1, 20.0] | [-3.6, 18.6] | -1 | +0..+1 | +9 | +2..+3 | +6 |

## Paired, without the 26 closed-book-correct questions (64 answerable left)

| comparison | net | wins/losses | sign p | 95% CI pp (questions) | 95% CI pp (components) |
|---|---|---|---|---|---|
| HYBRID_AGENT vs ITER_BM25 | +5 | 7/2 | 0.180 | [-1.6, 17.2] | [-5.9, 20.5] |
| HYBRID_AGENT vs HR_6000 | +4..+5 | 11/6; 11/7 | 0.332-0.481 | [-6.2, 20.3] | [-8.9, 20.4] |
| HYBRID_AGENT vs HR_15500 | +4..+5 | 8/3; 8/4 | 0.227-0.388 | [-4.7, 17.2] | [-2.9, 18.4] |
| HYBRID_AGENT vs HYBRID_6000 | +10..+11 | 14/3; 14/4 | 0.013-0.031 | [3.1, 29.7] | [1.7, 28.6] |
| HYBRID_AGENT vs LEX5B_6000 | +3..+4 | 10/7; 11/7 | 0.481-0.629 | [-7.8, 18.8] | [-6.8, 18.3] |
