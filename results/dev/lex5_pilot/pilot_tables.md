| arm | correct /12 | fa /6 | en /6 | p50 s | p95 s | LLM tokens/q (prompt / completion / of which reasoning) | LLM calls/q | selector ok / all-NONE / fallback | $/q | $ total (+warm-up) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 5 | 1 | 4 | 1.50 | 2.63 | 2143 (2049 / 94 / 21) | 1.0 | - | 0.00035 | 0.0042 (+0.00048) |
| HYBRID+rerank | 7 | 3 | 4 | 2.75 | 6.41 | 2235 (2084 / 150 / 34) | 1.0 | - | 0.00038 | 0.0046 (+0.00096) |
| HYBRID+intact | 3 | 1 | 2 | 2.01 | 3.52 | 2101 (1981 / 120 / 16) | 1.0 | - | 0.00031 | 0.0037 (+0.00036) |
| LEX5 | 8 | 3 | 5 | 3.10 | 6.16 | 9146 (8843 / 302 / 17) | 2.0 | 11 / 0 / 1 | 0.00083 | 0.0099 (+0.00063) |
| LEX5+selector=glm | 6 | 2 | 4 | 2.91 | 5.23 | 7496 (7384 / 112 / 24) | 2.0 | 8 / 0 / 4 | 0.00105 | 0.0126 (+0.00048) |

| arm | answer call tokens/q (p / c / r) | selector call tokens/q (p / c / r) | selector p50 / p95 ms | answer p50 / p95 ms | retrieval p50 ms (incl. selector) | embed calls/q (tokens) | reranker tokens/q non-pad / padded | reranker ms/q | selector statuses |
|---|---|---|---|---|---|---|---|---|---|
| HYBRID | 2049 / 94 / 21 | - | - | 1376 / 2492 | 127 | 1.0 (40) | - | - | - |
| HYBRID+rerank | 2084 / 150 / 34 | - | - | 1762 / 5288 | 678 | 1.0 (40) | 12274 / 14250 | 944 | - |
| HYBRID+intact | 1981 / 120 / 16 | - | - | 1852 / 3227 | 131 | 1.0 (40) | - | - | - |
| LEX5 | 2035 / 120 / 17 | 6808 / 182 / 0 | 1091 / 2075 | 1739 / 4404 | 1248 | 1.0 (40) | - | - | {'ok': 11, 'fail:duplicate_alternative': 1} |
| LEX5+selector=glm | 2008 / 88 / 19 | 5375 / 25 / 5 | 1166 / 3053 | 1287 / 2414 | 1373 | 1.0 (40) | - | - | {'ok': 8, 'fail:overflow_alternatives': 1, 'fail:deadline': 3} |

| comparison | LLM-token ratio | p50 ratio | p95 ratio | resource gate (all <= 1.5x) | net correct /12 |
|---|---:|---:|---:|---|---:|
| LEX5 vs HYBRID | 4.27 | 2.07 | 2.34 | FAIL | +3 |
| LEX5 vs HYBRID+rerank | 4.09 | 1.13 | 0.96 | FAIL | +1 |
| LEX5+selector=glm vs HYBRID | 3.50 | 1.94 | 1.99 | FAIL | +1 |
| LEX5+selector=glm vs HYBRID+rerank | 3.35 | 1.06 | 0.82 | FAIL | -1 |

Selector room under the token gate (tokens/q): LEX5: selector uses 6991, answer 2155; allowed selector <= 1060 vs HYBRID, <= 1197 vs HYBRID+rerank; LEX5+selector=glm: selector uses 5400, answer 2096; allowed selector <= 1118 vs HYBRID, <= 1256 vs HYBRID+rerank

Judge: 42 calls, $0.0102.
