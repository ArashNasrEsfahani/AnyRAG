## LEX7.1 step 2: order fixer v2 (200-token cap) on MMR-a1 (chunk packer, 54 answerable dev questions)

| arm | pack | top-24 | fa | en | T1 | T2 | T3 | RPT-fa | T3 all docs /18 | b/w vs rerank | b/w vs HYBRID | b/w vs a1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 0.720 (32) | 0.886 | 0.722 (17) | 0.718 (15) | 0.926 (16) | 0.623 (8) | 0.611 (8) | 0.556 (4) | 12 | 6/13 | 0/0 | 0/4 |
| HYBRID+rerank | 0.806 (35) | 0.906 | 0.877 (20) | 0.736 (15) | 0.981 (17) | 0.748 (9) | 0.690 (9) | 0.889 (7) | 13 | 0/0 | 13/6 | 10/6 |
| a1 RRF(bm25,dense,sparse) | 0.763 (34) | 0.921 | 0.772 (18) | 0.755 (16) | 0.981 (17) | 0.669 (9) | 0.639 (8) | 0.611 (5) | 14 | 6/10 | 4/0 | 0/0 |
| MMR-a1 (recomputed) | 0.767 (34) | 0.926 | 0.790 (18) | 0.744 (16) | 0.981 (17) | 0.663 (9) | 0.657 (8) | 0.667 (5) | 14 | 6/9 | 5/2 | 1/2 |
| O1 orderfix(MMR-a1) | 0.773 (35) | 0.926 | 0.790 (18) | 0.757 (17) | 0.981 (17) | 0.681 (10) | 0.657 (8) | 0.667 (5) | 14 | 6/9 | 6/1 | 2/1 |
| O2 orderfix-v2(MMR-a1) | 0.765 (33) | 0.926 | 0.778 (17) | 0.753 (16) | 0.981 (17) | 0.657 (8) | 0.657 (8) | 0.667 (5) | 14 | 5/9 | 6/2 | 2/2 |

Order fixer v2 calls: {"calls": 54, "answered": 53, "prompt_p50": 851.0, "prompt_max": 887, "completion_p50": 89.0, "completion_max": 200, "tokens_per_q_mean": 953.0377358490566, "ms_p50": 621.4, "ms_p95": 1033.7349999999983, "cost_usd": 0.00394745, "status": {"fail": 4, "ok": 40, "none": 10}, "reason": {"deadline": 1, "None": 50, "truncated": 3}}

Gained/lost: {"gained_vs_rerank": ["BOOK-en-T3-002 (0.17 vs 0.00)", "BOOK-en-T3-003 (1.00 vs 0.67)", "LAW-en-T2-001 (0.89 vs 0.00)", "LAW-en-T2-002 (1.00 vs 0.75)", "RPT-en-T3-003 (0.75 vs 0.50)"], "lost_vs_rerank": ["BOOK-fa-T2-002 (0.33 vs 0.67)", "BOOK-en-T2-001 (0.50 vs 0.75)", "BOOK-en-T2-002 (0.60 vs 0.80)", "BOOK-en-T3-001 (0.25 vs 0.75)", "LAW-fa-T3-002 (0.67 vs 1.00)", "RPT-fa-T2-001 (0.00 vs 0.50)", "RPT-fa-T2-003 (0.00 vs 1.00)", "RPT-fa-T3-002 (0.50 vs 1.00)", "RPT-en-T2-003 (0.50 vs 1.00)"], "gained_vs_O1": [], "lost_vs_O1": ["LAW-fa-T2-003 (0.67 vs 1.00)", "LAW-en-T2-001 (0.89 vs 1.00)"]}

