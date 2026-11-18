# LEX9 dev tables (54 answerable dev questions, 6,000-char chunk packer)

| arm | pack (complete) | fa | en | T1 | T2 | T3 | RPT-fa | complete fa / en | T3 all docs /18 | b/w vs HR | b/w vs HYBRID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 0.7200 (32) | 0.7222 (17) | 0.7177 (15) | 0.9259 (16) | 0.6228 (8) | 0.6111 (8) | 0.5556 (4) | 17 / 15 | 12 | 6/13 | 0/0 |
| HR | 0.8065 (35) | 0.8765 (20) | 0.7364 (15) | 0.9815 (17) | 0.7481 (9) | 0.6898 (9) | 0.8889 (7) | 20 / 15 | 13 | 0/0 | 13/6 |
| a1 | 0.7632 (34) | 0.7716 (18) | 0.7547 (16) | 0.9815 (17) | 0.6691 (9) | 0.6389 (8) | 0.6111 (5) | 18 / 16 | 14 | 6/10 | 4/0 |
| F2 recomputed | 0.7792 (33) | 0.7963 (17) | 0.7621 (16) | 0.9815 (17) | 0.6784 (8) | 0.6778 (8) | 0.7222 (5) | 17 / 16 | 13 | 6/8 | 7/1 |
| LEX9 learned scorer (synthetic-question supervision; chunk packer) | 0.7611 (32) | 0.7778 (18) | 0.7444 (14) | 0.9815 (17) | 0.6120 (7) | 0.6898 (8) | 0.6667 (5) | 18 / 14 | 14 | 6/10 | 7/3 |

Gap (budget-feasible complete bundle; lex8 audit Feas): {"u3_feasible": 48, "u4_feasible": 49, "lex9_top40_feasible": 46, "f2_top40_feasible": 46, "lex9_pack_complete_of_u3_feasible": 32, "f2_pack_complete_of_u3_feasible": 33, "hr_pack_complete_of_u3_feasible": 35}

Latency ms p50 / p95 (warm CPU; saved bge-m3 query encodes): channels 6.02 / 15.47; features_incl_channels 14.05 / 37.13; features_only 6.65 / 20.95; model_order 3.85 / 9.75

## Per question

| id | HYBRID | HR | F2 | LEX9 | u3 feasible | LEX9 top-40 feasible |
|---|---:|---:|---:|---:|---|---|
| BOOK-fa-T1-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| BOOK-fa-T1-002 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| BOOK-fa-T1-003 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| BOOK-fa-T2-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| BOOK-fa-T2-002 | 0.00 | 0.67 | 0.33 | 0.00 | True | False |
| BOOK-fa-T2-003 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| BOOK-fa-T3-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| BOOK-fa-T3-002 | 0.50 | 0.50 | 0.50 | 0.50 | False | False |
| BOOK-fa-T3-003 | 0.00 | 0.50 | 0.50 | 0.50 | True | True |
| BOOK-en-T1-001 | 0.00 | 1.00 | 1.00 | 1.00 | True | True |
| BOOK-en-T1-002 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| BOOK-en-T1-003 | 0.67 | 0.67 | 0.67 | 0.67 | False | False |
| BOOK-en-T2-001 | 0.50 | 0.75 | 0.50 | 0.50 | True | True |
| BOOK-en-T2-002 | 0.60 | 0.80 | 0.60 | 0.60 | False | False |
| BOOK-en-T2-003 | 0.00 | 0.00 | 0.00 | 0.00 | False | False |
| BOOK-en-T3-001 | 0.25 | 0.75 | 0.25 | 0.75 | True | True |
| BOOK-en-T3-002 | 0.33 | 0.00 | 0.33 | 0.50 | False | False |
| BOOK-en-T3-003 | 1.00 | 0.67 | 1.00 | 1.00 | True | True |
| LAW-fa-T1-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-fa-T1-002 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-fa-T1-003 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-fa-T2-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-fa-T2-002 | 0.33 | 0.33 | 0.33 | 0.33 | True | True |
| LAW-fa-T2-003 | 1.00 | 0.67 | 0.67 | 1.00 | True | True |
| LAW-fa-T3-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-fa-T3-002 | 0.67 | 1.00 | 0.67 | 0.67 | True | True |
| LAW-fa-T3-003 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-en-T1-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-en-T1-002 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-en-T1-003 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-en-T2-001 | 0.78 | 0.00 | 0.78 | 0.67 | True | True |
| LAW-en-T2-002 | 1.00 | 0.75 | 1.00 | 0.25 | True | True |
| LAW-en-T2-003 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-en-T3-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| LAW-en-T3-002 | 0.00 | 0.00 | 0.00 | 0.25 | False | False |
| LAW-en-T3-003 | 0.00 | 0.00 | 0.20 | 0.00 | True | False |
| RPT-fa-T1-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-fa-T1-002 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-fa-T1-003 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-fa-T2-001 | 0.00 | 0.50 | 0.50 | 0.50 | True | True |
| RPT-fa-T2-002 | 0.50 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-fa-T2-003 | 0.00 | 1.00 | 0.00 | 0.00 | True | True |
| RPT-fa-T3-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-fa-T3-002 | 0.50 | 1.00 | 0.50 | 0.50 | True | True |
| RPT-fa-T3-003 | 0.00 | 0.50 | 0.50 | 0.00 | True | True |
| RPT-en-T1-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-en-T1-002 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-en-T1-003 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-en-T2-001 | 1.00 | 1.00 | 1.00 | 0.67 | True | True |
| RPT-en-T2-002 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-en-T2-003 | 0.50 | 1.00 | 0.50 | 0.50 | True | True |
| RPT-en-T3-001 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-en-T3-002 | 1.00 | 1.00 | 1.00 | 1.00 | True | True |
| RPT-en-T3-003 | 0.75 | 0.50 | 0.75 | 0.75 | True | True |
