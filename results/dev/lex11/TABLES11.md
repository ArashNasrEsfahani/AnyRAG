# LEX11 offline tables

## Offline arms (54 answerable dev; decisive-span retention; cells: mean (complete))

| arm | budget | retention (complete) | fa | en | T1 | T2 | T3 | b/w vs HYBRID same budget | b/w vs F2 same budget | mean chars | mean items |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID@6000 | 6000 | 0.7200 (32) | 0.722 (17) | 0.718 (15) | 0.926 (16) | 0.623 (8) | 0.611 (8) | 0/0 | 1/7 | 5955 | 6.56 |
| F2@6000 | 6000 | 0.7792 (33) | 0.796 (17) | 0.762 (16) | 0.982 (17) | 0.678 (8) | 0.678 (8) | 7/1 | 0/0 | 5963 | 6.56 |
| LEX11-M@6000 | 6000 | 0.7792 (33) | 0.796 (17) | 0.762 (16) | 0.982 (17) | 0.678 (8) | 0.678 (8) | 7/1 | 0/0 | 5963 | 6.56 |
| F2-2ch@6000 | 6000 | 0.7206 (32) | 0.716 (17) | 0.725 (15) | 0.870 (15) | 0.669 (9) | 0.622 (8) | 3/1 | 1/5 | 5966 | 6.57 |
| F2-intact@6000 | 6000 | 0.7693 (32) | 0.796 (17) | 0.742 (15) | 0.982 (17) | 0.651 (7) | 0.676 (8) | 7/2 | 1/2 | 5777 | 5.98 |
| HYBRID@15500 | 15500 | 0.8471 (39) | 0.870 (22) | 0.824 (17) | 0.982 (17) | 0.796 (12) | 0.764 (10) | 0/0 | 2/7 | 15458 | 16.83 |
| F2@15500 | 15500 | 0.8824 (42) | 0.901 (22) | 0.864 (20) | 0.982 (17) | 0.841 (13) | 0.825 (12) | 7/2 | 0/0 | 15419 | 16.57 |
| LEX11-M@15500 | 15500 | 0.8824 (42) | 0.901 (22) | 0.864 (20) | 0.982 (17) | 0.841 (13) | 0.825 (12) | 7/2 | 0/0 | 15419 | 16.57 |
| F2-2ch@15500 | 15500 | 0.8361 (39) | 0.864 (22) | 0.808 (17) | 0.926 (16) | 0.771 (12) | 0.811 (11) | 4/3 | 0/5 | 15423 | 16.80 |
| F2-intact@15500 | 15500 | 0.8731 (40) | 0.901 (22) | 0.845 (18) | 0.982 (17) | 0.827 (12) | 0.811 (11) | 5/2 | 0/2 | 15283 | 15.61 |
| S-c (saved rule) | 6000 | 0.8540 (38) | 0.870 (20) | 0.838 (18) | 0.982 (17) | 0.771 (10) | 0.809 (11) | 16/3 | 11/2 | 5767 | 5.94 |
| S-c core only | 6000 | 0.7778 (33) | 0.796 (18) | 0.759 (15) | 0.870 (15) | 0.705 (8) | 0.758 (10) | 11/6 | 8/7 | 2736 | 2.78 |
| S-c core+20% | 6000 | 0.8148 (36) | 0.852 (20) | 0.778 (16) | 0.926 (16) | 0.732 (9) | 0.786 (11) | 13/5 | 10/6 | 3560 | 3.67 |

Sanity: {"F2@6000": 0.7792, "F2@15500": 0.8824, "LEX11-M@6000": 0.7792, "LEX11-M@15500": 0.8824, "S-c": 0.854, "F2-intact@6000": 0.7693} vs saved {"F2@6000": 0.7792, "F2@15500": 0.8824, "S-c": 0.854, "F2-intact@6000": 0.7693} -> OK

## LEX11-M@6000 reorder

- reordered questions 51/54; sparse-only items (H=0) per pack mean 0.04 (in 1 questions)
- sparse-only items holding a decisive span: 0 in 0 questions; holding a span NO H>0 item shows: 0 ()
- first decisive item position (mean) F2 1.59 -> LEX11-M 1.78; later in 10, earlier in 5 questions
- H with the HF dense ranks (diagnostic) gives a different order on 4/54

| id | lang | tier | chunk | F2 pos | LEX11-M pos | pack items | spans only here |
|---|---|---|---:|---:|---:|---:|---:|

F2-rescued evidence at 6000 (decisive items of the F2 pack holding a span HYBRID@6000's pack lacks): 8 items in 8 questions; LEX11-M moves 5 later, 1 earlier, 2 unchanged; mean position 4.12 -> 5.38

| id | lang | tier | chunk | F2 pos | LEX11-M pos | pack items | H | spans HYBRID lacks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BOOK-fa-T2-002 | fa | T2 | 1711 | 3 | 5 | 6 | 0.0143 | 1 |
| BOOK-fa-T3-003 | fa | T3 | 3345 | 5 | 5 | 6 | 0.0289 | 1 |
| BOOK-en-T1-001 | en | T1 | 2551 | 5 | 5 | 6 | 0.0243 | 1 |
| BOOK-en-T3-002 | en | T3 | 5757 | 2 | 4 | 6 | 0.0164 | 1 |
| LAW-en-T3-003 | en | T3 | 3901 | 7 | 6 | 7 | 0.0156 | 1 |
| RPT-fa-T2-001 | fa | T2 | 3657 | 4 | 8 | 8 | 0.0128 | 1 |
| RPT-fa-T2-002 | fa | T2 | 3836 | 3 | 5 | 8 | 0.0283 | 1 |
| RPT-fa-T3-003 | fa | T3 | 3870 | 4 | 5 | 7 | 0.0164 | 1 |

## LEX11-M@15500 reorder

- reordered questions 54/54; sparse-only items (H=0) per pack mean 0.26 (in 6 questions)
- sparse-only items holding a decisive span: 0 in 0 questions; holding a span NO H>0 item shows: 0 ()
- first decisive item position (mean) F2 2.00 -> LEX11-M 2.91; later in 13, earlier in 5 questions
- H with the HF dense ranks (diagnostic) gives a different order on 15/54

| id | lang | tier | chunk | F2 pos | LEX11-M pos | pack items | spans only here |
|---|---|---|---:|---:|---:|---:|---:|

F2-rescued evidence at 15500 (decisive items of the F2 pack holding a span HYBRID@15500's pack lacks): 10 items in 9 questions; LEX11-M moves 7 later, 3 earlier, 0 unchanged; mean position 9.8 -> 14.4

| id | lang | tier | chunk | F2 pos | LEX11-M pos | pack items | H | spans HYBRID lacks |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BOOK-fa-T2-002 | fa | T2 | 1711 | 3 | 13 | 17 | 0.0143 | 1 |
| BOOK-en-T2-001 | en | T2 | 5289 | 17 | 16 | 17 | 0.0242 | 1 |
| BOOK-en-T3-001 | en | T3 | 5038 | 15 | 11 | 16 | 0.0147 | 1 |
| LAW-en-T2-001 | en | T2 | 3311 | 10 | 16 | 18 | 0.0248 | 2 |
| LAW-en-T3-002 | en | T3 | 4058 | 8 | 15 | 17 | 0.0149 | 1 |
| LAW-en-T3-003 | en | T3 | 3901 | 7 | 13 | 18 | 0.0156 | 1 |
| LAW-en-T3-003 | en | T3 | 3906 | 14 | 16 | 18 | 0.0143 | 2 |
| RPT-fa-T2-001 | fa | T2 | 3657 | 4 | 16 | 17 | 0.0128 | 1 |
| RPT-fa-T3-003 | fa | T3 | 3870 | 4 | 14 | 17 | 0.0164 | 1 |
| RPT-en-T3-003 | en | T3 | 16304 | 16 | 14 | 17 | 0.0137 | 1 |

## Selector pruning replay (saved S-c outputs, 47 parsed + 7 fallback; 6,000 chars)

| variant | retention (complete) | chars mean (parsed q) | questions losing spans vs saved S-c | spans lost | complete lost / gained | calc+comparison q losing | negation/exception q losing |
|---|---:|---:|---:|---:|---:|---:|---:|
| S-c saved rule (core + full fill) | 0.8540 (38) | 5766 | - | - | - | - | - |
| S-c core only | 0.7778 (33) | 2283 | 9 | 10 | 5 / 0 | 5 | 3 |
| S-c core+20% | 0.8148 (36) | 3230 | 6 | 6 | 2 / 0 | 3 | 3 |

Core chars mean 2283; cores above 80% of the budget: 4

## Representation audit (same F2 order): raw chunks (chunk packer) vs intact lex4 packages

| budget | chunks retention (complete) | intact retention (complete) | chunks chars / items | intact chars / items | b/w intact vs chunks |
|---:|---:|---:|---:|---:|---:|
| 6000 | 0.7792 (33) | 0.7693 (32) | 5963 / 6.56 | 5777 / 5.98 | 1/2 |
| 15500 | 0.8824 (42) | 0.8731 (40) | 15419 / 16.57 | 15283 / 15.61 | 0/2 |
