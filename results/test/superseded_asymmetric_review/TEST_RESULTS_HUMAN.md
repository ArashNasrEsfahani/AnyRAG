# TRUE TEST RESULTS, HUMAN-ADJUDICATED (GLM-5.3-flash reader, 6,000-character review)

Computed 2026-10-05 by merging the user's blinded review (`human_check/human_test_export.json`: 77 items = 78 (question, arm) cells on 30 questions) into the machine verdicts behind `test_results.json`. Script: per-cell scores from `report_test.load()/score()` (machine totals reproduced exactly before merging). No model calls.

**Rule (user decision 2026-10-05, applied to dev and test alike):** a review note containing "half correct" (any case) counts as WRONG regardless of the button pressed; "unsure" keeps the machine verdict. Final verdict per (reader, arm, budget, question) = human verdict where the user reviewed that (question, arm) for the GLM reader at 6,000; otherwise the machine strict verdict (pinned gpt-oss-120b judge). needs_adjudication cells not covered by the review are reported as bounds `a-b` (0 and 1) and listed in section 6.

**Scope:** only GLM-5.3-flash at 6,000 was human-reviewed, and only on the discordant questions (LEX5B vs HYBRID/HR, SETR vs LEX5B). The 15,500 cells and all five other readers are machine-only (section 5).

## 1. GLM-5.3-flash: machine vs human-adjudicated

| arm | budget | machine /96 | human-adj. /96 | machine /90 | human-adj. /90 | cells reviewed |
|---|---|---:|---:|---:|---:|---:|
| HYBRID | 6,000 | 68-69 | **74** | 62-63 | **68** | 17 |
| HR | 6,000 | 75-77 | **79** | 69-71 | **73** | 16 |
| LEX5B | 6,000 | 74-75 | **82** | 68-69 | **76** | 30 |
| SETR | 6,000 | 69 | **78** | 63 | **72** | 15 |
| HYBRID | 15,500 | 74 | **74** | 68 | **68** | 0 |
| HR | 15,500 | 75-76 | **75-76** | 69-70 | **69-70** | 0 |
| LEX5B | 15,500 | 73 | **73** | 67 | **67** | 0 |
| SETR | 15,500 | 72-73 | **72-73** | 66-67 | **66-67** | 0 |

At 15,500 nothing was reviewed, so those rows equal the machine scores. The review is selective (discordant questions only): it corrects the cells where arms disagree, so at 6,000 the human-adjudicated totals are better estimates of the *differences* between arms than of absolute accuracy; questions where all compared arms agreed keep the machine verdict. Totals at 6,000 are therefore not directly comparable with the unreviewed 15,500 totals.

## 2. Gate re-evaluated on human-adjudicated GLM @6000 (SECONDARY analysis)

The pre-registered primary analysis used the machine judge and FAILED (TEST_RESULTS.md section 1). This re-evaluation applies the same criterion to the human-adjudicated verdicts. It is a secondary analysis and does not replace the primary result.

| check | machine (primary) | human-adjudicated (secondary) |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000, net /90 | +5..+7 (11/4, p 0.118) | **+8** (9/1, p 0.0215, Holm 0.0430) |
| LEX5B_6000 vs HR_6000, net /90 | -3..+0 (6/7, p 1.000) | **+3** (8/5, p 0.5811, Holm 0.5811) |
| accuracy gate (net >= +5 vs both) | FAILED | **FAILED** |
| tokens (<= 2.5x HYBRID) | PASSED (2.04-2.32x) | PASSED (unchanged) |
| superiority (Holm < 0.05) | not supported | vs HYBRID supported; vs HR not supported |

- **LEX5B_6000 vs HYBRID_6000** (human-adjudicated, 90 answerable): net +8, better 9 / worse 1, exact two-sided sign p = 0.0215 (Holm over 2: 0.0430); sign p across every assignment of unresolved cells 0.0215-0.0215.
  - Better: RPT-fa-T2-002, RPT-en-T2-001, RPT-en-T2-004, RPT-en-T3-001, RPT-en-T3-004, BOOK-fa-T2-003, BOOK-en-T3-005, LAW-fa-T3-005, LAW-en-T3-001.
  - Worse: BOOK-en-T2-002.
  - Unresolved: none.
- **LEX5B_6000 vs HR_6000** (human-adjudicated, 90 answerable): net +3, better 8 / worse 5, exact two-sided sign p = 0.5811 (Holm over 2: 0.5811); sign p across every assignment of unresolved cells 0.5811-0.5811.
  - Better: RPT-en-T2-003, RPT-en-T3-001, BOOK-fa-T2-003, BOOK-en-T3-005, LAW-fa-T1-002, LAW-fa-T3-004, LAW-fa-T3-005, LAW-en-T2-002.
  - Worse: RPT-fa-T2-003, RPT-en-T3-002, BOOK-en-T2-002, LAW-en-T3-003, LAW-en-T3-004.
  - Unresolved: none.
- Descriptive: SETR_6000 vs LEX5B_6000 (human-adjudicated): net -4, better 2 / worse 6, sign p 0.2891. Better: RPT-en-T3-002, BOOK-en-T2-002. Worse: RPT-en-T3-001, BOOK-fa-T2-003, BOOK-fa-T2-005, LAW-fa-T1-002, LAW-en-T2-002, LAW-en-T3-001.

## 3. Judge-human agreement on the reviewed cells

Unit = one (question, arm) cell; an item whose identical answer came from two arms counts once per arm. "Judge strict" = machine 0, human 1; "judge lenient" = machine 1, human 0. Cells where the machine verdict was needs_adjudication or the human pressed unsure are not comparable. Note the reviewed cells are the discordant ones, so agreement here is lower than it would be on a random sample.

**TEST** (77 items, 78 cells, 30 questions): agree 49, judge strict (m0,h1) 25, judge lenient (m1,h0) 0, machine needs_adjudication 4, human unsure 0. Agreement on comparable cells 49/74 = 66.2%. Half-correct rule applied to 6 cells (6 where the button said correct).

| arm | agree | judge strict (m0,h1) | judge lenient (m1,h0) | machine needs_adj. | human unsure |
|---|---:|---:|---:|---:|---:|
| HR_6000 | 11 | 3 | 0 | 2 | 0 |
| HYBRID_6000 | 10 | 6 | 0 | 1 | 0 |
| LEX5B_6000 | 22 | 7 | 0 | 1 | 0 |
| SETR_6000 | 6 | 9 | 0 | 0 | 0 |

Non-agreeing cells: T04 LAW-en-T2-005 HR_6000 m=None h=1; T09 LAW-en-T1-004 HYBRID_6000 m=0 h=1; T09 LAW-en-T1-004 SETR_6000 m=0 h=1; T12 BOOK-fa-T1-005 HYBRID_6000 m=0 h=1; T13 LAW-en-T2-001 SETR_6000 m=0 h=1; T14 RPT-en-T3-005 HYBRID_6000 m=0 h=1; T15 LAW-en-T2-005 HYBRID_6000 m=0 h=1; T16 BOOK-fa-T2-005 LEX5B_6000 m=0 h=1; T20 LAW-fa-T2-002 LEX5B_6000 m=None h=1; T24 RPT-en-T2-005 SETR_6000 m=0 h=1; T28 BOOK-en-T3-005 LEX5B_6000 m=0 h=1; T29 RPT-en-T3-005 HR_6000 m=0 h=1; T32 LAW-en-T2-002 LEX5B_6000 m=0 h=1; T33 RPT-en-T3-001 HR_6000 m=None h=0 (half correct); T34 LAW-fa-T1-002 LEX5B_6000 m=0 h=1; T36 LAW-fa-T2-004 SETR_6000 m=0 h=1; T38 BOOK-fa-T2-001 HYBRID_6000 m=0 h=1; T41 RPT-en-T3-001 LEX5B_6000 m=0 h=1; T45 RPT-fa-T2-003 HYBRID_6000 m=None h=0 (half correct); T54 BOOK-fa-T1-005 SETR_6000 m=0 h=1; T55 BOOK-fa-T1-005 HR_6000 m=0 h=1; T56 LAW-en-T2-004 HR_6000 m=0 h=1; T57 RPT-fa-T2-002 SETR_6000 m=0 h=1; T60 LAW-fa-T3-005 LEX5B_6000 m=0 h=1; T61 LAW-fa-T2-003 SETR_6000 m=0 h=1; T68 LAW-en-T2-005 SETR_6000 m=0 h=1; T70 LAW-en-T3-001 LEX5B_6000 m=0 h=1; T72 RPT-en-T3-005 SETR_6000 m=0 h=1; T73 LAW-en-T2-004 HYBRID_6000 m=0 h=1.

**DEV** (46 items, 47 cells, 20 questions): agree 31, judge strict (m0,h1) 14, judge lenient (m1,h0) 1, machine needs_adjudication 0, human unsure 1. Agreement on comparable cells 31/46 = 67.4%. Half-correct rule applied to 5 cells (0 where the button said correct).

| arm | agree | judge strict (m0,h1) | judge lenient (m1,h0) | machine needs_adj. | human unsure |
|---|---:|---:|---:|---:|---:|
| HR@6000 | 7 | 5 | 1 | 0 | 0 |
| HYBRID@6000 | 8 | 5 | 0 | 0 | 1 |
| LEX5B@6000 | 16 | 4 | 0 | 0 | 0 |

Non-agreeing cells: D03 BOOK-en-T1-001 HR@6000 m=0 h=1; D05 RPT-en-T2-001 HYBRID@6000 m=0 h=1; D07 RPT-en-T2-001 HR@6000 m=0 h=1; D10 RPT-en-T2-003 HR@6000 m=0 h=1; D11 LAW-fa-T2-003 LEX5B@6000 m=0 h=1; D14 BOOK-en-T2-002 HYBRID@6000 m=0 h=1; D20 BOOK-en-AB-001 HYBRID@6000 m=0 h=1; D21 LAW-en-T3-001 HYBRID@6000 m=0 h=1; D24 BOOK-en-T3-001 HYBRID@6000 m=0 h=None; D32 BOOK-fa-T1-001 LEX5B@6000 m=0 h=1; D34 RPT-fa-T2-002 HR@6000 m=0 h=1; D35 BOOK-en-T1-001 HYBRID@6000 m=0 h=1; D36 BOOK-en-T2-001 LEX5B@6000 m=0 h=1; D37 LAW-en-T2-002 LEX5B@6000 m=0 h=1; D39 RPT-en-T2-002 HR@6000 m=0 h=1; D43 LAW-fa-T3-002 HR@6000 m=1 h=0 (half correct).

## 4. Dev (60 questions, 54 answerable; GLM reader, 6,000) recomputed with the same rule

| arm | machine /60 | human-adj. /60 | machine /54 | human-adj. /54 | cells reviewed |
|---|---:|---:|---:|---:|---:|
| HYBRID | 35 | **40** | 30 | **34** | 14 |
| HR | 38 | **42** | 32 | **36** | 13 |
| LEX5B | 43 | **47** | 37 | **41** | 20 |

- LEX5B vs HYBRID (54 answerable): machine net +7 (10/3, p 0.092); human-adjudicated net **+7** (7/0, p 0.016).
- LEX5B vs HR (54 answerable): machine net +5 (9/4, p 0.267); human-adjudicated net **+5** (6/1, p 0.125).

On dev every half-correct note had already been pressed WRONG, so the rule changes nothing there relative to the buttons; the one "unsure" keeps its machine verdict.

## 5. Other readers: machine only

gpt-oss-20b, Qwen3-235B-2507, Llama-3.3-70B, Gemma-3-27B and DeepSeek-V4-flash were not human-reviewed; their numbers in TEST_RESULTS.md (sections 2 and 3) are machine-judge scores and are unchanged. GLM at 15,500 is likewise machine only. Merged totals for every reader x arm are in `test_results_human.json` (`cells`); outside GLM @6000 they equal the machine totals.

## 6. needs_adjudication cells

Of the 35 machine needs_adjudication cells, 4 were resolved by the review:

- GLM-5.3-flash HYBRID_6000 RPT-fa-T2-003 -> 0 (human_half_correct_rule)
- GLM-5.3-flash HR_6000 RPT-en-T3-001 -> 0 (human_half_correct_rule)
- GLM-5.3-flash HR_6000 LAW-en-T2-005 -> 1 (human_correct)
- GLM-5.3-flash LEX5B_6000 LAW-fa-T2-002 -> 1 (human_correct)

31 remain unresolved and are reported as bounds (0 and 1):

- GLM-5.3-flash HR_15500 RPT-en-T2-005 (needs_adjudication)
- GLM-5.3-flash SETR_15500 LAW-en-T2-004 (needs_adjudication)
- gpt-oss-20b HR_6000 RPT-en-T3-005 (needs_adjudication)
- gpt-oss-20b HR_6000 BOOK-fa-T3-004 (needs_adjudication)
- gpt-oss-20b LEX5B_6000 RPT-en-T3-005 (needs_adjudication)
- gpt-oss-20b HYBRID_15500 BOOK-fa-T3-002 (needs_adjudication)
- gpt-oss-20b LEX5B_15500 LAW-en-T2-003 (needs_adjudication)
- Qwen3-235B-2507 HR_6000 BOOK-en-T3-004 (needs_adjudication)
- Qwen3-235B-2507 HR_6000 LAW-fa-T3-002 (needs_adjudication)
- Qwen3-235B-2507 HYBRID_15500 BOOK-en-T3-001 (needs_adjudication)
- Qwen3-235B-2507 HYBRID_15500 LAW-fa-T2-002 (needs_adjudication)
- Qwen3-235B-2507 LEX5B_15500 RPT-fa-T2-003 (needs_adjudication)
- Qwen3-235B-2507 LEX5B_15500 RPT-en-T3-001 (needs_adjudication)
- Qwen3-235B-2507 LEX5B_15500 BOOK-en-T3-003 (needs_adjudication)
- Llama-3.3-70B HR_6000 LAW-fa-T3-005 (needs_adjudication)
- Llama-3.3-70B HR_15500 LAW-en-T3-001 (needs_adjudication)
- Llama-3.3-70B SETR_6000 LAW-fa-T2-002 (needs_adjudication)
- Gemma-3-27B HYBRID_6000 RPT-fa-T2-003 (needs_adjudication)
- Gemma-3-27B HYBRID_15500 LAW-fa-T2-003 (needs_adjudication)
- Gemma-3-27B HYBRID_15500 LAW-en-T2-003 (needs_adjudication)
- Gemma-3-27B HR_15500 BOOK-fa-T3-004 (needs_adjudication)
- Gemma-3-27B LEX5B_15500 LAW-fa-T2-002 (needs_adjudication)
- Gemma-3-27B SETR_6000 RPT-en-T3-001 (needs_adjudication)
- Gemma-3-27B SETR_6000 LAW-fa-T2-002 (needs_adjudication)
- DeepSeek-V4-flash HYBRID_6000 RPT-en-T3-001 (needs_adjudication)
- DeepSeek-V4-flash HR_6000 LAW-fa-T3-005 (needs_adjudication)
- DeepSeek-V4-flash LEX5B_6000 LAW-en-T3-001 (needs_adjudication)
- DeepSeek-V4-flash HYBRID_15500 RPT-en-T3-001 (needs_adjudication)
- DeepSeek-V4-flash HYBRID_15500 LAW-fa-T2-002 (needs_adjudication)
- DeepSeek-V4-flash HR_15500 LAW-fa-T2-002 (needs_adjudication)
- DeepSeek-V4-flash SETR_15500 LAW-fa-T2-002 (needs_adjudication)
