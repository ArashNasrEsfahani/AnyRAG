# TRUE TEST RESULTS, SYMMETRIC HUMAN ADJUDICATION (GLM-5.3-flash reader, 6,000-character review)

Computed 2026-10-06 by `test_run/human_merge_sym.py` (no model calls). Inputs: first review (`human_check/key_test.json` + `human_test_export.json`, 77 items = 78 cells) and the symmetric completion (`key_test_sym.json` + `human_test_sym_export.json`, 39 items = 40 cells shown, plus 2 cells inherited from a byte-identical reviewed answer of the same question). Machine verdicts from `report_test.load()/score()`; the earlier asymmetric merge (`test_results_human.json`) is reproduced cell-for-cell before the new verdicts are added.

**Rule:** a note containing "half correct" (any case) = WRONG regardless of button; "unsure" keeps the machine verdict. On the 30 selected questions every arm now carries a human verdict; on the other 66 the machine strict verdict stands, and a needs_adjudication cell there would be a bound `a-b`.

**Coverage:** 120 of 120 cells on 30 questions are human-judged (direct_first 76, direct_first_dedup 2, direct_sym 38, direct_sym_dedup 2, inherited_identical_answer 2). Gaps: none. Unsure cells: none. Export items missing: none.

Inherited cells (rule re-applied to the source item): LAW-en-T3-004 HYBRID_6000 <- T74 = 0; LAW-en-T3-004 SETR_6000 <- T74 = 0.

## 1. GLM-5.3-flash @6,000: correct answers

| arm | machine /96 | asym. human (old) /96 | **sym. human (new) /96** | machine /90 | asym. human (old) /90 | **sym. human (new) /90** | human cells old -> new |
|---|---:|---:|---:|---:|---:|---:|---:|
| HYBRID_6000 | 68-69 | 74 | **77** | 62-63 | 68 | **71** | 17 -> 30 |
| HR_6000 | 75-77 | 79 | **82** | 69-71 | 73 | **76** | 16 -> 30 |
| LEX5B_6000 | 74-75 | 82 | **82** | 68-69 | 76 | **76** | 30 -> 30 |
| SETR_6000 | 69 | 78 | **84** | 63 | 72 | **78** | 15 -> 30 |

The 15,500 arms and the five other readers are unchanged from TEST_RESULTS_HUMAN.md (machine only).

## 2. Paired comparisons (90 answerable; exact two-sided sign test)

| comparison | machine net (b/w, p) | asym. human old net (b/w, p) | **sym. human new net (b/w, p, Holm)** |
|---|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +5..+7 (11/4, p 0.1185, range 0.1185-0.3323, Holm 0.2369) | +8 (9/1, p 0.0215, Holm 0.0430) | **+5 (7/2, p 0.1797, Holm 0.3594)** |
| LEX5B_6000 vs HR_6000 | -3..+0 (6/7, p 1.0000, range 0.6072-1.0000, Holm 1.0000) | +3 (8/5, p 0.5811, Holm 0.5811) | **+0 (5/5, p 1.0000, Holm 1.0000)** |
| SETR_6000 vs LEX5B_6000 | -6..-5 (4/10, p 0.1796, range 0.1796-0.3018) | -4 (2/6, p 0.2891) | **+2 (3/1, p 0.6250)** |

Holm is over the two gate comparisons (LEX5B vs HYBRID, LEX5B vs HR); SETR vs LEX5B is descriptive. "range" = sign p over every assignment of unresolved (needs_adjudication) cells; absent when there are none.

- **LEX5B_6000 vs HYBRID_6000** (symmetric): net +5, better 7 / worse 2, p 0.1797, Holm 0.3594 (worst case 0.3594); sign-p over unresolved assignments 0.1797-0.1797.
  - Better: RPT-fa-T2-002, RPT-en-T2-001, RPT-en-T2-004, RPT-en-T3-004, BOOK-fa-T2-003, BOOK-en-T3-005, LAW-fa-T3-005.
  - Worse: BOOK-en-T2-002, LAW-en-T3-003.
  - Unresolved: none.
- **LEX5B_6000 vs HR_6000** (symmetric): net +0, better 5 / worse 5, p 1.0000, Holm 1.0000 (worst case 1.0000); sign-p over unresolved assignments 1.0000-1.0000.
  - Better: RPT-en-T2-003, RPT-en-T3-001, BOOK-fa-T2-003, LAW-fa-T3-004, LAW-fa-T3-005.
  - Worse: RPT-fa-T2-003, RPT-en-T3-002, BOOK-en-T2-002, LAW-en-T3-003, LAW-en-T3-004.
  - Unresolved: none.
- **SETR_6000 vs LEX5B_6000** (symmetric): net +2, better 3 / worse 1, p 0.6250; sign-p over unresolved assignments 0.6250-0.6250.
  - Better: RPT-en-T3-002, BOOK-en-T2-002, LAW-en-T3-003.
  - Worse: BOOK-fa-T2-003.
  - Unresolved: none.

## 3. Pre-registered gate re-checked (SECONDARY analysis)

The pre-registered primary analysis (machine judge) FAILED and stays the primary result. Criterion: LEX5B_6000 net >= +5/90 vs HYBRID_6000 AND vs HR_6000; superiority = Holm-adjusted sign p < 0.05.

| check | machine (primary) | asym. human (old) | **sym. human (new)** |
|---|---|---|---|
| net vs HYBRID_6000 /90 | +5..+7 | +8 | **+5** |
| net vs HR_6000 /90 | -3..+0 | +3 | **+0** |
| accuracy gate | FAILED | FAILED | **FAILED** |
| superiority vs HYBRID_6000 (Holm < 0.05) | no (0.2369) | yes (0.0430) | **no (0.3594)** |
| superiority vs HR_6000 (Holm < 0.05) | no (1.0000) | no (0.5811) | **no (1.0000)** |
| tokens (<= 2.5x HYBRID) | PASSED (2.04-2.32x) | unchanged | unchanged |

## 4. Judge-human agreement on all reviewed cells

Unit = one (question, arm) cell. "Judge strict" = machine 0, human 1; "judge lenient" = machine 1, human 0. needs_adjudication (machine) and unsure (human) cells are not comparable. The first review covered discordant questions only; the symmetric completion covers the remaining arms of the same questions.

**Directly reviewed cells (both reviews)** (118 cells): agree 77, judge strict 37, judge lenient 0, machine needs_adj. 4, human unsure 0; agreement on comparable cells 77/114 = 67.5%.

**Including the inherited cells** (120 cells): agree 79, judge strict 37, judge lenient 0, machine needs_adj. 4, human unsure 0; agreement on comparable cells 79/116 = 68.1%.

**First review only** (78 cells): agree 49, judge strict 25, judge lenient 0, machine needs_adj. 4, human unsure 0; agreement on comparable cells 49/74 = 66.2%.

**Symmetric review only** (40 cells): agree 28, judge strict 12, judge lenient 0, machine needs_adj. 0, human unsure 0; agreement on comparable cells 28/40 = 70.0%.

| arm | agree | judge strict (m0,h1) | judge lenient (m1,h0) | machine needs_adj. | human unsure | agreement |
|---|---:|---:|---:|---:|---:|---:|
| HYBRID_6000 | 20 | 9 | 0 | 1 | 0 | 20/29 = 69.0% |
| HR_6000 | 22 | 6 | 0 | 2 | 0 | 22/28 = 78.6% |
| LEX5B_6000 | 22 | 7 | 0 | 1 | 0 | 22/29 = 75.9% |
| SETR_6000 | 15 | 15 | 0 | 0 | 0 | 15/30 = 50.0% |

(Table includes the 2 inherited cells.) Non-agreeing cells: T04 LAW-en-T2-005 HR_6000 m=None h=1; T09 LAW-en-T1-004 HYBRID_6000 m=0 h=1; T09 LAW-en-T1-004 SETR_6000 m=0 h=1; T12 BOOK-fa-T1-005 HYBRID_6000 m=0 h=1; T13 LAW-en-T2-001 SETR_6000 m=0 h=1; T14 RPT-en-T3-005 HYBRID_6000 m=0 h=1; T15 LAW-en-T2-005 HYBRID_6000 m=0 h=1; T16 BOOK-fa-T2-005 LEX5B_6000 m=0 h=1; T20 LAW-fa-T2-002 LEX5B_6000 m=None h=1; T24 RPT-en-T2-005 SETR_6000 m=0 h=1; T28 BOOK-en-T3-005 LEX5B_6000 m=0 h=1; T29 RPT-en-T3-005 HR_6000 m=0 h=1; T32 LAW-en-T2-002 LEX5B_6000 m=0 h=1; T33 RPT-en-T3-001 HR_6000 m=None h=0 (half correct); T34 LAW-fa-T1-002 LEX5B_6000 m=0 h=1; T36 LAW-fa-T2-004 SETR_6000 m=0 h=1; T38 BOOK-fa-T2-001 HYBRID_6000 m=0 h=1; T41 RPT-en-T3-001 LEX5B_6000 m=0 h=1; T45 RPT-fa-T2-003 HYBRID_6000 m=None h=0 (half correct); T54 BOOK-fa-T1-005 SETR_6000 m=0 h=1; T55 BOOK-fa-T1-005 HR_6000 m=0 h=1; T56 LAW-en-T2-004 HR_6000 m=0 h=1; T57 RPT-fa-T2-002 SETR_6000 m=0 h=1; T60 LAW-fa-T3-005 LEX5B_6000 m=0 h=1; T61 LAW-fa-T2-003 SETR_6000 m=0 h=1; T68 LAW-en-T2-005 SETR_6000 m=0 h=1; T70 LAW-en-T3-001 LEX5B_6000 m=0 h=1; T72 RPT-en-T3-005 SETR_6000 m=0 h=1; T73 LAW-en-T2-004 HYBRID_6000 m=0 h=1; S06 LAW-fa-T1-002 SETR_6000 m=0 h=1; S08 RPT-en-T3-001 HYBRID_6000 m=0 h=1; S13 LAW-en-T3-001 SETR_6000 m=0 h=1; S18 LAW-en-T3-003 HYBRID_6000 m=0 h=1; S19 LAW-en-T2-002 SETR_6000 m=0 h=1; S22 BOOK-fa-T2-005 SETR_6000 m=0 h=1; S30 BOOK-en-T3-005 HR_6000 m=0 h=1; S31 LAW-fa-T1-002 HR_6000 m=0 h=1; S33 RPT-en-T3-001 SETR_6000 m=0 h=1; S35 LAW-en-T2-002 HR_6000 m=0 h=1; S36 LAW-en-T3-003 SETR_6000 m=0 h=1; S38 LAW-en-T3-001 HYBRID_6000 m=0 h=1.

## 5. needs_adjudication cells (four GLM @6,000 arms)

Resolved by human review: HYBRID_6000 RPT-fa-T2-003 -> 0 (human_half_correct_rule); HR_6000 RPT-en-T3-001 -> 0 (human_half_correct_rule); HR_6000 LAW-en-T2-005 -> 1 (human_correct); LEX5B_6000 LAW-fa-T2-002 -> 1 (human_correct).
Still unresolved (bounds): none.

## 6. What changes relative to TEST_RESULTS_HUMAN.md

- **Ranking at 6,000 changes.** With every arm human-judged on the same 30 questions, the /90 order is SETR 78 > HR 76 = LEX5B 76 > HYBRID 71 (old asymmetric: LEX5B 76 > HR 73 > SETR 72 > HYBRID 68). LEX5B is no longer the best GLM @6,000 arm.
- **Withdrawn: "LEX5B beats HYBRID significantly under human adjudication" (+8, 9/1, Holm 0.043).** Symmetric: +5 (7/2), p 0.180, Holm 0.359, not significant. The old margin was an artifact of asymmetric review: LEX5B was reviewed on all 30 questions but the controls only on discordant ones, and the judge errs almost only in the strict direction (37 m0->h1, 0 m1->h0), so reviewing more cells of one arm inflates that arm.
- **Withdrawn: "LEX5B leads HR by +3".** Symmetric: 0 (5/5), p 1.000.
- **Reversed: "SETR is worse than LEX5B" (-4, 2/6).** Symmetric: +2 (3/1), p 0.625, i.e. no detectable difference. The judge under-scored SETR most (15 of 30 SETR cells were m0,h1; agreement 50%).
- **Unchanged:** the pre-registered primary (machine) gate FAILED; the secondary human re-check also FAILS (net vs HYBRID exactly +5 meets that half, net vs HR 0 does not); no superiority claim survives Holm; the token gate still passes; the judge is never lenient on reviewed cells, and overall judge-human agreement stays about two-thirds (79/116 = 68.1%).
- **Caveat:** the 66 unreviewed questions keep machine verdicts for all arms, so absolute totals remain lower bounds of human-judged accuracy where the judge was strict; the comparison is now symmetric only within the 30 selected questions, which were chosen as discordant under the machine judge.
