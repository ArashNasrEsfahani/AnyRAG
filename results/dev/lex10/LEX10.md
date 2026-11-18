# LEX10: a wider evidence pack instead of an extra LLM call (2026-09-26)

**Idea.** The gates allow up to 1.5x HYBRID's per-question LLM tokens. LEX5 and LEX7 spent that allowance on an extra
call. LEX10 spends it on a wider evidence pack for the same single answer call, using F2 MaxMean order and the
unchanged chunk packer. The 6,000-char pack is a method parameter, not a gate.

**Rules.** Every rule was declared in `PROGRESS.md` before any computation: the budget rule, the arms, the matched
control and admission.

**Spend.** $0. The admission rule was not met, so the answer screen was not run.

**Code and harness.**

- `ragbench/approaches/lex10/tokbudget.py` and `ragbench/tests/test_lex10.py`, commit 67d755c on `upgrades`.
- Harness: `eval10.py` in this folder.

## 1. Pack size from the token rule (declared before any retention was computed)

**Tokenizer.** The GLM reader's tokenizer (glm-5.3-flash) cannot be obtained offline and cannot be verified to match
the served model. The rule therefore uses tiktoken `o200k_base` with a per-language factor.

**Calibration.**

- The factor was fitted on the saved GLM usage of the 24-question screen: 72 answer calls from HYBRID, HR and LEX7.1.
- Every prompt was re-rendered from its record, and all 72 match the ledger `prompt_chars` exactly.

| language | calls | k = GLM tokens / o200k tokens | error of k·o200k (mean abs) | GLM tokens per char | mean completion |
|---|---:|---:|---:|---:|---:|
| fa | 36 | 1.170 | 1.5% | 0.381 | 124.3 |
| en | 36 | 1.015 | 0.9% | 0.230 | 129.6 |

**Prediction.** For each of the 54 questions, predicted tokens = k × o200k(the real rendered prompt) + mean
completion. At 6,000 chars this predicts 2,188 tokens per question, against 2,169 measured for HYBRID on the 24.

| pack chars | predicted tokens/q | × HYBRID | fa | en | items |
|---:|---:|---:|---:|---:|---:|
| 6,000 | 2,188 | 1.01 | 2,689 | 1,687 | 6.6 |
| 7,500 | 2,656 | 1.22 | 3,291 | 2,020 | 8.1 |
| **8,500 = B\*** | **2,974** | **1.37** | **3,694** | **2,254** | **9.3** |
| 9,000 | 3,116 | 1.44 | 3,869 | 2,362 | 9.7 |
| 12,000 | 4,060 | 1.87 | 5,081 | 3,039 | 12.9 |

**B\* = 8,500 chars.** It is the largest 500-step size under the 1.40x cap of 3,037 tokens.

- The gate is on the mean. Persian questions alone would reach about 1.70x HYBRID's mean, because Persian costs 1.66x
  the tokens per char of English.

## 2. Offline retention (54 answerable dev questions)

**Sanity check.** At 6,000 chars, F2 and HYBRID reproduce the saved values exactly (maximum difference 0.0000).

| arm | pack (complete) | fa | en | T1 | T2 | T3 | RPT-fa | complete fa / en | T3 all docs /18 | better/worse vs HR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID @6,000 | 0.7200 (32) | 0.722 | 0.718 | 0.926 | 0.623 | 0.611 | 0.556 | 17 / 15 | 12 | 6/13 |
| **HR @6,000** | **0.8065 (35)** | 0.877 | 0.736 | 0.981 | 0.748 | 0.690 | 0.889 | 20 / 15 | 13 | – |
| F2 @6,000 | 0.7792 (33) | 0.796 | 0.762 | 0.981 | 0.678 | 0.678 | 0.722 | 17 / 16 | 13 | 6/8 |
| HYBRID @8,500 (matched control) | 0.7601 (35) | 0.796 | 0.724 | 0.926 | 0.688 | 0.667 | 0.611 | 20 / 15 | 12 | 7/10 |
| **F2 @8,500 (LEX10)** | **0.8080 (36)** | 0.821 | 0.795 | 0.981 | 0.723 | 0.719 | 0.722 | **19** / 17 | 13 | 8/6 |
| F2 @7,500, diagnostic | 0.7885 (33) | 0.796 | 0.781 | 0.981 | 0.692 | 0.692 | 0.722 | 17 / 16 | 13 | 7/7 |
| F2 @9,000, diagnostic | 0.8080 (36) | 0.821 | 0.795 | 0.981 | 0.723 | 0.719 | 0.722 | 19 / 17 | 13 | 8/6 |
| F2 @12,000, diagnostic (1.87x tokens) | 0.8463 (39) | 0.883 | 0.810 | 0.981 | 0.760 | 0.797 | 0.833 | 22 / 17 | 14 | 10/5 |

Complete bundles by tier (T1 / T2 / T3):

| arm | T1 | T2 | T3 |
|---|---:|---:|---:|
| HR | 17 | 9 | 9 |
| F2 @8,500 | 17 | 10 | 9 |
| HYBRID @8,500 | 16 | 10 | 9 |

**Ordering or pack size?** Both contribute.

- The wider pack alone lifts HYBRID by 0.040 (0.720 → 0.760).
- F2's order adds another 0.048 at the same budget: F2 @8,500 is better on 8 questions and worse on 1 against
  HYBRID @8,500.
- Widening the pack gains F2 +0.029 (0.779 → 0.808). Against HYBRID it is 12 better, 0 worse.

**The selection gap.**

- The channel union holds a complete bundle that fits the budget for 48 questions.
- Out of those 48, the number of complete packs is:
  - F2 @6,000: 33
  - HYBRID @8,500: 35
  - HR: 35
  - F2 @8,500: 36
  - F2 @12,000: 39
- About 9.3 items per pack recover three bundles that were cut at 6.6 items.

**Complete bundles, F2 @8,500 against HR.**

- Only F2 @8,500 completes BOOK-en-T3-003, LAW-fa-T2-003, LAW-en-T2-001 and LAW-en-T2-002.
- Only HR completes RPT-fa-T2-003, RPT-fa-T3-002 and RPT-en-T2-003.
- These are the paraphrase chunks from LEX7.1's diagnosis. F2 ranks them below the 9th item, and only the
  cross-encoder lifts them.

## 3. Admission (declared)

The rule required retention ≥ 0.8065, ≥ 38 complete bundles, and no language with fewer complete bundles than HR.

**F2 @8,500 is not admitted.**

| condition | result | met? |
|---|---|---|
| retention ≥ 0.8065 | 0.8080 | yes, by 0.0015 (less than one question) |
| ≥ 38 complete bundles | 36 | no |
| Persian complete bundles ≥ HR's 20 | 19 | no |
| English complete bundles ≥ HR's 15 | 17 | yes |

**Consequence.** The GLM answer and judge screen on the 24 was not run: $0, no answer rows.

- Measured latency, actual tokens, gate ratios and the distractor check therefore do not exist. We report none of
  them.
- The only resource figure is the prediction above: 1.37x HYBRID tokens.
- Latency would grow with the roughly 40% longer prompt, but it was not measured.

**What the diagnostics suggest (not eligible).** At 12,000 chars F2 would pass the retention bars: 0.846, 39 complete
bundles, fa 22. It would cost about 1.87x HYBRID's tokens, which breaks the 1.5x gate.

## 4. Finding

Spending the token allowance on a wider pack is a better use of it than an extra LLM call. At a predicted 1.37x HYBRID
tokens, F2 @8,500 reaches HR's retention (0.808 against 0.8065) with no cross-encoder and no second call. LEX7.1 reached
only 0.781, at 1.45x.

The gain comes from both a bigger pack and LEX's ordering. The matched-budget HYBRID gets only to 0.760.

It still does not clear the declared bar: 36 complete bundles against 38, and Persian 19 against 20. Persian is also
where the token gate bites: at 1.66x the English tokens per char, Persian's share of the budget buys fewer chars.

## Artifacts

| file | contents |
|---|---|
| `calib_rows.jsonl` | the calibration rows |
| `calib10.json` | the calibration fit |
| `budget_rows.jsonl`, `budget10.json` | predicted tokens per question and pack size, and B\* |
| `rows10.jsonl` | the packs per arm, with F2 and HYBRID top-40 orders |
| `TABLES10.md`, `results10.json` | the tables and results |
| `viz_lex10.json` | per question: retention, complete, packs, feasible bundle and predicted tokens; answer and verdict are null |

Registry rows `LEX10 F2@8500`, `LEX10 HYBRID@8500`, and F2 @7,500 / @9,000 / @12,000 marked DIAGNOSTIC.

---

# LEX10b: the same idea under the amended 2.5x token gate (2026-09-26)

**Protocol amendment (user, 2026-09-26 ~19:20).** The LEX token gate is raised to 2.5x HYBRID's mean per-question LLM
tokens. The latency gate (≤ 1.5x p50 / p95) and the accuracy gate (+3 vs each control) are unchanged.

**Honesty note.** The F2 @12,000 diagnostic above (0.846, 39 complete, 1.87x) was seen before LEX10b was declared, so
LEX10b is post-hoc to it. Its pack size was fixed by the token rule alone, before any new retention was computed
(`b/PROGRESS.md`). Spend: $0.024 (24 answers $0.020, judge $0.005).

## 1. Pack size from the token rule

Same predictor as LEX10 (o200k × per-language factor + mean completion). Cap 2.40 × 2,169 = 5,206 tokens; grid
6,000–30,000 in 500s.

| rule | size | predicted tokens/q | × HYBRID | fa | en |
|---|---:|---:|---:|---:|---:|
| **single B\*\*** | **15,500** | **5,135** | **2.37** | 6,449 | 3,821 |
| per-language | fa 12,000 / en 21,500 | 5,135 | 2.37 | 5,081 | 5,189 |

Under the single B\*\*, Persian questions alone predict 2.97x HYBRID's mean; the gate is on the mean.

## 2. Offline retention (54 answerable dev questions)

| arm | pack (complete) | fa | en | T1 | T2 | T3 | complete fa / en | complete T1 / T2 / T3 | T3 all docs /18 | b/w vs HR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HR @6,000 | 0.8065 (35) | 0.877 | 0.736 | 0.981 | 0.748 | 0.690 | 20 / 15 | 17 / 9 / 9 | 13 | – |
| F2 @8,500 (LEX10) | 0.8080 (36) | 0.821 | 0.795 | 0.981 | 0.723 | 0.719 | 19 / 17 | 17 / 10 / 9 | 13 | 8/6 |
| HYBRID @15,500 (matched control) | 0.8471 (39) | 0.870 | 0.824 | 0.981 | 0.796 | 0.764 | 22 / 17 | 17 / 12 / 10 | 15 | 8/2 |
| **F2 @15,500 (LEX10b, chosen)** | **0.8824 (42)** | 0.901 | 0.864 | 0.981 | 0.841 | 0.825 | **22 / 20** | 17 / 13 / 12 | 15 | 11/3 |
| HYBRID @fa 12,000 / en 21,500 | 0.8457 (40) | 0.833 | 0.858 | 0.981 | 0.794 | 0.761 | 20 / 20 | 17 / 13 / 10 | 15 | 11/4 |
| F2 @fa 12,000 / en 21,500 | 0.8824 (43) | 0.883 | 0.882 | 0.981 | 0.813 | 0.853 | 22 / 21 | 17 / 13 / 13 | 14 | 12/2 |

- **Choice.** The per-language variant tied F2 @15,500 exactly (0.882407 each). The declared rule required *higher*
  retention, so the single B\*\* = 15,500 was chosen.
- **Order vs size.** At the same 15,500, F2 beats HYBRID by 0.035 (7 better, 2 worse), and completes 42 bundles
  against 39.
- **Complete only in F2 @15,500 vs HR:** BOOK-en-T2-001, BOOK-en-T3-003, LAW-fa-T2-002, LAW-fa-T2-003,
  LAW-en-T2-001, LAW-en-T2-002, RPT-fa-T3-003, RPT-en-T3-003. **Only HR:** RPT-fa-T2-003.

**Admission (declared): met.** 0.8824 ≥ 0.8065; 42 ≥ 38; fa 22 ≥ 20; en 20 ≥ 15; predicted 2.37x ≤ 2.5x.

## 3. Answer screen (idea_test 24; the chosen arm only)

- **Setup.** Approach `LEX10` (`ragbench/approaches/lex10/approach.py`): F2 MaxMean over BM25 + bge-m3 dense + sparse,
  top 40 chunks, unchanged chunk packer at 15,500, one answer call.
- **Parity.** The online order matched the offline F2 order exactly on the smoke question.
- **Pinned settings.** GLM glm-5.3-flash with the providers.json pins, and judge v2 strict (gpt-oss-120b).
- **Controls.** HYBRID, HR and LEX7.1 are reused records measured earlier the same day, not interleaved with this
  run. The bge-m3 query encoder ran on cuda, as it did for LEX7.1.

| arm | correct /24 | fa | en | T1 | T2 | T3 | shown retention (100%) | p50 s | p95 s | tokens/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 12 | 5 | 7 | 6 | 2 | 4 | 0.728 (14) | 1.62 | 3.69 | 2,169 |
| HR | 13 | 6 | 7 | 6 | 1 | 6 | 0.828 (17) | 2.74 | 5.59 | 2,215 |
| LEX7.1 | 13 | 6 | 7 | 7 | 2 | 4 | 0.781 (15) | 2.30 | 4.38 | 3,152 |
| **LEX10b** | **15** | 7 | 8 | 7 | 3 | 5 | 0.898 (19) | **1.41** | **2.80** | **5,150** |

| vs | net correct | tokens | p50 | p95 | tokens ≤ 2.5x HYBRID | latency ≤ 1.5x | net ≥ +3 |
|---|---:|---:|---:|---:|---|---|---|
| HYBRID | +3 | 2.37 | 0.87 | 0.76 | ok | ok | yes |
| HR | +2 | 2.32 | 0.51 | 0.50 | ok | ok | no |
| LEX7.1 | +2 | 1.63 | 0.61 | 0.64 | ok | ok | no |

**Tokens.** Actual 5,150 tokens/q (fa 6,433, en 3,868) against 5,144 predicted on the same 24, a mean absolute error
of 2.3%. The declared predictor held.

**Latency.**

- Retrieval takes 26 ms at p50. The answer call takes 1.39 s at p50: the longer prompt costs less time than HR's
  cross-encoder and LEX7.1's second call.
- Latency is below every control, but the controls were measured at an earlier time.

**Distractor check vs HR.**

- 17 questions have full retention in both shown packs.
- On those, LEX10b is never wrong where HR is right.
- It is right where HR is wrong on 3: RPT-fa-T2-002, RPT-en-T2-002 and BOOK-en-T1-001.
- On this screen, the 2.6x wider pack showed no distractor harm.

## 4. Finding

With the 2.5x allowance spent on evidence, F2 @15,500 is the first LEX arm to pass every declared offline bar:

- retention 0.882 against HR's 0.807
- 42 complete bundles
- no language regression

It also leads the 24-question screen: 15 against 12 / 13 / 13, at lower latency than every control. It clears the
+3 bar only against HYBRID (+2 against HR and LEX7.1). The accuracy gate is defined on 54 questions, so this 24-question
screen does not decide it.

Most of the gain is the pack size, not LEX: HYBRID's own order at 15,500 reaches 0.847 / 39. F2's order adds 0.035
on top.

**Caveat.** The arm is post-hoc to the F2 @12,000 diagnostic, and the rule was chosen knowing that bigger packs
help. The next step is a full-54 answer run of LEX10b against HYBRID @15,500 as well as the 6,000-char controls.
HYBRID @15,500 is the fair size-matched control, and it has not been answered yet.

**Artifacts (`b/`).**

| file | contents |
|---|---|
| `PROGRESS.md` | the declarations and the log |
| `eval10b.py` | the offline stages |
| `budget10b.json`, `budget_rows10b.jsonl` | the budget table and per-question predictions |
| `rows10b.jsonl`, `TABLES10b.md`, `results10b.json` | the offline arms and tables |
| `answers10b.py`, `report10b.py` | the answer and report harness |
| `eval/runs`, `eval/judge`, `eval/ANSWER_TABLES10b.md`, `eval/answer_results10b.json` | the answer screen |
| `viz_lex10b.json` | per question: retention, packs, predicted tokens, answers, verdicts |

Registry rows `LEX10b …` (5, with `gate_tokens: 2.5x (user amendment 2026-09-26)`).

## 5. Size-matched control: HYBRID @15,500 on the same 24 (2026-09-26, `b_control/`)

Same as LEX10b except the chunk order: plain HYBRID RRF order (BM25 + dense, top 40), the same chunk packer at
15,500 chars, the same pinned GLM reader and judge. Code `ragbench/approaches/lex10/control.py`. The online order equals the
offline `hybrid_order40` (checked on the smoke question). Spend: $0.024.

| arm | correct /24 | fa | en | T1 | T2 | T3 | shown retention (100%) | p50 s | p95 s | tokens/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID @6,000 | 12 | 5 | 7 | 6 | 2 | 4 | 0.728 (14) | 1.62 | 3.69 | 2,169 |
| **HYBRID @15,500** | **19** | 9 | 10 | 7 | 7 | 5 | 0.878 (18) | 2.68 | 11.26 | 5,152 |
| LEX10b (F2 @15,500) | 15 | 7 | 8 | 7 | 3 | 5 | 0.898 (19) | 1.41 | 2.80 | 5,150 |

**Paired comparison with LEX10b.**

- Four questions differ. The control is right on all four, and all four are T2: LAW-fa-T2-002, BOOK-en-T2-002,
  RPT-en-T2-001 and BOOK-fa-T2-001.
- LEX10b is never right where the control is wrong.
- On all four, both shown packs have the same decisive-span retention (3 complete in both, 1 at 0.80 in both). The
  difference is therefore on the reader side. LEX10b's four answers fail with 2 wrong claims, 1 omission and 1
  unsupported claim.
- The two packs share 12–16 of their 15–16 chunks; the order differs.

**Tokens.** 5,152 per question (2.38x HYBRID): the same as LEX10b.

**Latency.** Not comparable to LEX10b's:

- The two runs were not interleaved.
- The query embedding runs through Ollama here.
- The first two questions (12.3 s and 16.0 s) ran while Ollama was warming up and a parallel run was using the GPU.
  They inflate p95.

**Finding.** The whole LEX10b gain over the 6,000-char controls comes from pack size. At a matched size, plain
HYBRID order scores 19 against F2's 15 on this screen. F2's offline retention edge (0.882 vs 0.847) did not turn into
correct answers. All four differences happen where both packs hold the same decisive evidence, which points to reader
variance or to order and distractor effects, not to missing evidence. On 24 questions, a 4-question gap is within
noise. Still, nothing here supports F2 ordering over HYBRID at 15,500.
