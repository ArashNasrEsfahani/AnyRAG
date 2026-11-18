# S2a "packs": FUSION (Lex-RAG without the selector), GOLD, non-LLM packers, budget sweep

Post-hoc and exploratory. The test split was unsealed before this ran. Nothing was tuned on test: every arm, budget
and parameter below was fixed before its single test run (MMR's λ = 0.7 was set a priori, not searched).
Reader: GLM-5.3-flash (together pin). Judge: the pinned gpt-oss-120b (deepinfra/bf16). Scores are out of the 90
answerable test questions, given as bounds lo–hi over unresolved judge cells.

## 1. What was run, exactly

**Step 0: reproduce the frozen packs** (`verify.py` -> `verify.json`, `verify_rows.jsonl`). The frozen ragbench code
(77924d4) was imported read-only. Packs were rebuilt from the saved traces only:
- HYBRID / HR: `retrieved_units` of each record -> `HybridRetrieval._offer` + `final_pack` or `pack_evidence`.
- LEX5B: `shown_chunks` -> `lex4.core.Packager` over the shared HybridIndex `chunks.jsonl` -> `pack_evidence`.
- All: then `build_answer_messages`.

Result: **576/576 packs match byte-for-byte** (6 arms × 96 questions). Each pack matched the record's
`answer_messages_sha256`, the pack file's `messages_sha256` and messages, and its `num_predict`. The fallback was
also recomputed: `lex5.bundles.hybrid_intact` over `f2_top40` with the exact `Coster` gives the saved
`fallback_chunks` in 96/96 questions at both budgets. Its rendered length equals its exact cost and it is never
clipped (96/96 at both budgets). Other checks (`build_checks.json`):
- `f2_top40` is identical at 6000 and 15500 (96/96).
- The HYBRID and HR top-40 lists are identical across the two budgets (96/96 each).
- At 6000, plain `pack_evidence` over HYBRID's top 40 gives the same pack as the `_offer` path (96/96).

**Arms** (`build.py` -> `packs/<ARM>/<qid>.json`, `packs_index.jsonl`, `packs_index_arm3.jsonl`):
- **FUSION_B**, B ∈ {3000, 6000, 10000, 15500, 24000}. This is the Lex-RAG fallback for every question:
  `hybrid_intact` over the 40 F2 packages in fusion order, whole items, exact cost, never clipped. It is rendered as
  LEX5B renders its shown packages. There is no selector call.
- **HYBRID_B**, B ∈ {3000, 10000, 24000}. `pack_evidence` over the saved HYBRID top 40 (the HYBRID15500 code path).
- **HR_B**, B ∈ {3000, 10000, 24000}. `_offer` + `final_pack` at B over the saved reranked top 40 (the HR15500
  `_Wide` path).
- **GOLD**. The first `evidence_bundle` of each answerable question, one unit item per evidence id (bundle order).
  - The text is `doc.text[start:end]` from the frozen corpus offsets, rendered by `pack_evidence` with no budget limit
    (the largest GOLD pack is 1,950 chars).
  - Check (`gold_check.txt`): for all 210 items the slice equals the annotated quote, and the document's
    `text_sha256` matches. The `note` field in `packs_index.jsonl` says "sha mismatch" for these items. That note is
    wrong: build.py hashed the slice, but `text_sha256` is a hash of the whole document. The packs are unaffected.
  - Not built for the 6 unanswerable questions, which have no evidence bundle.
- **KNAP_6000** (arm 3). Greedy cost-aware knapsack over the 40 F2 packages. Items are taken in order of F2 score
  divided by rendered cost. An item is added whole only if the exact cost of the full list still fits (frozen
  `_try_add`/`canonical` dedupe), with no clipping.
- **MMR_6000** (arm 3). MMR over the 40 F2 packages using the **saved bge-m3 chunk embeddings**
  (`_hybrid/<set>/embeddings.npy`, read on CPU with numpy; every chunk has a vector).
  - relevance = min-max-normalised F2 score; redundancy = maximum cosine to the packages already picked; λ = 0.7.
  - Items are added whole only when the exact cost still fits.
  - This is real embedding MMR, so the lexical-overlap substitute was not needed.

**Is 24000 feasible?** Yes. The candidate text available per question is at least 30,950 chars (FUSION, median
40,171), 30,124 (HYBRID, median 37,474) and 29,386 (HR, median 37,811). No pack was short of its budget for lack of
candidates.

**Answering** (`answer.py`). Each pack's exact messages and `num_predict` go through the frozen `OpenRouterReader`:
the MANIFEST reader pin `together` with its parameter list, 7 HTTP attempts, the same payload rule. The shared
`reader_budget.json` was not touched.
- A cell whose messages are byte-identical to a frozen GLM pack reuses the frozen answer and its frozen verdict:
  - 37 FUSION_6000 cells. These are exactly the questions where LEX5B's shown pack equals its fallback: 19 selector
    fallbacks plus 18 successful selections that gave the same pack.
  - 38 FUSION_15500 cells.
  - 9 MMR_6000 cells.
- A cell identical to another new cell reuses that cell's answer (24 cells).
- So FUSION − LEX5B differences come only from questions where the selector changed the pack.
- 1,230 reader calls, all served by Together, all `finish_reason=stop`, 0 errors.

**Judging** (`judge.py`). The frozen `judge_multi.judge_records` with the frozen judge question file
`test_v2judge_all.jsonl`. Settings: pin deepinfra/bf16, no fallbacks, temperature 0 / top_p 1 / seed 42,
3 HTTP attempts, 3 identical judge attempts, `on_malformed=adjudicate`. It ran in one process per arm
(`judge/<ARM>/`). The 3-question estimate's verdicts are in `judge/` and were seeded into the shards.
- 1,122 judge calls, 0 served-provider mismatches.
- 5 cells need adjudication; they are counted as [0, 1].
- Scoring of new cells uses `report_test.final_verdicts_by_answer`; frozen cells use `report_test.score`.

**Statistics** (`report.py` -> `results.json`, `report_out.txt`). These are the `revision_stats.py` functions
`compare`, `sign_range`, `boot_mean`, `diff_vectors`, `sign_p` and `find`, taken from its source with `ast` and run
unchanged. Importing the module would rewrite the paper's own outputs. They give:
- paired net bounds;
- the exact two-sided sign-test p, as a range over the assignments of unresolved cells;
- 95% bootstrap intervals in percentage points, over questions and over the 20 shared-family components.

## 2. Results

### 2.1 Pipeline × budget (GLM-5.3-flash; accuracy /90 as bounds; 6 unanswerable reported separately)
Tokens are the mean prompt tokens of the **answer call** per question (`ptok`). Persian packs cost about 1.66–1.79×
more tokens per character than English ones. `all` is the mean of all query-time LLM tokens per question (prompt +
completion, selector included). Cells marked * are frozen test cells.

| pipeline | budget | acc /90 | mean chars | ptok en | ptok fa | all tokens/q | calls/q | abstain ok /6 |
|---|---|---|---|---|---|---|---|---|
| HYBRID | 3000 | 49–51 | 2,942 | 857 | 1,416 | 1,239 | 1 | 6 |
| HYBRID | 6000* | 62–63 | 5,956 | 1,531 | 2,655 | 2,216 | 1 | 6 |
| HYBRID | 10000 | 63–64 | 9,951 | 2,457 | 4,265 | 3,482 | 1 | 6 |
| HYBRID | 15500* | 68 | 15,451 | 3,683 | 6,527 | 5,236 | 1 | 6 |
| HYBRID | 24000 | **72** | 23,946 | 5,607 | 9,979 | 7,925 | 1 | 6 |
| HR | 3000 | 56 | 2,937 | 850 | 1,421 | 1,237 | 1 | 6 |
| HR | 6000* | 69–71 | 5,958 | 1,553 | 2,669 | 2,237 | 1 | 6 |
| HR | 10000 | 70 | 9,941 | 2,490 | 4,308 | 3,528 | 1 | 6 |
| HR | 15500* | 69–70 | 15,439 | 3,751 | 6,561 | 5,287 | 1 | 6 |
| HR | 24000 | 66 | 23,935 | 5,673 | 10,024 | 7,985 | 1 | 6 |
| FUSION | 3000 | 43–44 | 2,796 | 822 | 1,354 | 1,190 | 1 | 6 |
| FUSION | 6000 | 58–59 | 5,765 | 1,476 | 2,584 | 2,150 | 1 | 6 |
| FUSION | 10000 | 66 | 9,740 | 2,385 | 4,213 | 3,426 | 1 | 6 |
| FUSION | 15500 | 65 | 15,221 | 3,631 | 6,471 | 5,183 | 1 | 6 |
| FUSION | 24000 | 70 | 23,731 | 5,551 | 9,923 | 7,865 | 1 | 6 |
| LEX5B (Lex-RAG) | 6000* | 68–69 | 5,772 | 1,507 | 2,570 | 4,511 | 2 | 6 |
| LEX5B (Lex-RAG) | 15500* | 67 | 15,214 | 3,629 | 6,463 | 7,542 | 2 | 6 |
| SETR | 6000* | 63 | 5,949 | 1,546 | 2,656 | 7,503 | 2 | 6 |
| SETR | 15500* | 66–67 | 15,450 | 3,688 | 6,528 | 10,882 | 2 | 6 |
| GOLD (first bundle) | none | **83** | 572 | 363 | 396 | 481 | 1 | n/a |
| KNAP (arm 3) | 6000 | 44 | 5,639 | 1,448 | 2,551 | 2,116 | 1 | 6 |
| MMR (arm 3) | 6000 | 60 | 5,788 | 1,490 | 2,588 | 2,158 | 1 | 6 |

### 2.2 The selector call (FUSION = Lex-RAG minus its selector), paired over 90
Columns: net = correct difference /90; p = exact sign test; CIs in percentage points over questions and over the 20
components; w/l = wins/losses at resolved pairs.

| comparison | net | sign p | 95% CI (questions) | 95% CI (components) | w/l |
|---|---|---|---|---|---|
| LEX5B_6000 vs FUSION_6000 | **+9..+11** | 0.007–0.035 | [+2.2, +20.0] | [+3.3, +20.6] | 12/2 |
| LEX5B_15500 vs FUSION_15500 | +2 | 0.79 | [−5.6, +10.0] | [−5.6, +9.5] | 8/6 |
| FUSION_6000 vs HYBRID_6000 | −5..−3 | 0.23–0.51 | [−13.3, +3.3] | [−13.9, +1.4] | 3/6 |
| FUSION_6000 vs HR_6000 | −13..−10 | 0.011–0.041 | [−24.4, −2.2] | [−24.4, −5.4] | 5/15 |
| FUSION_15500 vs HYBRID_15500 | −3 | 0.55 | [−10.0, +3.3] | [−8.8, +2.6] | 4/7 |
| FUSION_15500 vs HR_15500 | −5..−4 | 0.27–0.39 | [−13.3, +3.3] | [−12.2, +3.2] | 4/8 |

At 6000 the selector changed the pack in 59 of 96 questions, and every FUSION–LEX5B difference comes from those.
In the other 37 the pack was identical: 19 selector fallbacks and 18 selections equal to the fallback.

### 2.3 Budget sweep: adjacent steps, and the pipelines at each budget (net /90, sign p)
- HYBRID: 3000→6000 +11..+14 (p ≤ 0.007); 6000→10000 0..+2; 10000→15500 +4..+5 (p 0.18–0.29); 15500→24000 +4
  (p 0.29).
- HR: 3000→6000 +13..+15 (p ≤ 0.002); then flat (−1..+1, −1..0); 15500→24000 −4..−3 (p 0.34–0.55).
- FUSION: 3000→6000 +14..+16 (p ≤ 0.004); 6000→10000 +7..+8 (p 0.06–0.09); 10000→15500 −1; 15500→24000 +5
  (p 0.27).
- At 3000: HR vs HYBRID +5..+7 (p 0.12–0.27); FUSION vs HR −13..−12 (p ≤ 0.008); FUSION vs HYBRID −8..−5
  (p 0.06–0.23).
- At 10000: HR vs HYBRID +6..+7 (p 0.04–0.07); FUSION vs HR −4 (p 0.42); FUSION vs HYBRID +2..+3.
- At 24000: HR vs HYBRID −6 (p 0.21); FUSION vs HR +4 (p 0.45); FUSION vs HYBRID −2.

All bootstrap CIs are in `results.json` and `report_out.txt`.

**Token-matched reading.** Persian answer prompts cost about 1.7× the English ones at the same character budget.
- A Persian 6000-char pack (≈2,570–2,670 tokens) costs about as much as an English 10000-char pack (≈2,390–2,490).
- In total query tokens, LEX5B_6000 (4,511 tokens/q, 2 calls) costs more than FUSION_10000 (3,426) or HR_10000
  (3,528), which reach 66 and 70.

### 2.4 Tier and language (lo–hi correct; 30 per tier, 45 per language)
| arm | T1 | T2 | T3 | en | fa |
|---|---|---|---|---|---|
| HYBRID_6000* | 28 | 20–21 | 14 | 27 | 35–36 |
| HR_6000* | 28 | 23–24 | 18–19 | 33–35 | 36 |
| LEX5B_6000* | 29 | 23–24 | 16 | 32 | 36–37 |
| FUSION_6000 | 29 | 14–15 | 15 | 27 | 31–32 |
| HYBRID_15500* | 28 | 22 | 18 | 33 | 35 |
| HR_15500* | 28 | 21–22 | 20 | 31–32 | 38 |
| LEX5B_15500* | 28 | 20 | 19 | 31 | 36 |
| FUSION_15500 | 26 | 23 | 16 | 31 | 34 |
| HYBRID_24000 | 28 | 25 | 19 | 35 | 37 |
| HR_24000 | 28 | 20 | 18 | 33 | 33 |
| FUSION_24000 | 27 | 24 | 19 | 35 | 35 |
| **GOLD** | 27 | **30** | 26 | 40 | 43 |
| KNAP_6000 | 22 | 9 | 13 | 18 | 26 |
| MMR_6000 | 29 | 17 | 14 | 27 | 33 |

Every arm at every budget is in `results.json` → `groups`.

**Gold ceiling.** With only the first annotated bundle, GLM gets 83/90:
- tiers: T1 27/30, T2 30/30, T3 26/30;
- languages: en 40/45, fa 43/45.

That is 12–14 above the best frozen 6000 cell, HR_6000 at 69–71 (paired GOLD vs LEX5B_6000: +14..+15, p ≤ 0.003; vs HYBRID_15500: +15,
p 0.0015). The 7 misses were not examined. They may be reader errors, judge errors, or a first bundle that does not cover
everything the question needs.

### 2.5 Abstention (6 unanswerable)
Every retrieval arm abstains correctly on 6/6 at every budget, including 24000, KNAP and MMR. GOLD has no pack for
these questions.

## 3. Cost
- Total stream spend: **$0.842** of the $1.50 cap. Source: `ledger.jsonl` (reader) plus `ledger_judge_<ARM>.jsonl`
  (judge shards).
  - Reader: 1,230 GLM calls, $0.695.
  - Judge: 1,122 gpt-oss-120b calls, $0.147.
- The 3-question estimate cost $0.024 for 35 reader calls plus $0.003 for 24 judge calls. Extrapolated, that was
  about $0.75 + $0.14.
- The key's `limit_remaining` was checked before every batch of ≤ 50 calls. It never fell below $7.3, so the $2.00
  floor never triggered.

## 4. Failed or skipped
- **No dev smoke test.** The new arms add no model-side code, only packing over saved traces. The dev smoke was
  replaced by the stronger byte-for-byte reproduction of all 576 frozen packs. The 3-question cost estimate ran on
  test, as the brief asked. No arm was iterated on test.
- **LEX5B (with its selector) and SETR were not swept** to 3000/10000/24000: that needs new selector calls.
- **Reader nondeterminism was not measured.** No identical pack was answered twice. Byte-identical packs reuse the
  frozen answer by design.
- **Arm 3 limitation.** The knapsack's score/char rule favours short packages and is far below fusion order
  (44 vs 58–59). This is a negative result for that rule, not a tuned packer.
- **Deliberately not done.**
  - Human verdicts: machine verdicts only.
  - The 5 needs-adjudication cells are left as [0, 1] bounds.
  - Latency is not reported.
- **No GPU** was used. The MMR embeddings were read from disk.

## 5. Facts for the paper
1. The rebuilt packs match the frozen test packs byte-for-byte: 576/576 messages sha256 (HYBRID/HR/LEX5B at 6000 and
   15500). The Lex-RAG fallback is reproduced exactly in 96/96 questions at both budgets (`verify.json`).
2. Without the selector call, Lex-RAG's fusion pack scores 58–59/90 at 6000 against 68–69 with it: +9..+11, 12 wins /
   2 losses, sign p 0.007–0.035, component CI [+3.3, +20.6] pp. At 15500 the selector adds only +2/90 (65 vs 67,
   p 0.79) (`results.json`, `report_out.txt`).
3. On its own, the fusion order is no better than plain hybrid RRF: FUSION vs HYBRID is −5..−3 at 6000 and −3 at
   15500. It is below the cross-encoder at 6000 (−13..−10, p 0.011–0.041). So, within this post-hoc test, Lex-RAG's 6000-char gain
   over HYBRID is attributable to the selector call, not to the fusion (`results.json`).
4. Over five budgets the curves differ:
   - HYBRID rises 49–51 → 62–63 → 63–64 → 68 → 72.
   - HR is flat at 69–71 from 6000 to 15500 and falls to 66 at 24000.
   - FUSION rises 43–44 → 58–59 → 66 → 65 → 70.

   Persian answer prompts cost ≈1.7× more tokens than English at the same character budget (`results.json` table,
   ptok columns).
5. The annotated first evidence bundle alone (about 572 chars) gives 83/90: T2 30/30, T3 26/30. That is
   about 12–14 questions above the best 6000-char pipeline (HR 69–71) and 11 above the best of any budget (HYBRID_24000,
   72). Most of the remaining gap is in what the packs contain, not in the reader
   (`results.json`).
6. Two non-LLM packers on the same 40 fusion candidates at 6000 do not replace the selector:
   - an embedding-MMR packer scores 60/90 (vs FUSION +1..+2, vs LEX5B −9..−8);
   - a score-per-character knapsack scores 44/90 (vs FUSION −15..−14).

   (`results.json`)
