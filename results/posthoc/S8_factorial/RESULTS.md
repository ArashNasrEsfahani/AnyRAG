# S8 "factorial": Lex-RAG across budgets, selector over HR's candidates, no-deadline selector, judge 2, dev replication

Post-hoc and exploratory (COMMON.md). The test split was unsealed before this ran. Nothing was tuned on test: every
arm below was defined before its single test run, and no arm was re-run after seeing results.
Reader: `z-ai/glm-5.3-flash`, pinned to `together` (MANIFEST reader pin, frozen `OpenRouterReader`). Judge 1: the pinned
`openai/gpt-oss-120b` (deepinfra/bf16, temperature 0, seed 42, no fallbacks, 3 judge attempts, `on_malformed=adjudicate`)
through the frozen `ragbench.tools.judge_multi.judge_records`. Frozen code: `D:\KareOne\RAG-Paper-upgrades` @ 77924d4,
imported read-only. No GPU (`CUDA_VISIBLE_DEVICES=""`). Accuracy is /90 answerable test questions as bounds lo..hi
over unresolved judge cells; the 6 unanswerable are reported as abstention /6.

All numbers come from `results.json` / `tables.md` (written by `analyze.py`), `step0_checks.json`, `split_nodl.json`,
`dev_fusion/gate.json`, `spend.json`, `ledger.jsonl` and `ledger_j2.jsonl` in this folder.

## 1. What was run

**Code.** `s8lib.py` is a copy of `S2b_variance/lib.py` (own folder, own $1.20 cap, own `spend.json`), with two
additions: any pack budget, and the literal selector view over any chunk list. `s8run.py` runs steps 1-3,
`judge2_posthoc.py` runs step 4, `dev_fusion.py` runs step 5, `analyze.py` and `split_nodl.py` compute the results.
Statistics use the `revision_stats.py` helpers, extracted with `ast` and run unchanged (as S2a did). The one exception:
when a comparison has more than 12 unresolved cells, the sign-p range is computed by an exact DP over the reachable
(better, worse) counts instead of the 2^k enumeration. The output is the same quantity.

**Step 0: checks, no model calls** (`step0_checks.py` -> `step0_checks.json`). All 96/96 pass:
- (a) The saved LEX5B_6000 and LEX5B_15500 selector prompts are byte-identical, and so are their F2 top-40 lists. The
  selector prompt does not depend on the budget.
- (b) The frozen literal view (`lex5b/views.Viewer.fit`), rebuilt from the saved F2 top 40, reproduces the saved
  6000 prompt byte for byte. The literal view never reads the lex8 unit index or the bge-m3 unit scores, so it needs
  no encoder.
- (c) Re-packing the saved replies with the frozen code reproduces the saved answer-message sha256 at 6000 and at
  15500.
- (d) The fallback re-packed at 3000, 6000, 10000, 15500 and 24000 equals S2a's FUSION_<B> pack sha.

**Step 1: Lex-RAG at 3,000 / 10,000 / 24,000** (`s8run.py lexbud`).
- Because of (a), **the saved LEX5B_6000 selector reply was reused at every budget**. No new selector calls were made.
- The reply was re-packed by the frozen `Lex5b._answer` steps 5-7 at budget B. The class follows the `budget_matrix`
  pattern (`Lex5b_15500` with only `budget_chars` changed): fallback, exact bundle packing, fill and rendered pack
  are all at B.
- With the saved reply, the selector outcome is the same at every budget: 19 fallbacks (15 deadline, 2
  overflow_alternatives, 1 duplicate_alternative, 1 id_out_of_range), 71 ok and 6 all_none.
- A pack byte-identical to S2a's FUSION_B pack reuses S2a's answer and verdict: 40 cells at 3000, 37 at 10000 and 37
  at 24000. These are the 19 fallbacks plus selections that give the fallback pack.
- The other 174 cells got a fresh GLM answer (0 errors) and judge 1.
- With the frozen 6000 and 15500 cells, Lex-RAG, FUSION, HYBRID and HR now each have five budgets. The 15500 Lex-RAG
  cell is the frozen one, with its own selector call.

**Step 2: the selector over HR's reranked candidates** (`s8run.py selhr`). This was feasible without a GPU:
- HR's saved `retrieved_units` (rerank order, 40 kept) carry `meta.chunk`, which is the same HybridIndex chunk id
  space as the F2 list. So the frozen lex4 packages and the literal view apply unchanged; no new embedding or rerank
  is needed.
- The frozen view shows HR's top 24, followed by the same selector call, the strict grammar, and exact bundle packing
  plus fill in HR order. The fallback is `hybrid_intact` over the HR order.
- **The selector was called with no deadline** (600-token cap, 1 attempt, effort low kept). That makes it directly
  comparable with step 3, so SELHR vs LEX5B_NODL isolates the candidate source.
  - 8 of 96 calls took more than 3 s; under the frozen deadline those would have fallen back.
  - The new selector replies fell back in 4 cells (all overflow_alternatives); 6 were all_none.
- All 96 cells were answered fresh.

**Step 3: the no-deadline selector at 6,000 on all 96** (`s8run.py nodl`).
- S2b's exp4 covered only the 15 deadline fallbacks, so all 96 were re-run here.
- The saved prompt was re-sent to the pinned GLM selector with no deadline (600-token cap, 1 attempt, effort low).
  The reply was re-packed by the frozen code.
- An unchanged pack reuses the frozen LEX5B_6000 answer and verdict (51 cells). The other 45 were answered fresh.
- 0 calls failed. 14 of 96 calls took more than 3 s (median 2.1 s).

**Step 4: judge 2 on the post-hoc arms** (`judge2_posthoc.py`). Only partly completed at the time of writing; see section 3.1.
> **Update 2026-11-11 (coordinator):** section 3.1 is stale. The judge-2 retry (`logs/j2_retry2.log`, then `analyze.py`)
> completed on 2026-11-10 for FUSION_6000, SETR_NOFILL, LISTWISE, ITER_BM25, GREP_AGENT, GOLD and CLOSEDBOOK. The current
> judge-2 accuracies and contrasts are in `tables.md` ("Accuracy, judge 2"), which the paper cites. The selector-over-HR
> and no-deadline arms and the 3,000/10,000/24,000 sweep were never scored by judge 2.
- Setup, exactly as S1: `mistralai/mistral-large-2512` pinned to `mistral/zdr` (`allow_fallbacks` false,
  `require_parameters` true), temperature 0, top_p 1, seed 42, max_tokens = num_predict + 64. It runs through the
  frozen `judge_multi.judge_records` (same rubric, prompt and parser, 3 judge attempts, on_malformed adjudicate). The
  client is S1's, copied.
- Records are exactly those each stream fed to judge 1. FUSION_6000 cells that are the frozen LEX5B_6000 answer take
  S1's judge-2 verdict of that answer.
- One deviation from S1: up to 12 instead of 6 transport retries for HTTP 429, and failed records re-tried. This
  affects transport only, not judge attempts.

**Step 5: dev replication** (`dev_fusion.py`, added by the coordinator). FUSION_6000 was rebuilt on all 60 dev
questions from the saved dev LEX5B@6000 traces: the 24 idea_test records in `lex5b/eval/runs` and the 36 in
`dev60/runs`, which are the records dev60 scored.
- Gate (`dev_fusion/gate.json`):
  - the rebuilt fallback equals the saved `fallback_chunks` in 60/60;
  - in the 11 saved selector-fallback cells, the rebuilt shown chunks and the evidence chars equal the saved ones
    (11/11).
- All 60 were answered fresh with the frozen answer prompt and judged by judge 1 using `dev_v2judge_all.jsonl`.
- The frozen dev arms use the saved dev60 verdicts (`dev60/dev60_results.json` per_q). These reproduce DEV60.md's
  numbers: HYBRID 30, HR 32, LEX5B 37, HYBRID@15500 38.

## 2. Results (judge 1)

### 2.1 Pipeline × budget (correct /90; * = frozen test cell; FUSION/HYBRID/HR off-6000/15500 cells from S2a)

| pipeline | 3,000 | 6,000 | 10,000 | 15,500 | 24,000 |
|---|---|---|---|---|---|
| HYBRID | 49..51 | 62..63* | 63..64 | 68* | **72** |
| HR | 56 | 69..71* | 70 | 69..70* | 66 |
| FUSION (Lex-RAG without selector) | 43..44 | 58..59 | 66 | 65 | 70 |
| **Lex-RAG (LEX5B)** | **59** | 68..69* | 69..70 | 67* | **72** |

Lex-RAG at the new budgets, by tier and language:
- 3000: T1 30, T2 16, T3 13; fa 31, en 28.
- 10000: T1 29, T2 24, T3 16..17; fa 37, en 32..33.
- 24000: T1 28, T2 25, T3 19; fa 37, en 35.

Abstention is 6/6 at every budget.

Mean rendered evidence is 2,810 / 9,764 / 23,745 chars.

Tokens and calls per question: 2 calls, the saved selector call (S6: 2,284 prompt + 60 completion tokens per
question) plus the answer call. The mean answer prompt over the freshly answered cells is 1,083 / 3,210 / 7,498 tokens
at 3000 / 10000 / 24000.

### 2.2 What the selector adds at each budget (paired /90; sign p; 95% CI pp over questions / 20 components)

| comparison | net | b/w | sign p | CI q | CI comp |
|---|---|---|---|---|---|
| LEX5B vs FUSION @3000 | **+15..+16** | 17/2 | 0.0004-0.0007 | [7.8, 26.7] | [7.7, 30.0] |
| LEX5B vs FUSION @6000* | **+9..+11** | 12/2 | 0.007-0.035 | [2.2, 20.0] | [3.3, 20.6] |
| LEX5B vs FUSION @10000 | +3..+4 | 4/1 | 0.22-0.38 | [-1.1, 10.0] | [0.0, 8.3] |
| LEX5B vs FUSION @15500* | +2 | 8/6 | 0.79 | [-5.6, 10.0] | [-5.6, 9.5] |
| LEX5B vs FUSION @24000 | +2 | 4/2 | 0.69 | [-3.3, 7.8] | [-2.3, 6.2] |
| LEX5B vs HYBRID @3000 | +8..+10 | 15/5 | 0.041-0.13 | [-1.1, 21.1] | [0.0, 23.0] |
| LEX5B vs HYBRID @6000* | +5..+7 | 11/4 | 0.12-0.33 | [-3.3, 16.7] | [-4.3, 17.7] |
| LEX5B vs HYBRID @10000 | +5..+7 | 7/2 | 0.065-0.18 | [0.0, 15.6] | [0.0, 15.3] |
| LEX5B vs HYBRID @15500* | -1 | 5/6 | 1 | [-8.9, 5.6] | [-8.9, 6.6] |
| LEX5B vs HYBRID @24000 | 0 | 4/4 | 1 | [-6.7, 6.7] | [-5.3, 6.5] |
| LEX5B vs HR @3000 | +3 | 13/10 | 0.68 | [-7.8, 14.4] | [-6.6, 13.9] |
| LEX5B vs HR @6000* | -3..0 | 6/7 | 0.61-1 | [-12.2, 7.8] | [-10.3, 7.5] |
| LEX5B vs HR @10000 | -1..0 | 7/8 | 1 | [-8.9, 8.9] | [-5.8, 8.1] |
| LEX5B vs HR @15500* | -3..-2 | 7/9 | 0.63-0.80 | [-12.2, 6.7] | [-11.5, 6.8] |
| LEX5B vs HR @24000 | +6 | 11/5 | 0.21 | [-2.2, 15.6] | [-1.0, 15.9] |

Lex-RAG's own budget steps:
- 3000→6000: +9..+10 (11/1, p 0.006-0.023);
- 6000→10000: 0..+2;
- 10000→15500: -3..-2;
- 15500→24000: +5 (p 0.18);
- 6000→24000: +3..+4 (p 0.39-0.55).

Tier and language splits of every row are in `tables.md`. At 3000, the selector gain over FUSION is T1 +1, T2 +9,
T3 +5..+6, and fa +7..+8, en +8.

### 2.3 The selector over HR's reranked top 24 (SELHR_6000; no-deadline selector)

| comparison | net | b/w | sign p | CI q | CI comp |
|---|---|---|---|---|---|
| SELHR_6000 = **70** /90 (T1 28, T2 22, T3 20; fa 35, en 35; abst 6/6) | | | | | |
| SELHR vs LEX5B_NODL_6000 (same selector setting, F2 candidates) | -2..-1 | 6/7 | 0.79-1 | [-10.0, 6.7] | [-14.9, 9.0] |
| SELHR vs HR_6000* (same candidates, no selector) | -1..+1 | 9/8 | 1 | [-11.1, 10.0] | [-11.9, 9.6] |
| SELHR vs LEX5B_6000* (frozen, 3 s deadline) | +1..+2 | 7/6 | 0.79-1 | [-6.7, 10.0] | [-10.5, 12.2] |

The selector over the cross-encoder's candidates adds nothing measurable over HR's own pack. Feeding the selector
reranked rather than F2 candidates does not help either.

Selector tokens per question: 2,736 prompt + 86 completion. Answer prompt: 2,053.

### 2.4 No-deadline selector at 6,000 (LEX5B_NODL_6000), and the 19 saved fallbacks

LEX5B_NODL_6000 scores **71..72**/90 (T1 29, T2 24, T3 18..19; fa 38..39, en 33; abst 6/6).

| comparison | net | b/w | sign p | CI q | CI comp |
|---|---|---|---|---|---|
| NODL vs LEX5B_6000* | +2..+4 | 3/1 | 0.22-0.63 | [-2.2, 10.0] | [0.0, 9.2] |
| NODL vs FUSION_6000 | +12..+14 | 14/2 | 0.0013-0.0042 | [5.6, 24.4] | [6.7, 24.8] |
| NODL vs HYBRID_6000* | +8..+10 | 13/4 | 0.031-0.096 | [0.0, 20.0] | [0.0, 21.9] |
| NODL vs HR_6000* | 0..+3 | 7/6 | 0.61-1 | [-7.8, 11.1] | [-7.6, 13.0] |

**The 19 saved fallbacks** (`split_nodl.json`, `tables.md` per-question table):
- **Pack.** With no deadline, 16 of the 19 get a different pack.
  - All 16 have a valid reply (status ok).
  - Of the 3 unchanged: LAW-fa-T3-001 fails the grammar again (duplicate_alternative), LAW-fa-AB-001 returns
    all_none and so gives the fallback pack, and BOOK-fa-T1-003 selects the fallback pack.
- **Answers.** 18 of the 19 are answerable. Of these, 2 move from wrong to right (RPT-en-T3-002, LAW-fa-T3-005), 0
  move from right to wrong, and 1 is unresolved (BOOK-fa-T3-003, wrong in the saved run, 0..1 now).
  - Net on the fallbacks: +2..+3 (8 → 10..11 of 18).
- **Timing.** The 15 deadline fallbacks were all re-sent: 7 of these calls again took more than 3 s; the other 8
  finished under 3 s this time. Over all 96 calls, 14 exceed 3 s.
- The 4 grammar fallbacks: 3 now parse (overflow_alternatives ×2, id_out_of_range), 1 does not.

**The 77 saved successful selections** also change: re-sending the same prompt gives a different pack in 29 of them.
On the 72 answerable ones (28 with a changed pack), the net is 0..+1 (1/1). So the NODL − LEX5B difference comes from
the fallbacks. It is not caused by selector resampling.

Agreement with S2b exp4 (a separate no-deadline run of the 15 deadline fallbacks): the same pack in only 6 of 15. The
selector's choices are themselves nondeterministic.

## 3. Results under judge 2 (step 4) and on dev (step 5)

### 3.1 Judge 2: incomplete

**Not completed.** The pinned judge-2 endpoint (`mistral/zdr`, the only Mistral endpoint) was rate-limited upstream for
the whole session. The error was HTTP 429 "temporarily rate-limited upstream ...
`limit_source: upstream_provider_shared_pool`".
- **Throughput.** 241 requests were sent; 36 succeeded and 205 got a 429. Throughput fell to about one verdict every
  2-5 minutes, so the ~560 remaining calls would have taken more than 10 hours.
- **What was done.** I ran the 3-question estimate on all 7 arms, then gave FUSION_6000 priority and stopped the run
  at 18:59.
- **Where the arms stand.** Judge 2 has resolved 48 of 90 answerable FUSION_6000 cells; 37 of these are frozen
  LEX5B_6000 answers already judged by S1. The other arms have 0-11 resolved cells (`tables.md`, "Accuracy, judge 2").
- **No judge-2 contrasts are reported.** With this many unresolved cells, the bounds span almost the whole range.
- **Partial agreement** (`tables.md`, "Judge 2 partial"; cells resolved by both judges, all 96 questions):

| arm | resolved by both | agree | J1=1, J2=0 | J1=0, J2=1 |
|---|---|---|---|---|
| FUSION_6000 | 54 | 49 | 3 | 2 |
| ITER_BM25 | 11 | 10 | 1 | 0 |
| SETR_NOFILL / LISTWISE / GREP_AGENT / GOLD | 3 / 2 / 3 / 1 | all | 0 | 0 |
| CLOSEDBOOK | 0 | - | - | - |

This is consistent with S1's 91% agreement, but it is not evidence about the contrasts.
- **To resume.** The run is resumable at no extra cost for the finished cells:
  `S8_J2_TRANSPORT=12 python judge2_posthoc.py --arms FUSION_6000,SETR_NOFILL,LISTWISE,ITER_BM25,GREP_AGENT,GOLD,CLOSEDBOOK --retry-failures`,
  then `python analyze.py`. That fills the "Key post-hoc contrasts under both judges" table in `tables.md`, which
  already pairs each contrast's J1 and J2 rows.
- **Estimated cost to finish:** about $0.50. The 3-question estimate was $0.00086 per judge-2 call, and about 560 calls
  remain.

### 3.2 Dev replication (54 answerable dev questions; judge 1)

| cell | correct /54 | abstention /6 |
|---|---|---|
| FUSION@6000 (new) | **29** | 5 |
| HYBRID@6000 | 30 | 5 |
| HR@6000 | 32 | 6 |
| LEX5B@6000 | 37 | 6 |
| HYBRID@15500 | 38 | 6 |
| HR@15500 | 38 | 6 |

| comparison (dev) | net /54 | b/w | sign p | 95% CI pp (questions) |
|---|---|---|---|---|
| LEX5B@6000 vs FUSION@6000 | **+8** | 9/1 | **0.021** | [3.7, 25.9] |
| FUSION@6000 vs HYBRID@6000 | -1 | 2/3 | 1 | [-9.3, 5.6] |
| FUSION@6000 vs HR@6000 | -3 | 3/6 | 0.51 | [-16.7, 5.6] |
| HYBRID 15500 vs 6000 | +8 | 11/3 | 0.057 | [1.9, 27.8] |
| HR 15500 vs 6000 | +6 | 6/0 | 0.031 | [3.7, 20.4] |

- **Identical packs.** In 25 of 60 dev questions the fallback pack equals the saved LEX5B pack; 20 of those are
  answerable. On those 20, the fresh FUSION answers score 13 and the saved LEX5B answers also score 13.
- **No significance test here.** The dev bootstrap is over questions only.
- **Dev is not untouched.** It is independent of test, but these dev questions were used to develop and select
  Lex-RAG (DEV60.md).

## 4. Cost

- **Total S8 spend: $0.324** of the $1.20 cap.
  - **GLM and judge 1: $0.2839**, 913 calls (`spend.json`; per-call rows in `ledger.jsonl`):
    - selector: nodl 96 calls $0.013, selhr 96 calls $0.044;
    - answers: lexbud 174 $0.114, nodl 45 $0.010, selhr 96 $0.034, dev 60 $0.021;
    - judge 1: 346 calls $0.048.
  - **Judge 2: $0.0405**, 36 successful calls (`ledger_j2.jsonl`). The 205 HTTP-429 responses were not billed.
- **Estimate first.** The 3-question estimate cost $0.006 for steps 1-3 (19 calls) and extrapolated to about
  $0.2 for those steps; the actual cost was $0.26 including the dev step. Judge 2 was estimated at $0.00086 per call.
- **Floor checks.** `limit_remaining` was checked before every batch of at most 50 calls (`money_checks.jsonl`, plus
  the judge-2 client's own check every 50 calls). The lowest value logged was $4.75 and the final value $4.72, so the
  $2.00 floor never triggered.
- **Query-time tokens per question.**
  - Lex-RAG at the new budgets makes 2 calls per question: the saved selector call (about 2,284 + 60 tokens, S6), plus
    answer prompts of 1,083 / 3,210 / 7,498 tokens at 3000 / 10000 / 24000 (mean over fresh cells).
  - SELHR: selector 2,736 + 86, answer 2,053 + 125.
  - NODL: selector 2,727 + 87, answer 2,037 + 160 over the 45 fresh cells.
  - **Latency** is descriptive only: the median selector wall time was 2.1 s (NODL) and 1.9 s (SELHR).

## 5. Failed, skipped, deviations

- **Judge 2 (step 4) is incomplete:** the provider rate-limited the pinned endpoint upstream (section 3.1). No
  judge-2 contrast is reported, and CLOSEDBOOK has no judge-2 cells. The resume command is in section 3.1.
- **Step 1 reuses the saved selector reply** instead of new selector calls. This follows from the check that the
  prompt is identical across budgets (step0 a/b). The reply is therefore the same draw at all five budgets, except
  15500, which is the frozen cell with its own call. Its 19 fallbacks (15 deadline) carry over to every budget.
- **Steps 2 and 3 use no deadline,** by design; this differs from the frozen 3 s.
  - The SELHR vs LEX5B_6000 comparison therefore mixes the candidate source with the deadline. SELHR vs
    LEX5B_NODL_6000 is the clean comparison.
- **Not run:**
  - **SetR at 3,000/10,000/24,000** (the other half of review item 10) was not in this brief.
  - **The selector over HYBRID's candidates** was not in this brief either. It would work the same way as SELHR,
    since the HYBRID units also carry `meta.chunk`.
- **Unresolved judge-1 cells:**
  - S8's own: 1 needs_adjudication in lexbud (RPT-en-T3-001 at 10000) and 1 in nodl (BOOK-fa-T3-003). Both are reported as [0, 1] bounds and
    were not adjudicated.
  - Reused cells carry S2a's and the frozen arms' unresolved cells.
- **Bootstrap scope.** Dev bootstrap intervals are over questions only; the source-family components are defined for
  test.
- **Dev is not untouched data.** It was used to develop Lex-RAG; it is only independent of test.
- **No GPU** was used. No new dense, sparse or rerank computation was made: SELHR uses HR's saved rerank order.
- `CUDA_VISIBLE_DEVICES=""` was set in every process. Nothing outside `S8_factorial` was written.

## 6. Facts for the paper

1. **The selector prompt is budget-free.** It is byte-identical at 6,000 and 15,500 (96/96), and the frozen code
   re-packs the saved replies exactly (`step0_checks.json`). Re-packing the saved 6,000 reply at each budget completes
   the grid: Lex-RAG scores 59 / 68..69 / 69..70 / 67 / 72 of 90 at 3k / 6k / 10k / 15.5k / 24k. For comparison:
   - HYBRID: 49..51 / 62..63 / 63..64 / 68 / 72;
   - HR: 56 / 69..71 / 70 / 69..70 / 66;
   - FUSION: 43..44 / 58..59 / 66 / 65 / 70.

   (`results.json` acc_j1, `tables.md`)
2. **The selector's gain over its own fusion candidates shrinks as the budget grows:** +15..+16 at 3,000 (17/2, sign p
   <= 0.0007), +9..+11 at 6,000, +3..+4 at 10,000, +2 at 15,500 and +2 at 24,000. Against HR, Lex-RAG is never
   significantly different (nets -3..+6, all p >= 0.21) (`results.json` cmp_j1).
3. **Running the Lex-RAG selector over HR's reranked top 24** (no GPU; same chunk ids) gives 70/90. That is no
   different from HR's own pack (-1..+1, 9/8) or from the same selector over F2 candidates (-2..-1 vs
   LEX5B_NODL_6000) (`results.json` cmp_j1 "SELHR_6000 vs ...").
4. **Without the 3 s deadline,** 16 of the 19 saved LEX5B_6000 fallbacks get a different pack. On the 18 answerable
   ones this turns 2 wrong answers right, none right to wrong, and leaves 1 unresolved (+2..+3). The full no-deadline
   arm scores 71..72 vs 68..69 (+2..+4, 3/1, p 0.22-0.63). Re-sampling the 77 successful selections changes 29 packs
   but not the score (net 0..+1) (`split_nodl.json`, `results.json` nodl_fallbacks).
5. **On the 54 answerable dev questions,** the selector-free FUSION_6000 scores 29. That is below LEX5B (37; +8,
   9/1, p 0.021) and level with HYBRID (30) and HR (32). Plain budget raises HYBRID 30->38 (p 0.057) and HR 32->38
   (6/0, p 0.031). This replicates the test pattern: the 6,000-char gain is the selector's, and budget substitutes for
   it (`results.json` dev; `dev_fusion/gate.json` for the 60/60 rebuild check).
6. **Judge 2 could not be run on the post-hoc arms** because Mistral's pinned endpoint was rate-limited upstream.
   Where both judges resolved a FUSION_6000 cell, they agree in 49 of 54 (`tables.md` "Judge 2 partial").
