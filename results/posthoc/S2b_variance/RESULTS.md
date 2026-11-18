# S2b: variance (post-hoc, exploratory). Run on 2026-11-10

All numbers below come from `results.json`, `verify_repack.json`, `verify_setr.json`, `judge_consistency.json`,
`spend.json` and the `exp*/assign.jsonl` files, all in this folder. `analyze.py` builds `results.json` from the saved
frozen records/verdicts and the files in this folder, with no model call. Scores use the 90 answerable test questions.
Bounds `a..b` cover unresolved judge cells (needs_adjudication = [0,1]). Machine verdicts only; no human adjudication.
Nothing was tuned on test. Every experiment ran once.

## 1. What was run

Common setup: frozen ragbench @ 77924d4 is imported read-only (`lib.py`). The reader is `z-ai/glm-5.3-flash` built by
`reader_replay.make_reader` (pin `together`, the endpoint parameter list from MANIFEST, 7 HTTP attempts). The judge is
`openai/gpt-oss-120b` pinned to `deepinfra/bf16` (temp 0, seed 42, no fallbacks) through the frozen
`judge_multi.judge_records` (3 identical attempts, on_malformed=adjudicate). The shared `reader_budget.json` was
replaced in-process by this stream's `spend.json`, which holds the $1.50 cap. Before every batch of 50 calls or fewer,
`/api/v1/key` was checked: 107 checks, none below $2 (`money_checks.jsonl`). No GPU was used. Retrieval comes from the
saved traces.

**Gate (no model call).** `verify_repack.py` re-packed every saved LEX5B selector reply with the frozen code
(`lex4.Packager`, `lex5.bundles.parse/pack_exact/fill/hybrid_intact`, `Lex5b.pack` / `Lex5b_15500.pack`,
`build_answer_messages`). Its inputs were the saved `f2_top40`, raw reply and finish reason. **Answer-message sha256
matched the saved pack in 96/96 cells at 6000 and 96/96 at 15500** (shown chunks also 96/96).
`verify_setr.py` did the same for SETR_6000 (Hits rebuilt from the saved `order40`, `parse_selection`, `reorder`,
`pack_ordered`): **96/96 sha256 match**, selections and offered lists 96/96. Only after both gates passed did
experiments 2-6 make any calls.

| exp | what | calls |
|---|---|---|
| 1 reader repeats | the identical saved pack messages of all 8 arms x 96 questions sent again to GLM, replicates 1 and 2 (replicate 0 = the saved October answer) | 1,536 answers, 0 errors |
| 2 selector repeats (LEX5B_6000) | the saved selector prompt re-sent to GLM with the frozen selector settings (pin, 1 attempt, 3 s deadline, 600-token cap, effort low), twice; re-packed. Unchanged pack: reuse the exp-1 reader-replicate-k answer (an independent GLM draw on identical messages). Changed pack: new GLM answer. Replicate k therefore re-runs everything after retrieval | 192 selector + 94 answers |
| 2b selector repeat (SETR_6000) | the same for SETR (8 s deadline, 2,000 cap), **one** replicate. The prompt was rebuilt with the frozen `build_prompt`; prompt_chars matched the saved value for 96/96 | 96 selector + 52 answers |
| 3 selector swap | LEX5B_6000 selector = `google/gemini-3.1-flash-lite` (unpinned, served by Google). Same prompt, 3 s deadline, 600-token cap incl. reasoning, 1 attempt, effort low. GLM answers. Unchanged pack: saved answer | 96 selector + 55 answers |
| 4 relaxed deadline | the 15 LEX5B_6000 questions whose saved selector call hit the 3 s deadline: GLM selector with **no** deadline (600 cap kept), re-pack, answer | 15 selector + 13 answers |
| 5 shared selection | the saved 6000 reply packed at 15500, and the saved 15500 reply packed at 6000. Unchanged pack: saved answer of that arm | 82 answers |

Every new answer was judged (1,472 judge calls). Reused answers keep their existing verdict.

## 2. Results

### 2.1 Experiment 1: reader-only variance (identical messages)

Accuracy /90 per replicate (r0 = saved test, r1 and r2 = this run). SD is the SD of the three counts. "Flip q" is the
number of questions whose verdict differs across the 3 replicates, out of the questions resolved in all 3.

| arm | r0 | r1 | r2 | SD (count) | flip q / resolved | pairwise flips 0-1 / 0-2 / 1-2 | identical answer text 0-1 / 0-2 / 1-2 (of 96) |
|---|---|---|---|---|---|---|---|
| HYBRID_6000 | 62..63 | 60 | 58..59 | 2.0..2.1 | 10/88 | 6/9/5 | 28/23/39 |
| HR_6000 | 69..71 | 65 | 67..68 | 2.0..3.0 | 10/87 | 6/7/7 | 22/26/40 |
| LEX5B_6000 | 68..69 | 63..64 | 63 | 2.9..3.2 | 8/88 | 6/6/4 | 25/26/43 |
| SETR_6000 | 63 | 68..69 | 68 | 2.9..3.2 | 15/89 | 11/11/9 | 27/25/32 |
| HYBRID_15500 | 68 | 68 | 66 | 1.2 | 10/90 | 6/8/6 | 20/23/31 |
| HR_15500 | 69..70 | 70..71 | 73 | 1.5..2.1 | 9/88 | 6/9/4 | 24/26/32 |
| LEX5B_15500 | 67 | 68 | 68 | 0.6 | 12/90 | 9/7/8 | 23/23/34 |
| SETR_15500 | 66..67 | 67 | 72 | 2.9..3.2 | 13/89 | 13/6/7 | 21/23/31 |

- Variance components. The within-question variance over the 3 replicates gives a reader-only SD of one run's
  accuracy of **1.8-2.5 pp** per cell (`reader_sd_pp_from_within_q_var`), about 2 answers out of 90. For a one-run
  paired difference between two arms it gives **2.5-2.8 answers** (`reader_sd_net_one_run`). Between 8 and 15 of
  roughly 89 questions (9-17%) change verdict across three draws.
- Abstention: 6/6 unanswerable abstained in every replicate of every arm.
- Judge determinism: in cells where the answer text was identical to r0, the verdict agreed 382/385 times
  (`judge_consistency.json`). The judge adds a small part of the flips.
- Caveat on r0: r0 was generated 5 weeks earlier. Identical-text rates are higher between r1 and r2 (31-43) than
  between r0 and either new replicate (20-28). This suggests a time or provider component beyond pure sampling. The SDs
  above include it, and r0 is not exchangeable with r1/r2 in the strict sense.

**Do the primary paired differences survive?** Net = A minus B in answers /90. The sign p-value is a range over
unresolved cells. "Mean of 3" pairs each question's mean score over the 3 replicates. The cross column covers all 9
pairings of an A replicate with a B replicate.

| comparison | r0 net (p) | r1 net (p) | r2 net (p) | 9 cross pairings, net range | mean of 3: net, sign p, 95% CI pp question / component |
|---|---|---|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +5..+7 (0.12-0.33) | +3..+4 (0.45-0.61) | +4..+5 (0.27-0.39) | 0..+11; >0 in 7/9 | +4.0..+5.3, p 0.23-0.69, [-2.6, 13.0] / [-3.0, 14.6] |
| LEX5B_6000 vs HR_6000 | -3..0 (0.61-1.0) | -2..-1 (0.80-1.0) | -5..-4 (0.33-0.48) | -8..+4 | -3.3..-1.7, p 0.40-1.0, [-11.1, 5.6] / [-10.0, 5.4] |
| HYBRID_15500 vs HYBRID_6000 | +5..+6 (0.18-0.30) | +8 (0.022) | +7..+8 (0.039-0.065) | +3..+10; >0 in 9/9 | +6.7..+7.3, p 0.017-0.053, [1.9, 14.1] / [2.0, 14.4] |
| HR_6000 vs HYBRID_6000 | +6..+9 (0.049-0.18) | +5 (0.33) | +8..+10 (0.041-0.12) | +2..+13; >0 in 9/9 | +6.3..+8.0, p 0.052-0.31, [-0.4, 16.7] / [-1.7, 17.8] |
| SETR_6000 vs LEX5B_6000 | -6..-5 (0.18-0.30) | +4..+6 | +5 | -6..+6 | +1.0..+2.0, p 0.21-0.81 |
| LEX5B_15500 vs HYBRID_15500 | -1 | 0 | +2 | -1..+2 | +0.3 |
| LEX5B_15500 vs HR_15500 | -3..-2 | -3..-2 | -5 | -6..-1; <0 in 9/9 | -3.7..-3.0, p 0.26-0.65 |
| LEX5B budget (15500 vs 6000) | -2..-1 | +4..+5 | +5 | -2..+5 | +2.3..+3.0, p 1.0 |
| HR budget (15500 vs 6000) | -2..+1 | +5..+6 | +5..+6 | -2..+8 | +2.7..+4.3 |

By group (mean of 3 replicates, LEX5B_6000 minus HYBRID_6000, net bounds): T1 +0.3, T2 +1.3..+2.3, T3 +2.3..+2.7;
en +3.0..+3.3, fa +1.0..+2.0 (`results.json` exp1.comparisons[...].mean_of_3_by_group). Per-replicate tier and
language accuracies for every arm are in `exp1.cells[arm].by_group_rep`.

### 2.2 Experiment 2: selector + reader repeats (LEX5B_6000, GLM selector)

| | r0 (saved) | sel rep 1 | sel rep 2 |
|---|---|---|---|
| accuracy /90 | 68..69 | 65 | 67..68 |
| pack differs from saved pack (of 96) | - | 44 | 50 |
| fallback (any reason) | 19 | 10 | 12 |
| deadline fallbacks | 15 | 7 | 10 |
| other fallback reasons | 2 overflow, 1 duplicate, 1 id range | 2 overflow, 1 duplicate | 2 overflow |
| selector latency median / p90 (ms) | - | 2010 / 2698 | 1995 / 3003 |
| vs HYBRID_6000 (same reader replicate) | +5..+7 | +5 (p 0.30) | +8..+10 (p 0.021-0.057) |
| vs HR_6000 (same reader replicate) | -3..0 | 0 (p 1.0) | -1..+1 (p 1.0) |

- Replicates 1 and 2 produce the same pack for only 61/96 questions. Re-running the GLM selector on the identical
  prompt changes the evidence pack for 44-50 of 96 questions. On those changed-pack questions the accuracy moved
  little against the reader-only replicate on the same questions: rep 1 25 vs 23..24 of 43, rep 2 34..35 vs 30 of 49.
- The full-pipeline (selector + reader) between-replicate SD of LEX5B_6000 accuracy is 1.5..2.1 answers. This is no
  larger than the reader-only SD (2.9..3.2 over r0/r1/r2), so selector randomness adds no detectable variance beyond the
  reader's.
- Selector tokens: mean prompt 2,745 / 2,733, completion 77 / 72 (of which reasoning 53 / 48). The saved run had
  2,707 / 71.

**2b. SETR_6000 selector, one replicate.** Accuracy 68/90 vs 63 at r0 (+5, p 0.23, component CI [0.0, 11.4] pp). The
selection was identical for only 33/96 questions and the pack changed for 52/96. Fallbacks: 5 (4 deadline, 1 empty
selection) vs 6 saved. Against SETR's reader-only replicate 1 (68..69) it is -1..0. The SETR r0 score looks like a low
reader draw rather than a selector effect. SETR (sel rep 1) vs LEX5B_6000 (sel rep 1): +3 (p 0.63).

### 2.3 Experiment 3: selector swap (gemini-3.1-flash-lite selects, GLM answers), LEX5B_6000

| | saved GLM selector (r0) | gemini-3.1-flash-lite |
|---|---|---|
| accuracy /90 | 68..69 | **59** |
| fallback | 19 (15 deadline) | **40** (21 bad_ids, 11 deadline, 3 truncated, 2 overflow, 1 part-count, 1 no END, 1 id range) |
| replies missing the 3 s deadline | 15 | 11 |
| pack changed vs saved | - | 55/96 |
| selector tokens, mean prompt / completion (reasoning) | 2,707 / 71 | 2,642 / 159 (134) |
| selector latency median / p90 | - | 1,873 / 3,001 ms |
| selector cost (96 calls) | - | $0.076 |

Paired: vs LEX5B r0 -10..-9 (3 better / 13 worse, p 0.021-0.049, CI component [-18.2, -2.3] pp). vs HYBRID_6000 r0
-4..-3 (p 0.52-0.66). vs HR_6000 r0 -12..-10 (p 0.023-0.064). By tier: T1 28, T2 16, T3 15 (r0 29 / 23..24 / 16). By
language: en 27, fa 32 (r0 32 / 36..37). Unanswerable abstention 6/6. The loss is concentrated in T2. The swap does not
show "a stronger selector helps". Under the strict, unrepaired LEX5 bundle grammar this model mostly produces invalid
ids, and its packs then fall back to F2-intact.

### 2.4 Experiment 4: relaxed deadline (the 15 saved deadline fallbacks)

With no deadline, **14/15** selector calls now return a valid parse (1 bad_ids). Latencies: median 2,985 ms, range
1,198-5,034. 8 of 15 would have met 3 s this time, so the deadline outcome itself varies from run to run. 13/15 packs
changed. Accuracy on the 14 answerable questions: **7 vs 5 saved** (+2: 2 better, 0 worse, p 0.50).

### 2.5 Experiment 5: shared selection across budgets (LEX5B)

The 6000 and 15500 cells made separate GLM selector calls. Their parsed parts and status agree for only **37/96**
questions. Re-packing one budget's reply at the other budget changes the pack for 41/96 in each direction.

| budget effect (15500 minus 6000), net /90 | net | b/w | sign p | CI pp question / component |
|---|---|---|---|---|
| frozen (separate selector calls) | -2..-1 | 7/9 | 0.80-1.0 | [-11.1, 7.8] / [-10.6, 5.0] |
| shared selection = the 6000 reply | -3..0 | 5/7 | 0.58-1.0 | [-11.1, 7.8] / [-10.7, 7.7] |
| shared selection = the 15500 reply | -1..+1 | 6/7 | 1.0 | [-8.9, 10.0] / [-8.7, 7.3] |

Accuracies: 6000 reply @15500 = 66..68, 15500 reply @6000 = 66..68 (saved: 6000 cell 68..69, 15500 cell 67). Swapping
the reply at a fixed budget moves accuracy by -3..0 at 6000 and -1..+1 at 15500. Under shared selection, the LEX5B
budget effect stays approximately zero. The frozen "no budget gain for LEX5B" is therefore not an artefact of the two
budgets drawing different selector replies. Exp-1 shows that the reader alone moves this effect by about ±3 (r0 -2..-1,
r1 +4..+5, r2 +5).

## 3. Cost

Total **$1.3239** of the $1.50 cap (`spend.json`): GLM-5.3-flash $1.0433 (2,136 calls, 7.26 M prompt tokens),
gemini-3.1-flash-lite $0.0782 (98 calls, including 2 from a discarded first run, see 4), gpt-oss-120b judge $0.2024
(1,472 calls). The account's limit_remaining at the last check was $5.06. Query-time tokens of exp-1 answer calls (mean
prompt / completion per call): 6000 arms 2,038-2,111 / 117-123, 15500 arms 5,046-5,156 / 123-137. Each re-asked cell
makes 1 call; LEX5B/SETR cells in exp 2-4 make 1 selector call plus 0-1 answer calls.

## 4. Failed / skipped

- **SETR selector replicate 2: skipped** for money. The cap left about $0.18 after judging, and a replicate costs about
  $0.13 plus judging. Only one SETR selector replicate was run.
- A first exp-3 attempt on 3 questions hit a harness bug: an asyncio loop race in my threaded runner, which gave a
  "bound to a different event loop" call error, not a model failure. Its rows were discarded
  (`exp3_discarded_harness_bug/`, about $0.003), the bug was fixed, and exp-3 was re-run from scratch.
- Unresolved judge cells (needs_adjudication), all counted as [0,1]: exp-1 5 cells (1 each in HR_15500, HR_6000,
  HYBRID_6000, LEX5B_6000, SETR_6000), exp-2 1, exp-5 1. No judge call failures. No reader errors.
- The gemini selector was not provider-pinned (OpenRouter served Google for all 85 replies). The GLM selector used the
  frozen `together` pin.
- Answers to identical messages were reused (by sha256) in exp 2/2b (from the exp-1 replicate) and 3-5 (from r0). This
  is a design choice, stated in section 1.
- Every replicate made fresh selector calls from this machine 5 weeks after the test. Deadline outcomes depend on
  network and provider latency at that time.

## 5. Facts for the paper

1. Re-sending the identical saved GLM-5.3-flash answer messages twice more moves a cell's accuracy by an SD of about
   2 answers out of 90 (1.8-2.5 pp per cell from the within-question variance). 8-15 of about 89 questions per arm
   change verdict across the three draws. A one-run paired difference has a reader-only SD of about 2.5-2.8 answers.
   (`results.json` exp1.cells / exp1.comparisons, `reader_sd_*`)
2. LEX5B_6000 minus HYBRID_6000 stays positive in every same-replicate draw (+5..+7, +3..+4, +4..+5) but is never
   significant. Mean of 3: +4.0..+5.3, p ≥ 0.23, component CI [-3.0, 14.6] pp. LEX5B vs HR at 6000 stays at or below
   zero in all three draws. The HYBRID budget effect (+5..+8, mean +6.7..+7.3, component CI [2.0, 14.4] pp) is the
   only primary effect positive in all 9 cross-pairings. (`results.json` exp1.comparisons)
3. Re-running the GLM selector on the identical prompt changes the LEX5B_6000 evidence pack for 44-50/96 questions.
   Deadline fallbacks were 7 and 10 vs 15 saved. Full-pipeline accuracy was 65 and 67..68 vs 68..69, an SD of 1.5-2.1
   answers, no larger than the reader-only variation. (`results.json` exp2, `exp2/assign.jsonl`)
4. Swapping in gemini-3.1-flash-lite as the LEX5B selector (same prompt, 3 s, 600 tokens) lowered accuracy to 59/90
   (-10..-9 vs GLM-selected, p 0.02-0.05). It fell back on 40/96 questions, 21 of them for invalid ids, and 11 replies
   missed the deadline. Under the strict grammar, the selector-reader coupling cannot be separated by this swap in
   favour of "a stronger selector". (`results.json` exp3)
5. Without a deadline, 14 of the 15 saved deadline fallbacks return a valid selection (median 2.99 s). Accuracy on
   those questions is 7 vs 5 of 14. Under shared selection the LEX5B budget effect remains about 0 (-3..0 and -1..+1 vs
   frozen -2..-1), so the absence of a LEX5B budget gain is not caused by separate selector draws.
   (`results.json` exp4, exp5)
