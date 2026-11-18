# S6 baselines: CLOSED-BOOK, LLM-LISTWISE (RankGPT), SETR-NOFILL (post-hoc, exploratory)

All numbers below are taken from `analysis.json` (written by `analyze.py`), `verify_repack.jsonl`, `cost_estimate.json` and `ledger.jsonl` in this folder.
These runs are post-hoc and come after the test was unsealed. No test question was used for tuning. The listwise prompt was checked on 3 dev questions and was not changed after that.
Reader: `z-ai/glm-5.3-flash`, pinned to `together` with the manifest's param list, 7 HTTP attempts, and the frozen `OpenRouterReader` payload rule. Judge: `openai/gpt-oss-120b`, pinned to `deepinfra/bf16` with no fallbacks, temperature 0, top_p 1, seed 42, at most 3 identical attempts, malformed replies sent to adjudication. It ran through the frozen `ragbench.tools.judge_multi.judge_records`. The frozen code is imported read-only from `D:\KareOne\RAG-Paper-upgrades` at 77924d4.
**The frozen arms are compared on their saved machine verdicts.** On those verdicts HYBRID / HR / LEX5B / SETR at 6000 characters score 62..63 / 69..71 / 68..69 / 63 of 90. The paper's 71 / 76 / 76 / 78 come from the symmetric human merge, and the new arms have no human verdicts.

## 1. What was run

**Step 0, re-pack check (`verify_repack.py`, no model calls).** I rebuilt RetrievedItems from the saved `retrieved_units` and ran them through the frozen `_offer` + `final_pack` + `build_answer_messages`.
- HYBRID_6000: 96/96 messages sha256 match the saved packs and the records' `answer_messages_sha256`.
- SETR_6000: the HYBRID order came from `trace.setr.order40`, the selection from `trace.setr.selector.selected`, and the order from `setr.reorder(top=20)` with fill. Result: **96/96 byte-identical** to `test_run/packs/SETR/`.
- Re-parsing the saved raw selector reply with the frozen `parse_selection` gives the saved selection in 90/90 non-fallback cells. The other 6 cells are selector fallbacks. num_predict matches in all cells.

**CLOSEDBOOK** (96 questions). I used the codebase's own no-evidence variant of the answer prompt, `build_answer_messages(query, None)`:
- the same per-format system prompt (v2_single / list / comparison / series / explanation, same answer language, same num_predict);
- the evidence rule is replaced by `SOURCE_RULE_NONE` ("No documents are provided; answer from your own knowledge.");
- the abstain rule is replaced by `ABSTAIN_RULE_NONE` ("If you do not know the answer, reply with exactly NOT_FOUND and nothing else.");
- the user message is `Question: <q>`.

I did not use the literal "empty evidence section" version: the test prompt says "use only the numbered evidence passages", so that version would only measure abstention.

**LISTWISE** (96 questions). RankGPT (Sun et al. 2023) listwise reranking with GLM, one call per question:
- Candidates: HYBRID_6000's saved top-20 fused chunks.
- Prompt: RankGPT's multi-turn prompt verbatim (system "You are RankGPT…", prefix, `[i] passage` / "Received passage [i]." turns, then the post-instruction asking for `[] > []` output).
- Truncation: newlines collapsed, each passage cut to its first **700 characters** plus " …". On average 15.8 of the 20 passages were cut (chunks are at most 1,000 chars, median 871). The mean ranker prompt was 14,902 characters.
- Ranker call: same pinned GLM endpoint, reasoning effort low, max_tokens 2000 including reasoning, 3 HTTP attempts, no deadline.
- Parsing: RankGPT's `receive_permutation` (digits in reply order, deduplicated, in range; missing ids appended in HYBRID order). Candidates 21-40 follow in HYBRID order.
- Packing: HYBRID's frozen `_offer` (fill) + packer (same clipping rule), then the shared answer call.

**SETR_NOFILL** (96 questions). SETR_6000's saved selected set, in selection order, offered alone to the frozen packer: no `_offer` fill from the HYBRID list, but the packer's own 6,000-char cut still applies.
- In the 6 selector-fallback cells there is no selected set, so SETR's own fallback pack was used (HYBRID order with fill).
- 7 NOFILL packs came out byte-identical to the saved SETR packs. These were still answered fresh; no saved answer was reused.

Statistics follow revision_stats: bounds over unresolved cells, the exact two-sided sign-test p range over all assignments of unresolved cells, and 95% bootstrap intervals (10,000 resamples, seed 0) over questions and over the 20 connected components of source families. The helper code is copied verbatim into `analyze.py`, because importing revision_stats would rewrite its output files.

## 2. Results (strict correct of 90 answerable questions; lo..hi = bounds over needs_adjudication cells)

| arm | all /90 | T1 /30 | T2 /30 | T3 /30 | fa /45 | en /45 | BOOK /30 | LAW /30 | RPT /30 |
|---|---|---|---|---|---|---|---|---|---|
| CLOSEDBOOK | 26 | 9 | 8 | 9 | 6 | 20 | 12 | 14 | 0 |
| LISTWISE | 66..67 | 28 | 21 | 17..18 | 33..34 | 33 | 19 | 22..23 | 25 |
| SETR_NOFILL | 63 | 27 | 18 | 18 | 31 | 32 | 17 | 22 | 24 |
| HYBRID_6000 (frozen) | 62..63 | 28 | 20..21 | 14 | 35..36 | 27 | 19 | 22 | 21..22 |
| HR_6000 (frozen) | 69..71 | 28 | 23..24 | 18..19 | 36 | 33..35 | 20 | 23..24 | 26..27 |
| LEX5B_6000 (frozen) | 68..69 | 29 | 23..24 | 16 | 36..37 | 32 | 20 | 22..23 | 26 |
| SETR_6000 (frozen) | 63 | 27 | 18 | 18 | 33 | 30 | 20 | 19 | 24 |

Unresolved cells: 1 in LISTWISE (needs_adjudication), 0 in CLOSEDBOOK and SETR_NOFILL. There were no run errors and no answers cut off by length in any S6 arm.

**Closed-book by pool** (correct /15):

| | BOOK-fa | BOOK-en | LAW-fa | LAW-en | RPT-fa | RPT-en |
|---|---|---|---|---|---|---|
| CLOSEDBOOK | 3 | 9 | 3 | 11 | 0 | 0 |

- On answerable questions, CLOSEDBOOK abstained (NOT_FOUND) 22/90 times: RPT 19 (fa 14, en 5), BOOK-fa 3, LAW 0.
- Cell overlap with HYBRID_6000 (resolved cells): both correct 18, closed-book only 8, HYBRID only 44, neither 19, unresolved 1.

**Unanswerable questions** (6; correct = judge rule "abstained on an unanswerable question"):
- Every evidence arm scores 6/6. The frozen HYBRID and HR score 6/6 too, with 5 literal abstentions each.
- CLOSEDBOOK scores **3/6**. It abstained on RPT-fa, RPT-en and LAW-fa, and answered from memory on BOOK-fa, BOOK-en and LAW-en.

**Paired comparisons** (all 90 answerable questions):
- net = a - b in questions (bounds);
- b/w = better/worse count over resolved pairs;
- p = exact sign-test range;
- CI = 95% bootstrap in percentage points, over questions and over the 20 components.

| comparison | net | b/w | sign p | CI q (pp) | CI comp (pp) | net T1 / T2 / T3 | net fa / en |
|---|---|---|---|---|---|---|---|
| CLOSEDBOOK vs HYBRID | -37..-36 | 8/44 | 2.4e-7 - 4.0e-7 | [-54.4, -26.7] | [-66.7, -20.9] | -19 / -13..-12 / -5 | -30..-29 / -7 |
| LISTWISE vs HYBRID | +3..+5 | 12/7 | 0.359-0.664 | [-6.7, 15.6] | [-6.7, 15.5] | 0 / 0..+1 / +3..+4 | -3..-1 / +6 |
| LISTWISE vs HR | -5..-2 | 7/10 | 0.359-0.815 | [-14.4, 6.7] | [-11.6, 4.6] | 0 / -3..-2 / -2..0 | -3..-2 / -2..0 |
| HR vs HYBRID (frozen, reference) | +6..+9 | 10/4 | 0.049-0.180 | [-1.1, 18.9] | [-1.0, 21.2] | 0 / +2..+4 / +4..+5 | 0..+1 / +6..+8 |
| LISTWISE vs LEX5B | -3..-1 | 7/9 | 0.629-1.000 | [-12.2, 7.8] | [-11.7, 7.0] | -1 / -3..-2 / +1..+2 | -4..-2 / +1 |
| LISTWISE vs SETR | +3..+4 | 10/6 | 0.454-0.629 | [-5.6, 13.3] | [-7.6, 14.9] | +1 / +3 / -1..0 | 0..+1 / +3 |
| SETR_NOFILL vs SETR | 0 | 4/4 | 1.000 | [-6.7, 6.7] | [-6.7, 7.9] | 0 / 0 / 0 | -2 / +2 |
| SETR_NOFILL vs HYBRID | 0..+1 | 10/9 | 1.000 | [-10.0, 10.0] | [-10.9, 11.5] | -1 / -3..-2 / +4 | -5..-4 / +5 |
| SETR_NOFILL vs LEX5B | -6..-5 | 6/12 | 0.238-0.359 | [-15.6, 3.3] | [-17.4, 3.6] | -2 / -6..-5 / +2 | -6..-5 / 0 |
| SETR vs HYBRID (frozen, reference) | 0..+1 | 8/7 | 1.000 | [-8.9, 10.0] | [-9.3, 9.5] | -1 / -3..-2 / +4 | -3..-2 / +3 |

**Pack descriptives**
- LISTWISE ranker: 96/96 calls ok, no fallback, all 96 returned a full 20-id permutation, no length stops. Median ranking wall time 2.0 s (max 8.9 s).
- LISTWISE packed chunk sets: mean Jaccard 0.42 with HR's and 0.44 with HYBRID's. The packed order equalled HYBRID's in 0 cells.
- SETR_NOFILL: 90 non-fallback sets with a mean of 3.04 selected chunks. Set sizes: 1 chunk 8, 2 chunks 22, 3 chunks 30, 4 chunks 20, 5 chunks 8, 6 chunks 2.
- Rendered evidence: 3,056 chars on average, against 5,949 for SETR with fill. The last item was cut in 6 cells. The pack was identical to SETR's in 7 cells. Abstentions on answerable questions: 6 (SETR: 4).

**Query-time LLM tokens per question** (96 questions; usage known for every S6 call):

| arm | calls/q | prompt tok/q | completion tok/q | notes |
|---|---|---|---|---|
| CLOSEDBOOK | 1 | 199 | 118 | answer only |
| LISTWISE | 2 | 4,675 (rank) + 2,099 (answer) = 6,774 | 139 + 115 = 254 | ranker reasoning 5,508 tok over 96 calls |
| SETR_NOFILL | 1 (+ SETR's saved selector call) | 1,170 answer (+ 5,060 saved selector) | 114 (+ 221) | the selector call was not re-run |
| HYBRID_6000 (frozen) | 1 | 2,093 | 123 | |
| HR_6000 (frozen) | 1 LLM call (+ cross-encoder, no LLM tokens) | 2,111 | 126 | |
| LEX5B_6000 (frozen) | 2 | 2,284 + 2,038 | 60 + 129 | 15 selector calls with unknown usage |
| SETR_6000 (frozen) | 2 | 5,060 + 2,101 | 221 + 121 | 6 selector calls with unknown usage |

## 3. Cost
- **Total S6 spend: $0.1760** (sum of `usage.cost` in `ledger.jsonl`), within the $0.40 cap.
  - Test: CLOSEDBOOK answers $0.0085; LISTWISE rank $0.0740 + answers $0.0346; SETR_NOFILL answers $0.0204; judge 247 calls $0.0330.
  - Dev smoke: $0.0056.
- The 3-question estimate in `cost_estimate.json` was $0.20 for the test.
- The OpenRouter key's `limit_remaining` was $9.38 at the start and $6.02 at the last check (other streams were spending too). It never came near the $2 floor; checks are in `floor_checks.jsonl`.

## 4. Failed, skipped, deviations
- **First judge pass stopped.** It was a single sequential process (~12 s per call) and I stopped it after 5 verdicts so the arms could be judged in three parallel processes. Its partial output is kept in `test/judge_stopped_partial/` and is **not used**. Those 5 records were judged again in the per-arm dirs `test/judge_<ARM>/`, and the repeat calls are included in the cost.
- **One unresolved cell.** One LISTWISE cell is needs_adjudication (no parsable judge reply after 3 attempts). It is reported as bounds and has not been adjudicated.
- **Listwise prompt work.** It took 3 dev questions from `dev60` HYBRID_6000 traces, all of which parsed to full permutations on the first try, so the prompt was not changed. The dev judge pass (7 cells) only checked that the pipeline runs; nothing was tuned on it.
- **LISTWISE timing policy differs from SETR.** The ranker had no deadline and 3 HTTP attempts, while the frozen SETR selector had an 8 s deadline and 1 attempt. The aim was to measure ranking quality, and latency is descriptive only. No call failed, so there were no fallbacks.
- **Passage cut binds.** The 700-char cut applied to most passages. A longer cut or top-40 candidates were not tried, to stay within "do not iterate on test".
- **Nothing here needed a GPU.** No dense or rerank computation was redone.
- **The judge's truncation rule never fires for these records**, as for the frozen records, because neither has `trace.calls`. No S6 answer call stopped on length.

## 5. Facts for the paper
1. Without evidence, GLM-5.3-flash answers 26 of 90 answerable test questions under the same machine judge: 12/30 books, 14/30 statutes, 0/30 company reports; 20/45 English vs 6/45 Persian. It also abstains correctly on only 3 of the 6 unanswerable questions. HYBRID_6000 answers 36..37 more (sign p ≤ 4e-7). (`analysis.json`: accuracy.CLOSEDBOOK, unanswerable, paired["CLOSEDBOOK vs HYBRID_6000"])
2. Report questions cannot be answered from memory, while some English book and statute questions can (BOOK-en 9/15, LAW-en 11/15). In 8 cells the closed-book answer was correct where HYBRID's was not. (`analysis.json`: accuracy.CLOSEDBOOK, closedbook_vs_hybrid_cells)
3. A one-call RankGPT listwise reranking by GLM over HYBRID's top-20 scores 66..67/90. That is +3..+5 over HYBRID (sign p 0.36-0.66) and -5..-2 against the bge-reranker HR (69..71, p 0.36-0.82); neither difference is significant. It costs 6,774 prompt and 254 completion tokens per question in 2 calls, against 2,111 / 126 in 1 LLM call for HR. (`analysis.json`: paired, tokens_s6.LISTWISE; HR from tokens_frozen_ref)
4. Packing SETR's selected set alone (3.0 chunks and 3,056 chars on average, against 5,949 with fill) gives 63/90, exactly SETR's 63 (4 wins, 4 losses, net 0). At this budget, SETR's result comes from its set selection, not from the HYBRID fill. (`analysis.json`: paired["SETR_NOFILL vs SETR_6000"], descriptives.setr_nofill)
5. Re-packing the saved SETR selections with the frozen code reproduces all 96 saved SETR answer packs byte for byte (messages sha256), as it does for HYBRID. (`verify_repack.jsonl`)
