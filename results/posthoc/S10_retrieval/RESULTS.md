# S10 "retrieval type": dense-only vs BM25-only, with HYBRID's frozen packer and answer prompt

This work is post-hoc and exploratory. The test split had already been unsealed. Nothing was tuned on test: both arms,
both budgets and every parameter were fixed before the single test run. No GPU was used.

- Reader: GLM-5.3-flash, with the MANIFEST `together` pin.
- Judge: the pinned gpt-oss-120b (deepinfra/bf16).
- Scores are out of the 90 answerable test questions, given as bounds lo..hi over unresolved judge cells.
- All numbers below come from `results.json` (written by `report.py`) and `tables.md` (written by `tables.py`), unless
  another file is named.

## 1. What was run, exactly

**Inputs (no new embedding).**
- Chunk vectors, BM25 index and chunks: the frozen shared HybridIndex `work\indexes\_hybrid\<set>-<fp8>\`
  (`embeddings.npy`, `has_vec.npy`, `bm25\`, `chunks.jsonl`). These are exactly the directories named in
  `work\indexes\HYBRID\<set>-*\hybrid_ref.json` for all 6 sets.
- All 77 index files hash-match `test_freeze\index_manifests\_hybrid.json`, with 0 mismatches (`verify.json` →
  `index_files`).
- Query vectors: `S9_gpu\align_vectors.npz` (`q_oll`, by `qids`), from the frozen Ollama bge-m3 query path.
- Code: the frozen `ragbench` @ 77924d4, imported read-only (`s10lib.py`).

**Checks before any reader call** (`build.py verify` → `verify.json`, `verify_rows.jsonl`; 96 test questions):

| check | result |
|---|---|
| saved dense cosines (every `retrieved_units` item of HYBRID/HR at 6,000 and 15,500 with a dense score) vs `E[c]·q_oll` | 10,566 / 10,566 within 1e-5; max abs diff 6.6e-7 (the saved values are rounded to 6 decimals) |
| saved BM25 scores vs CPU recomputation | 10,232 / 10,232 within 1e-4; max abs diff 5.0e-7 |
| saved dense-channel ranks vs my dense list | 2,954 / 2,954 positions; 96 / 96 questions all-match |
| saved BM25-channel ranks vs my BM25 list | 2,923 / 2,923 positions; 96 / 96 questions |
| my fused RRF top 40 vs saved HYBRID list (`retrieved_units` and `trace.top`) | 96 / 96 identical |
| HYBRID packer on that fused list vs saved HYBRID packs: 6,000 (`_offer` + `final_pack`) | 96 / 96 byte-identical messages (sha matches both the pack file and the record; same `num_predict` and chars) |
| same check at 15,500 (`pack_evidence` over the first 40, the `lex10/control.py` path) | 96 / 96 byte-identical |

S9's RESULTS.md did not exist when I checked. Its `align_results.json` reports `hyb40_repro_equal_trace` 96/96, and the
cosine check above confirms directly that `q_oll` is the frozen query embedding (to 6-decimal rounding).

Limitation of the channel check: the saved traces hold channel ranks only for chunks that reached a fused top 40/50.
So the check covers a contiguous dense prefix of mean depth 27.4 (min 20) plus scattered deeper ranks, not all 40
positions. Every saved position matched.

**Arms** (`build.py build` → `packs/<ARM>/<qid>.json`, `packs_index.jsonl`):
- **DENSE_B**: `HybridIndex.search(channels=("dense",), k=40, depth=40, query_vector=q_oll)`, i.e. bge-m3 cosine over
  the question's pool scope.
- **BM25_B**: `search(channels=("bm25",), k=40, depth=40)`, the frozen BM25 with the B2 floor (score > 0).
- One channel through the frozen `rrf_fuse` keeps that channel's order.
- Both arms use HYBRID's packer, unchanged: B = 6,000 uses `_offer` + `final_pack` (HybridRetrieval), and B = 15,500 uses
  `pack_evidence` over the first 40 (HYBRID15500). The clipping rule is unchanged.
- Then the frozen `build_answer_messages` and `answer_num_predict`.
- Every question had a full 40-chunk list for both channels.
- One BM25_6000 pack is byte-identical to a frozen HYBRID/HR pack, so it reuses the frozen answer and verdict.
  No other duplicates.

**Smoke test** (dev, 3 questions × 4 arms; `dev_packs/`, `dev_answers.jsonl`, `dev_judge/`): 12 reader calls and 7 judge
model calls, all ok, $0.0091.

**Test run**:
- `answer.py`: 383 reader calls, all served by Together, all `finish_reason=stop`, 0 errors.
- `judge.py`: frozen `judge_multi.judge_records` with the frozen `test_v2judge_all.jsonl`, pin deepinfra/bf16,
  3 attempts, `on_malformed=adjudicate`. 383 verdicts, **0 unresolved** cells, 0 served-provider mismatches.
- Statistics: the `revision_stats.py` helpers via `S7_analysis\s7lib.py`: exact two-sided sign test as a range over
  unresolved assignments, and 95% bootstrap CIs over questions and over the 20 source-family components.
- Gold coverage:
  - pack level: S7's `t1_reach` method (`s7lib.gold_reach` on the exact reader messages);
  - top-40 list level: S7's `t5_channels` span method.

## 2. Results

### 2.1 Accuracy /90 (the 6 unanswerable are scored separately)

| arm | correct /90 | unresolved | abstain correct /6 | evidence chars | answer prompt tokens | completion tokens | LLM calls / q |
|---|---|---|---|---|---|---|---|
| **DENSE_6000** | **53** | 0 | 6 | 5,937 | 2,110 | 127 | 1 |
| **BM25_6000** | **48** | 0 | 6 | 5,953 | 2,085 | 118 | 1 |
| HYBRID_6000 | 62..63 | 1 | 6 | 5,953 | 2,097 | 130 | 1 |
| HR_6000 | 69..71 | 2 | 6 | 5,955 | 2,116 | 132 | 1 |
| **DENSE_15500** | **63** | 0 | 6 | 15,437 | 5,143 | 140 | 1 |
| **BM25_15500** | **61** | 0 | 6 | 15,451 | 5,052 | 128 | 1 |
| HYBRID_15500 | 68 | 0 | 6 | 15,448 | 5,113 | 137 | 1 |
| HR_15500 | 69..70 | 1 | 6 | 15,440 | 5,165 | 140 | 1 |

LEX5B rows are in `tables.md`.
- Query-time cost: one LLM call per question (the answer) for every arm in this table.
- DENSE also needs one bge-m3 query embedding (local, not an LLM call). BM25 needs none.
- HR also needs a local reranker pass.
- No latency was measured: all retrieval here was replayed offline.

### 2.2 Paired comparisons

All 12 comparisons were declared before looking at the results, so the comparison family has 12 members. Bonferroni
α = 0.05/12 = 0.0042. Net = a − b over 90.

| comparison | a | b | net | a better / b better | sign p | 95% CI pp (questions) | 95% CI pp (20 components) |
|---|---|---|---|---|---|---|---|
| DENSE_6000 vs BM25_6000 | 53 | 48 | +5 | 18 / 13 | 0.4731 | [-6.7, 17.8] | [-5.6, 13.4] |
| DENSE_6000 vs HYBRID_6000 | 53 | 62..63 | -10..-9 | 7 / 16 | 0.0639-0.0931 | [-21.1, 0.0] | [-25.0, -1.1] |
| BM25_6000 vs HYBRID_6000 | 48 | 62..63 | -15..-14 | 2 / 16 | **0.0007-0.0013** | [-25.6, -6.7] | [-24.7, -7.1] |
| DENSE_6000 vs HR_6000 | 53 | 69..71 | -18..-16 | 4 / 20 | **0.0005-0.0015** | [-30.0, -7.8] | [-33.8, -10.1] |
| BM25_6000 vs HR_6000 | 48 | 69..71 | -23..-21 | 2 / 23 | **<0.0001** | [-35.6, -13.3] | [-33.3, -15.4] |
| DENSE_15500 vs BM25_15500 | 63 | 61 | +2 | 7 / 5 | 0.7744 | [-5.6, 10.0] | [-4.9, 11.3] |
| DENSE_15500 vs HYBRID_15500 | 63 | 68 | -5 | 4 / 9 | 0.2668 | [-13.3, 2.2] | [-12.6, 3.9] |
| BM25_15500 vs HYBRID_15500 | 61 | 68 | -7 | 1 / 8 | 0.0391 | [-14.4, -2.2] | [-13.5, -1.8] |
| DENSE_15500 vs HR_15500 | 63 | 69..70 | -7..-6 | 5 / 12 | 0.1435-0.2379 | [-16.7, 2.2] | [-16.5, 5.8] |
| BM25_15500 vs HR_15500 | 61 | 69..70 | -9..-8 | 3 / 12 | 0.0352-0.0768 | [-18.9, 0.0] | [-15.5, -1.4] |
| DENSE_15500 vs DENSE_6000 | 63 | 53 | +10 | 15 / 5 | 0.0414 | [2.2, 21.1] | [2.8, 23.9] |
| BM25_15500 vs BM25_6000 | 61 | 48 | +13 | 16 / 3 | 0.0044 | [5.6, 23.3] | [5.8, 20.4] |

Bold p-values pass Bonferroni over the family of 12.

### 2.3 Breakdowns (correct; each tier n=30, each language n=45, each genre n=30)

| arm | T1 | T2 | T3 | fa | en | reports (RPT) | books (BOOK) | statutes (LAW) |
|---|---|---|---|---|---|---|---|---|
| DENSE_6000 | 24 | 15 | 14 | 27 | 26 | 16 | 17 | 20 |
| BM25_6000 | 26 | 13 | 9 | 24 | 24 | 17 | 14 | 17 |
| HYBRID_6000 | 28 | 20..21 | 14 | 35..36 | 27 | 21..22 | 19 | 22 |
| HR_6000 | 28 | 23..24 | 18..19 | 36 | 33..35 | 26..27 | 20 | 23..24 |
| DENSE_15500 | 27 | 20 | 16 | 34 | 29 | 23 | 16 | 24 |
| BM25_15500 | 27 | 20 | 14 | 33 | 28 | 22 | 19 | 20 |
| HYBRID_15500 | 28 | 22 | 18 | 35 | 33 | 24 | 20 | 24 |
| HR_15500 | 28 | 21..22 | 20 | 38 | 31..32 | 24..25 | 23 | 22 |

The per-pool (genre × language) and per-subtype tables are in `tables.md`. The question files carry `subtype` for all
90 answerable questions. `integration_operation` is free text, filled for 54, and so is not tabulated. Selected subtypes
(n): DENSE_6000 / BM25_6000 / HYBRID_6000.
- T1_fact (27): 22 / 24 / 25
- T2_comparison_derivation (13): 6 / 5 / 8
- T2_rule_exception (7): 5 / 3 / 5
- T3_cross_time_version (11): 10 / 6 / 9
- T3_cross_subject (9): 2 / 1 / 2
- T3_cross_doc_bridge (7): 1 / 2 / 2

### 2.4 Gold-bundle coverage (complete / partial / none, of 90)

| arm | delivered pack: all | T1 | T2 | T3 | fa | en | top-40 list: all | T1 | T2 | T3 | fa | en |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DENSE_6000 | 47/36/7 | 25/0/5 | 11/19/0 | 11/17/2 | 26/13/6 | 21/23/1 | 71/17/2 | 28/0/2 | 25/5/0 | 18/12/0 | 38/6/1 | 33/11/1 |
| BM25_6000 | 45/32/13 | 26/0/4 | 11/15/4 | 8/17/5 | 25/15/5 | 20/17/8 | 66/20/4 | 30/0/0 | 21/8/1 | 15/12/3 | 37/6/2 | 29/14/2 |
| HYBRID_6000 | 57/26/7 | 27/0/3 | 18/11/1 | 12/15/3 | 33/8/4 | 24/18/3 | 72/17/1 | 29/0/1 | 26/4/0 | 17/13/0 | 39/6/0 | 33/11/1 |
| HR_6000 | 63/23/4 | 29/0/1 | 19/10/1 | 15/13/2 | 34/11/0 | 29/12/4 | 75/14/1 | 29/0/1 | 27/3/0 | 19/11/0 | 39/6/0 | 36/8/1 |
| DENSE_15500 | 62/24/4 | 27/0/3 | 20/10/0 | 15/14/1 | 35/7/3 | 27/17/1 | (same list as 6000) | | | | | |
| BM25_15500 | 59/23/8 | 28/0/2 | 18/10/2 | 13/13/4 | 33/9/3 | 26/14/5 | (same list as 6000) | | | | | |
| HYBRID_15500 | 65/22/3 | 28/0/2 | 21/9/0 | 16/13/1 | 36/7/2 | 29/15/1 | (same list) | | | | | |
| HR_15500 | 70/18/2 | 29/0/1 | 24/6/0 | 17/12/1 | 38/7/0 | 32/11/2 | (same list) | | | | | |

By genre, see `tables.md`.

## 3. Where each wins

These are paired, resolved, discordant questions between DENSE and BM25 at the same budget; the full question-level
list with glosses is in `tables.md`. None of these subgroup splits is significant on its own (the counts are small),
so they are descriptive.

- **BM25 (lexical) wins on single exact facts.**
  - It is the only channel that puts a complete gold bundle in its top 40 for all 30 T1 questions (dense: 28).
  - At 6,000: T1 26 vs 24 correct; T1 discordant pairs 4 for BM25, 2 for dense. Pack-level T1 "none" is 4 for BM25 vs 5
    for dense.
  - Typical BM25-only wins are Persian report and statute look-ups keyed on a name, year or term: Snapp acceptance
    seconds in 1404, Irancell village coverage in 1401, the 1311 Commercial Code bankruptcy-after-death rule. On these,
    dense delivered no gold item at all.
  - BM25 also wins on some English book questions that hinge on a rare token (Tocqueville's 1830 figures, Darwin →
    Malthus). At 15,500 the only genre where BM25 leads is books (0 dense-only vs 3 BM25-only).
- **Dense (semantic) wins on paraphrased, multi-part and cross-time questions.**
  - At 6,000: T3 14 vs 9 correct, discordant 8 vs 3 (sign p 0.23).
  - T3_cross_time_version: 10 vs 6 (discordant 5 vs 1, p 0.22).
  - T2_rule_exception: 5 vs 3.
  - Statutes: 20 vs 17 at 6,000 and 24 vs 20 at 15,500 (discordant 4 vs 0).
  - Its top-40 list holds more complete T2/T3 bundles (T2 25 vs 21, T3 18 vs 15) and more English ones (33 vs 29).
  - Typical dense-only wins are scenario-style questions whose wording does not match the text: a barber's expired
    licence and Instagram advertising, a DMA gatekeeper that fails to notify. Others are "expected vs actual" report
    questions (Mowi guidance vs harvest, ArcelorMittal decarbonisation capex, Digikala 1399 vs 1403).
- **Neither single channel matches the hybrid.**
  - The channels are complementary: at 6,000 they are discordant on 31 of 90 questions. An oracle that takes either
    channel's correct answer reaches 66, above HYBRID_6000's 62..63; at 15,500 the oracle reaches 68, the same as
    HYBRID_15500.
  - RRF fusion gets most of that union: DENSE_6000 is −9..−10 and BM25_6000 −14..−15 vs HYBRID_6000.
  - The loss is in **ranking within the pack, not recall**. Dense's top-40 list is almost as complete as HYBRID's
    (71 vs 72 complete bundles), but its 6,000-char pack is much less so (47 vs 57). Fusion lifts the chunks that both
    channels agree on into the first few pack slots.
- **Budget closes most of the gap.** At 15,500 the single channels are 63 and 61, vs HYBRID 68 and HR 69..70. Going
  6,000 → 15,500 adds +10 (dense) and +13 (BM25), vs +5..+6 for HYBRID; BM25's p is 0.0044, just above α.
  - At 15,500 dense vs BM25 is a near tie (+2, p 0.77).
  - Only BM25_15500 vs HYBRID_15500 stays nominally significant (p 0.039), and it does not pass Bonferroni.
- **By language.** Persian loses the most from dropping fusion: fa 27 dense / 24 BM25 vs 35..36 HYBRID at 6,000.
  English barely loses: 26 / 24 vs 27.

## 4. Cost

From `ledger*.jsonl`. The cap was raised by the coordinator from $0.25 to $0.45 mid-run.
- **Total spent: $0.2763.**
- Reader: 395 calls, $0.2321 (dev smoke 12 calls, $0.0078).
- Judge: 334 model calls, $0.0443 (dev 7 calls, $0.0013).
- The key's `limit_remaining` was checked before every batch of ≤ 50 calls. It never fell below $3.87 (floor $2.00).
- No GPU time: none was used.

## 5. Failed, skipped, deviations

- The first reader pass stopped at a self-imposed $0.20 per-call guard, left to keep room for judging under the original
  $0.25 cap. 22 BM25_15500 cells were unanswered at that point. After the cap was raised, a second pass answered them
  and the 24 unanswerable cells. The packs, messages and parameters were the same; only the call time differs.
- The judge shards stopped once at the old cap (`judge_*.log`). They resumed with skip-done (`judge2_*.log`).
  No verdict was recomputed.
- The dense channel check covers only the saved ranks (see §1). The full dense list cannot be checked against saved
  data; the cosine-level match makes this moot in practice.
- The `bm25_full40_known` field in `verify.json` is an uninformative check (0/96, because saved BM25 ranks are not
  contiguous to 40). Ignore it.
- Nothing needed the GPU.

## 6. Facts for the paper

1. At 6,000 chars, dense-only scores 53/90 and BM25-only 48/90, vs 62..63 for HYBRID and 69..71 for HR. Both single
   channels lose to HR, and BM25-only also loses to HYBRID, under Bonferroni over 12 comparisons (p ≤ 0.0015).
   (`results.json` → `table`, `compare`)
2. At 15,500 chars the single channels nearly catch up: dense 63, BM25 61, HYBRID 68, HR 69..70. Dense vs BM25 is +2
   (p 0.77). (`results.json`)
3. Lexical search shines on single exact facts. BM25's top 40 holds a complete gold bundle for 30/30 T1 questions (dense
   28/30), and at 6,000 it gets 26 vs 24 T1 questions right. (`tables.md`, list coverage and tier tables)
4. Semantic search shines on paraphrased, multi-part and cross-time questions. At 6,000, dense gets T3 14 vs 9 and
   T3 cross-time/version 10/11 vs 6/11, and statutes 20 vs 17 (24 vs 20 at 15,500). (`tables.md`)
5. Fusion's gain is ordering, not recall. Dense-only's top 40 contains a complete gold bundle as often as HYBRID's
   (71 vs 72 of 90), but its 6,000-char pack does so less often (47 vs 57). (`results.json` → `reach`, `list_reach`)
6. The two channels disagree on 31/90 questions at 6,000. An either-channel oracle reaches 66/90, above HYBRID_6000.
   (`results.json` → `wins.6000`)

Files: `s10lib.py`, `build.py`, `answer.py`, `judge.py`, `report.py`, `tables.py`, `verify.json`, `verify_rows.jsonl`,
`packs_index.jsonl`, `packs/`, `answers.jsonl`, `judge/<ARM>/`, `results.json`, `tables.md`, `ledger*.jsonl`, `*.log`.
