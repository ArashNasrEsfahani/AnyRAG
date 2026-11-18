# S11 "query rewriting": HyDE and multi-query on top of the frozen HYBRID at 6,000 characters

Post-hoc and exploratory. The prompts were written once and checked on 5 dev questions without changes, then run once on
the 96 test questions. Reader: GLM-5.3-flash with the pinned provider and params. Judge: the pinned judge
(gpt-oss-120b, deepinfra/bf16), run through `ragbench.tools.judge_multi` with run_test.py's arguments. Scores are strict
machine verdicts on the 90 answerable questions. `a..b` gives the bounds over unresolved cells.

## 1. What was run

**Pipeline (code: `s11lib.py`, `rewrite.py`, `encode.py`, `run.py`, `judge.py`, `analyze.py`).** Everything except the
query side is the frozen HYBRID_6000: the same `HybridIndex` (BM25 + Ollama bge-m3 dense, stored chunk vectors), scope
masks, per-channel depth 40, the frozen `rrf_fuse` (k=60), 40 candidates, the B2/B3 offer rule plus `pack_evidence`
at 6,000 characters, `build_answer_messages`, `answer_num_predict` and the GLM pin (S9's `glm()` copied unchanged).
- **HYDE_6000.** GLM writes one hypothetical passage, 60–120 words, in the question's language and in the style of the
  pool's genre (annual report, book, or law). The prompt is `HYDE_PROMPT` in `s11lib.py`. The BM25 channel uses the
  original question. The dense channel uses the passage's embedding (`hx.search(question, query_vector=passage
  vector)`). Then RRF and the frozen packer. Mean passage length is 649 characters.
- **MQ_6000.** GLM writes 3 paraphrases or sub-queries in the question's language (`MQ_PROMPT`). BM25 and dense
  top-40 lists come from the original question and from each rewrite, so 8 lists in all. They are fused by `rrf_fuse`
  (one contribution per list), and the top 40 go to the frozen packer. All 96 questions got exactly 3 rewrites.
- **Order of steps.** (1) Rewrites, API only (`rewrites_{dev,test}.jsonl`). (2) One GPU phase per split: the original
  question, the HyDE passage and the 3 MQ rewrites, each embedded alone with
  `OllamaClient.embed([normalize(text)])`, the same call `HybridIndex.query_vector` makes. Cached to
  `vectors_{dev,test}.npz`. (3) Retrieval, packing and answering on CPU plus API (`run.py`). (4) Judging.
  The gpu.lock rule was lifted mid-stream by the updated COMMON.md, so no lock was taken. VRAM was checked before each
  phase. The Ollama bge-m3 model is shared with other streams and stayed loaded under Ollama's control.
- **Dev (5 questions, prompts not changed after).** V2-dev-RPT-fa-T2-001, RPT-en-T3-002, BOOK-fa-T3-001, BOOK-en-T3-002
  and LAW-fa-T2-002. All rewrites parsed, and the pipeline ran end to end. Dev answers were not judged.

**Reproduction checks (`repro_test.jsonl`, `gpu_phases.jsonl`).**
- The re-implemented HYBRID_6000 uses the re-encoded test question as the query vector. It gives answer messages
  **byte-identical** to the saved `test_run/packs/HYBRID/<qid>__<setting>.json` on **96/96** questions: the JSON is equal
  and the sha256 is equal.
- This file's own per-channel list code plus `rrf_fuse`, which is the MQ code path, gives the same 40 candidates in the
  same order as the frozen `hx.search` on 96/96 questions.
- Re-encoded original questions compared with S9's re-encoding (`S9_gpu/align_vectors.npz`, same Ollama path): 75/96
  bit-identical on test, and the rest differ at cosine ≥ 0.9999999. The 5 dev questions are 5/5 identical. This is GPU
  float nondeterminism under concurrent load. It changed no pack (see the 96/96 above).

## 2. Results (`results.json`, `tables.md`)

### Accuracy (90 answerable) and abstention (6 unanswerable)

| arm | correct /90 | unanswerable correct /6 | complete gold bundle /90 | mean query LLM tokens / question |
|---|---|---|---|---|
| HYDE_6000 | 61 | 6 | 51 | 2,551 |
| MQ_6000 | 58..59 | 6 | 54 | 2,438 |
| HYBRID_6000 (frozen) | 62..63 | 6 | 57 | 2,216 |
| HR_6000 (frozen) | 69..71 | 6 | 63 | 2,237 |
| HYBRID_15500 (frozen) | 68 | 6 | 65 | 5,236 |

- Unresolved cells: MQ_6000 has 1 (V2-test-LAW-fa-T2-002, needs_adjudication after 3 judge retries). HYDE_6000 has 0.
- Abstention: HYDE_6000 answered NOT_FOUND on 5 of the 6 unanswerable questions, and the judge accepted the sixth
  answer as well. MQ_6000 answered NOT_FOUND on 6/6.
- No system failures. All 192 answer calls and all 192 rewrite calls succeeded with finish=stop.

### Paired comparisons (90 answerable; exact two-sided sign test as a range over unresolved cells; bootstrap 10,000, seed 0)

| comparison | net | better/worse | sign p | 95% CI pp (questions) | 95% CI pp (20 components) |
|---|---|---|---|---|---|
| HYDE_6000 vs HYBRID_6000 | -2..-1 | 8/9 | 0.815-1.000 | [-11.1, 7.8] | [-17.6, 8.9] |
| HYDE_6000 vs HR_6000 | -10..-8 | 5/13 | 0.041-0.096 | [-21.1, 0.0] | [-26.4, 0.0] |
| HYDE_6000 vs HYBRID_15500 | -7 | 4/11 | 0.118 | [-16.7, 0.0] | [-20.0, 1.4] |
| MQ_6000 vs HYBRID_6000 | -5..-3 | 6/9 | 0.332-0.607 | [-14.4, 4.4] | [-14.9, 6.3] |
| MQ_6000 vs HR_6000 | -13..-10 | 2/12 | 0.002-0.013 | [-23.3, -3.3] | [-21.2, -6.5] |
| MQ_6000 vs HYBRID_15500 | -10..-9 | 3/12 | 0.021-0.035 | [-20.0, -2.2] | [-18.4, -1.5] |
| HYDE_6000 vs MQ_6000 | +2..+3 | 9/7 | 0.629-0.804 | [-6.7, 12.2] | [-11.8, 11.8] |

### Breakdowns (correct, bounds)

| group | n | HYDE_6000 | MQ_6000 | HYBRID_6000 | HR_6000 | HYBRID_15500 | HYDE−HYBRID | MQ−HYBRID |
|---|---|---|---|---|---|---|---|---|
| T1 | 30 | 27 | 26 | 28 | 28 | 28 | -1 | -2 |
| T2 | 30 | 18 | 17..18 | 20..21 | 23..24 | 22 | -3..-2 | -4..-2 |
| T3 | 30 | 16 | 15 | 14 | 18..19 | 18 | +2 | +1 |
| fa | 45 | 31 | 29..30 | 35..36 | 36 | 35 | -5..-4 | -7..-5 |
| en | 45 | 30 | 29 | 27 | 33..35 | 33 | +3 | +2 |
| BOOK | 30 | 19 | 16 | 19 | 20 | 20 | +0 | -3 |
| LAW | 30 | 24 | 19..20 | 22 | 23..24 | 24 | +2 | -3..-2 |
| RPT | 30 | 18 | 23 | 21..22 | 26..27 | 24 | -4..-3 | +1..+2 |

### Gold-bundle coverage (S7's `gold_reach` on the exact reader messages; complete / partial / none)

| arm | all 90 | T1 | T2 | T3 | fa | en |
|---|---|---|---|---|---|---|
| HYDE_6000 | 51/31/8 | 26/0/4 | 14/14/2 | 11/17/2 | 26/13/6 | 25/18/2 |
| MQ_6000 | 54/27/9 | 26/0/4 | 16/12/2 | 12/15/3 | 28/10/7 | 26/17/2 |
| HYBRID_6000 | 57/26/7 | 27/0/3 | 18/11/1 | 12/15/3 | 33/8/4 | 24/18/3 |
| HR_6000 | 63/23/4 | 29/0/1 | 19/10/1 | 15/13/2 | 34/11/0 | 29/12/4 |
| HYBRID_15500 | 65/22/3 | 28/0/2 | 21/9/0 | 16/13/1 | 36/7/2 | 29/15/1 |

The frozen rows reproduce S7's `reach_tables.md` exactly. Compared with HYBRID_6000, HYDE gains a complete bundle on 2
questions and loses one on 8. MQ gains on 4 and loses on 7. The HYDE and MQ packs differ from the HYBRID_6000 pack on
95 of 96 questions.

### Query-time cost per question (96 questions, means, from `ledger.jsonl`)

| arm | LLM calls | rewrite prompt + completion tokens (reasoning) | answer prompt + completion | total tokens | $/question | embeds/question |
|---|---|---|---|---|---|---|
| HYDE_6000 | 2 | 150 + 196 (7) | 2,087 + 119 | 2,551 | 0.00047 | 1 (passage)* |
| MQ_6000 | 2 | 153 + 70 (0) | 2,101 + 114 | 2,438 | 0.00041 | 4 |
| HYBRID_6000 | 1 | – | – | 2,216 | – | 1 |

\*HYDE as specified uses only the passage vector at query time. The original question's vector is not used.
Latency is descriptive only. The rewrite call adds one GLM round trip (the `wall_ms` of each call is in `ledger.jsonl`).

### GPU wall time (`gpu_phases.jsonl`)

| phase | texts embedded | wall | VRAM before → after (nvidia-smi, whole card) |
|---|---|---|---|
| dev encode | 25 (5 questions) | 14.3 s | 787 → 787 MiB |
| test encode | 480 (96 × [question, passage, 3 rewrites]) | 557.2 s | 3,754 → 793 MiB |

The test phase ran while other streams were using the same Ollama bge-m3 instance and the GPU (90% utilization at the
time). That works out to about 1.2 s per embed call, against about 0.06 s per question in S9's quieter align run. Treat
this wall time as an upper bound under contention, not as the method's intrinsic cost. Query-time GPU work is 1 embed
per question for HYDE (same as HYBRID) and 4 for MQ.

## 3. Cost

| item | USD |
|---|---|
| GLM rewrites + answers (dev 5 q + test 96 q), `ledger.jsonl` | 0.0895 |
| judge (192 test cells), `judge/*/test/judge_ledger_*.jsonl` | 0.0225 |
| **total** | **0.112** (cap 0.30) |

The key's `limit_remaining` was checked before each batch of at most 45 calls: every 40 rewrite calls, every 10
questions in the answer run, and before each judge batch of 45. The lowest logged reading was $4.10 (rewrite log), above
the $2.00 floor. The answer and judge runs only print a reading when it falls below the floor, and none did.

## 4. Failed or skipped

- No call failed. One MQ cell (LAW-fa-T2-002) is unresolved by the judge and is carried as bounds.
- Dev answers were not judged. With 5 questions they could not have guided a choice, and no prompt was changed after dev.
- I did not compare with frozen *query vectors* directly, because the test run did not save them. The check was done
  on the end product instead: the 96/96 byte-identical HYBRID_6000 packs, plus the comparison with S9's re-encoding.
- The HyDE variant uses one passage and does not average it with the question vector, as the brief specified. The
  original paper averages several passages with the query. That variant was not tried.

## 5. Facts for the paper

- Neither query-rewriting method improved the frozen HYBRID at 6,000 characters. HyDE scored 61/90 and multi-query
  58..59/90, against 62..63 for HYBRID_6000. The paired nets are -2..-1 (sign p ≥ 0.815) and -5..-3 (p ≥ 0.332).
  (`results.json` totals and paired; `tables.md`)
- Reranking beats rewriting on the same pipeline. HR_6000 (69..71) beats multi-query by 10–13 questions
  (2 better / 12 worse, p 0.002–0.013, component CI [-21.2, -6.5] pp) and HyDE by 8–10 (p 0.041–0.096).
  (`results.json` paired)
- Rewriting lowers gold-evidence delivery. Complete gold bundles reach the reader on 51 (HyDE) and 54 (multi-query) of
  90 questions, against 57 for HYBRID_6000 and 63 for HR_6000. The losses are concentrated in Persian (fa complete:
  26 and 28 against 33). (`results.json` coverage)
- Both methods hurt Persian and slightly help English. Against HYBRID_6000 the nets are fa -5..-4 (HyDE) and -7..-5
  (MQ), and en +3 and +2. Both are +1..+2 on T3 and negative on T1/T2. (`results.json` breakdown)
- The cost is one extra GLM call per question: about +220 to +335 query tokens (2,438–2,551 against 2,216) and about
  $0.0004–0.0005 per question. Multi-query also needs 4 query embeddings instead of 1. (`results.json` tokens,
  `ledger.jsonl`)
- The re-implementation is exact. Re-encoding the test questions through the frozen Ollama path and running the
  frozen search, offer and pack code reproduces all 96 saved HYBRID_6000 answer packs byte for byte.
  (`repro_test.jsonl`)
