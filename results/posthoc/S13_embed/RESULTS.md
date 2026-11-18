# S13: a second embedding model (multilingual-e5-large) in place of bge-m3. Post-hoc, exploratory

All numbers below come from `results.json` (written by `report.py`), `retrieve_checks.json`, `gpu_phase.json` and the
ledgers in this folder. The test split was unsealed before this work. Each arm ran once on test; nothing was tuned.

## 1. What was run

**Model.** `intfloat/multilingual-e5-large`, HF commit `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3`. It was not in any
local HF cache, so it was downloaded on 2026-11-10 into `work/models/hf` (safetensors and tokenizer only, 2.24 GB). It
loaded without problems, so no fallback model was needed. Inference: fp16 on cuda, mean pooling over the attention
mask, L2 normalisation, max_length 512 (the model's limit). Prefixes as the model card requires:
- chunks: `"passage: " + index_text(chunk)`. This is the same `"<breadcrumb>\n<chunk>"` text that bge-m3 embedded for
  the frozen HybridIndex.
- questions: `"query: " + normalize(question text)`. This is the same text `HybridIndex.query_vector` embeds.

**GPU phase** (`embed.py`). It embedded every chunk of the six test sets (58,218 frozen 1,000-character chunks:
BOOK_en 9,661, BOOK_fa 8,905, C 6,921, D_en 25,685, LAW_en 5,427, LAW_fa 1,619) and all 156 questions (96 test and 60
dev). Vectors are cached in `vectors/`. The model was unloaded at the end of the phase (nvidia-smi went from 3,754 to
926 MiB). By the time the GPU phase started, the coordinator had replaced the lock rule with a VRAM guard, so no
`gpu.lock` was taken. nvidia-smi was checked before loading the model.

| GPU phase | wall time | detail |
|---|---|---|
| probe (200 chunks) | 22.0 s | 31.3 chunks/s at batch 16 (peak 1,209 MB); 37.8 chunks/s at batch 32 (peak 1,339 MB) |
| encode (all) | **1,058.8 s** (17.6 min) | model load 7.3 s; 156 questions 1.5 s; chunks 1,049.6 s (about 55 chunks/s at batch 32, GPU shared with other streams); peak allocated 1,470 MB; no OOM |

The 512-token limit truncated 65 of the 58,218 chunk texts (27 in C, 38 in D_en) and no question. The download took
about 5 minutes and is not counted in GPU time.

**Retrieval and packs** (`retrieve.py`, CPU only). The frozen `HybridIndex` (same chunks, same BM25, same pool doc
scope, k = 40, depth 40) is used with only the dense matrix and the query vector replaced:
- **E5-DENSE**: `search(channels=("dense",))`, e5 cosine ranking only.
- **E5-HYBRID**: BM25 + e5, equal-weight RRF with k = 60. This is the frozen HYBRID with the dense model swapped.

Both arms go through the frozen HYBRID_6000 packer (`HybridRetrieval._offer` + `final_pack` at 6,000 characters, via
S2a's `packs.pack_offer`) and the frozen `build_answer_messages`.

**Correctness checks** (`retrieve_checks.json`). With S9's saved Ollama bge-m3 query vectors, the same harness
reproduces the frozen HYBRID_6000 exactly:
- fused list equals the trace on 96/96 questions;
- answer-message sha256 equals the frozen pack on 96/96;
- setting equals the frozen setting on 96/96;
- BM25 ranks are consistent between the bge-m3 and e5 hybrids on 96/96.

The harness's bge-m3 dense-only top 40 equals S10's DENSE_6000 `list40` on 96/96. No e5 pack is byte-identical to a
frozen pack (0/96 for each arm), so every cell was answered fresh.

**Reader and judge** (`llm.py`). GLM-5.3-flash with the MANIFEST provider pin and attempts, using the exact pack
messages and num_predict. The pinned judge (`judge_pin.json`: gpt-oss-120b, deepinfra/bf16, 3 attempts, adjudicate on
malformed) runs through the frozen `judge_multi.judge_records`. Smoke test first: 5 dev questions × 2 arms, all
answered and judged without errors (`answers_dev.jsonl`, `judge/dev/`). Cost estimate on 3 test questions, then one
test run: 192 answer calls (0 errors, all `done_reason=stop`) and 192 judged cells. One cell needs adjudication
(E5_DENSE, V2-test-RPT-en-T3-005) and is reported as a bound.

**Comparators.**
- HYBRID_6000 and HR_6000: frozen machine verdicts through `s7lib`.
- S10 DENSE_6000 (bge-m3 dense only, at 6000): S10 had finished, so it is scored read-only from S10's answers and judge
  shards using S10's identity rule (`s10_hook.py`). It reproduces S10's own 53/53.

## 2. Results (90 answerable test questions, machine strict verdicts; bounds over unresolved cells)

| arm | dense model | correct /90 | unresolved | abstention correct /6 | mean pack chars | answer prompt tokens (mean) | LLM calls/q |
|---|---|---|---|---|---|---|---|
| **E5-DENSE** | e5-large | **54..55** | 1 | 6/6 | 5,923 | 2,091 | 1 |
| **E5-HYBRID** | e5-large (+BM25) | **58** | 0 | 6/6 | 5,923 | 2,079 | 1 |
| S10 DENSE_6000 | bge-m3 | 53 | 0 | 6/6 | 5,937 (S10) | 2,110 (S10) | 1 |
| HYBRID_6000 | bge-m3 (+BM25) | 62..63 | 1 | 6/6 | | | 1 |
| HR_6000 | bge-m3 (+BM25) + reranker | 69..71 | 2 | 6/6 | | | 1 |

Query-time tokens per question (prompt + completion of the single answer call): E5-DENSE 2,200, E5-HYBRID 2,193.

### Paired comparisons (family of 6 tests; Bonferroni alpha = 0.05/6 = 0.0083)
Columns: net = wins minus losses (bounds); exact two-sided sign test p (range over unresolved cells); 95% bootstrap
CIs in percentage points, resampled over questions and over the 20 connected source components; resolved better/worse
counts; net within each language.

| comparison | net | sign p | CI pp (questions) | CI pp (components) | better/worse | net fa | net en |
|---|---|---|---|---|---|---|---|
| E5-DENSE vs HYBRID_6000 | -9..-7 | 0.049..0.167 | [-18.9, 1.1] | [-16.0, 0.0] | 4/13 | -7..-6 | -2..-1 |
| E5-HYBRID vs HYBRID_6000 | -5..-4 | 0.180..0.344 | [-12.2, 2.2] | [-11.6, 2.7] | 2/7 | -5..-4 | 0 |
| E5-DENSE vs HR_6000 | -17..-14 | 0.0009..0.0066 | [-28.9, -5.6] | [-26.7, -7.8] | 4/19 | -7 | -10..-7 |
| E5-HYBRID vs HR_6000 | -13..-11 | 0.0044..0.0127 | [-23.3, -3.3] | [-22.9, -4.7] | 3/14 | -5 | -8..-6 |
| E5-HYBRID vs E5-DENSE | +3..+4 | 0.424..0.607 | [-5.6, 12.2] | [-2.9, 10.4] | 9/5 | +2 | +1..+2 |
| E5-DENSE vs S10 DENSE_6000 (bge-m3) | +1..+2 | 0.815..1.0 | [-7.8, 11.1] | [-7.0, 14.0] | 10/8 | +2 | -1..0 |

Only E5-DENSE vs HR_6000 is below the Bonferroni threshold over its whole range. E5-HYBRID vs HR_6000 crosses it
(0.0044..0.0127). Both of those comparisons also differ by the reranker, so they are not tests of the embedding model.

The like-for-like model swaps are E5-HYBRID vs HYBRID_6000 and E5-DENSE vs bge-m3 DENSE. Neither difference is
significant: -4..-5 and +1..+2 questions.

### By tier, language and genre (correct, lower..upper / n)

| arm | T1 /30 | T2 /30 | T3 /30 | fa /45 | en /45 | BOOK /30 | LAW /30 | RPT /30 |
|---|---|---|---|---|---|---|---|---|
| E5-DENSE | 25 | 18 | 11..12 | 29 | 25..26 | 15 | 18 | 21..22 |
| E5-HYBRID | 27 | 18 | 13 | 31 | 27 | 15 | 22 | 21 |
| S10 DENSE_6000 (bge-m3) | 24 | 15 | 14 | 27 | 26 | 17 | 20 | 16 |
| HYBRID_6000 | 28 | 20..21 | 14 | 35..36 | 27 | 19 | 22 | 21..22 |
| HR_6000 | 28 | 23..24 | 18..19 | 36 | 33..35 | 20 | 23..24 | 26..27 |

Per pool (in `results.json` → `groups`): E5-HYBRID vs HYBRID_6000 loses on BOOK-en (9 vs 11), BOOK-fa (6 vs 8), LAW-fa
(13 vs 14) and RPT-fa (12 vs 13..14). It gains on LAW-en (9 vs 8) and RPT-en (9 vs 8).

### Gold-bundle coverage of the reader pack (S7's `gold_reach`: >= 90% of each gold quote verbatim in a same-document item)

| arm | complete | partial | none | complete fa /45 | complete en /45 | correct with complete | miss with complete |
|---|---|---|---|---|---|---|---|
| E5-DENSE | 46 | 30 | 14 | 27 | 19 | 43 | 3 |
| E5-HYBRID | 53 | 29 | 8 | 30 | 23 | 50 | 3 |
| S10 DENSE_6000 (bge-m3) | 47 | 36 | 7 | 26 | 21 | 43 | 4 |
| HYBRID_6000 | 57 | 26 | 7 | 33 | 24 | 54 | 3 |
| HR_6000 | 63 | 23 | 4 | 34 | 29 | 60 | 3 |

### Recall@40 of complete gold bundles (top-40 list, before packing)
A gold item counts as present when chunks of the same document in the top 40 cover >= 90% of its character span. A
bundle counts as complete when all of its items are present.

Consistency check: a pack is a subset of its top 40. On 0 questions was a pack "complete" (S7's text measure) while
its list was not (span measure), for HYBRID_6000, E5-DENSE and E5-HYBRID.

| list | all /90 | fa /45 | en /45 | T1 | T2 | T3 |
|---|---|---|---|---|---|---|
| bge-m3 dense | 71 | 38 | 33 | 28 | 25 | 18 |
| e5 dense | 69 | 40 | 29 | 29 | 25 | 15 |
| bge-m3 hybrid (frozen HYBRID) | 72 | 39 | 33 | 29 | 26 | 17 |
| e5 hybrid | 72 | 40 | 32 | 29 | 26 | 17 |

Paired discordant questions (e5 only / bge-m3 only):

| pair | fa | en | all |
|---|---|---|---|
| dense | 2/0 | 0/4 | 2/4, sign p = 0.69 |
| hybrid | 1/0 | 1/2 | 2/2 |

Mean top-40 overlap with bge-m3 over 96 questions:

| channel | all | fa | en |
|---|---|---|---|
| dense | 22.8 | 24.0 | 21.6 |
| hybrid | 31.0 | 31.5 | 30.5 |

## 3. Cost
$0.0920 in total, against a cap of $0.25:
- reader: 202 GLM calls, $0.0716 (10 dev smoke, 6 for the 3-question estimate, 186 for the rest of test);
- judge: 156 model calls, $0.0204. The remaining verdicts came from the rule path or the cache.

Key `limit_remaining` was $4.02 at the first check and $3.49 at the last, always above the $2.00 floor
(`money_checks.jsonl`).

## 4. Failed, skipped, caveats
- Nothing failed. One judge cell needs adjudication (E5-DENSE, V2-test-RPT-en-T3-005) and is counted as a 0..1 bound.
- No lock was used for the GPU phase: the author's rule change, relayed by the coordinator, replaced it with the VRAM
  guard before the phase ran. The peak stayed at 1.47 GB, under the 2.5 GB limit.
- e5 truncates at 512 tokens; bge-m3 (Ollama) used num_ctx 8192. This affected 65 chunks (0.11%).
- Each arm was run once. Reader nondeterminism is not separated from the retrieval effect, so a gap of a few
  questions is within what a re-run could move.
- HR_6000 adds a cross-encoder reranker, so the comparisons with HR are pipeline comparisons, not embedding comparisons.

## 5. Facts for the paper
- Swapping bge-m3 for multilingual-e5-large inside the frozen HYBRID (BM25 + dense RRF, 6,000 characters) gave 58/90
  against 62..63/90 (net -5..-4; sign p 0.18..0.34; component CI [-11.6, +2.7] pp). The whole loss was on Persian
  questions (fa net -5..-4, en 0). Source: `results.json` (table, compare "E5_HYBRID vs HYBRID_6000").
- With dense retrieval alone, the two models were indistinguishable: e5 54..55/90 vs bge-m3 53/90 (net +1..+2, p >= 0.81).
  Source: `results.json` (compare "E5_DENSE vs S10_DENSE_6000").
- At the candidate-list level the models tie: a complete gold bundle is in the top 40 for 72/90 with either hybrid,
  and for 69/90 (e5) vs 71/90 (bge-m3) dense-only. e5 is slightly ahead on Persian (40 vs 38 of 45) and behind on
  English (29 vs 33 of 45). Only 22.8 of the 40 dense candidates are shared. Source: `results.json` → `recall40`.
- The accuracy gap therefore comes from ordering inside the top 40, not from the lists. The 6,000-character pack
  carries a complete bundle for 53/90 with e5-hybrid vs 57/90 with bge-m3 hybrid (Persian 30 vs 33). Source:
  `results.json` → `reach`.
- The choice of embedding model mattered less than adding a reranker: HR_6000 (bge-m3 + reranker) beats E5-HYBRID by
  11..13 questions (p 0.0044..0.0127) and E5-DENSE by 14..17 (p <= 0.0066). Source: `results.json` → `compare`.
- Embedding all 58,218 chunks and 156 questions with e5-large took 17.6 min of GPU wall time at 1.47 GB peak
  allocation on a shared RTX 4060 Laptop. Source: `gpu_phase.json`.
