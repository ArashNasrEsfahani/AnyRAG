# S12 "chunk size": HYBRID and HR at 500 / 1,000 / 2,000-character chunks

This is post-hoc and exploratory work: the test split was already unsealed. Nothing was tuned. The two new sizes and
their overlaps (500/100 and 2,000/400) were fixed in the brief, and each arm ran once on test. Reader: GLM-5.3-flash
(Together pin). Judge: the pinned gpt-oss-120b (deepinfra/bf16). Scores are out of the 90 answerable test questions,
given as lo-hi bounds over unresolved judge cells. The 1,000-character arms are the frozen HYBRID_6000 / HR_6000
cells, scored from their frozen verdicts.

## 1. What was run, exactly

**Code.** Frozen ragbench (77924d4), imported read-only. The only config change is
`Config.copy_with({"chunking": {"chunk_chars": S, "overlap_chars": O}, "paths": {"index_root": S12_chunks/idx<S>,
"locks_root": S12_chunks/locks<S>}})`. The path override keeps every new index and the new embedding cache inside this
folder (`s12.py:cfg_for`). Everything else is unchanged:
- chunker options (`min_window_frac` 0.5, packing/merging rules);
- `index_with_path` breadcrumbs, tokenizer, bm25s parameters;
- embedding model/digest, truncate, num_ctx 8192, batch 16;
- `chunking.top_k` 8, `max_candidates` 40, RRF k=60, depth 40, reranker settings, evidence budget 6,000.

`packs.py` is a verbatim copy of S2a's verified pack helpers. `answer.py` and `judge.py` are S2a's scripts with only
the cap ($0.30) and the experiment id changed.

**Index** (`index.py`). For each size and each of the six pool sets: the frozen `HybridIndex.build`.
- Chunks: `baselines.build_chunks`. BM25 runs on the CPU.
- Every chunk is embedded with bge-m3 through the frozen `OllamaClient` (`embed_missing`, batch 16).
- Output: `idx<S>/_hybrid/<set>-<fp8>/{chunks.jsonl, bm25/, embeddings.npy, has_vec.npy}`. The vectors are also in
  `idx<S>/_embed_cache/embeddings.sqlite`.
- Every set is complete: `dense_complete` true, 0 missing vectors (`index_<S>.jsonl`).
- Each test question's scope is its whole pool, so "chunks per pool" equals chunks per set.

**Retrieval and packs** (`retrieve.py`). This mirrors `HybridRetrieval._answer` without the answer call.
- `qvec`: the 96 question vectors, computed once through the frozen OllamaClient (`embed([normalize(query.text)])`).
  They are shared by all sizes, since the question vector does not depend on the chunking.
- `search`: `hx.search(k=40, depth=40)` gives HYBRID. `hx.search(k=50, depth=40)` gives HR's candidates.
- `rerank`: one GPU phase per size, all 96 questions in each phase. It uses the frozen `CrossEncoderReranker`
  (bge-reranker-v2-m3, fp16 cuda, batch 16, max_length 512). The script aborts if the model is not on CUDA or falls
  back to the CPU. Scores are cached in `rerank_scores.jsonl`.
- `pack`: the frozen `rerank_hits` (keep 40) runs on the cached scores. Items are built exactly as in `_answer`. The
  pack is the frozen `_offer` + `final_pack` at 6,000, followed by `build_answer_messages`.
- Outputs: `packs/<ARM>/<qid>.json`, `packs_index.jsonl`, `pack_traces.jsonl` (packed chunk spans and the top 40).

**Checks before building anything new:**
- **Size 1,000 reproduces the frozen packs byte-for-byte on all 96 questions, for both arms.** This used the frozen
  index, my question vectors, my rerank phase and my pack path. HYBRID_6000 matched 96/96 and HR_6000 matched 96/96
  (messages, messages sha256 and num_predict; `verify_1000.json`). I checked all 96 questions rather than a handful.
- The build path also reproduces the frozen 1,000 index (`check_build.py` -> `check_build.json`):
  - With the `copy_with` config, `build_chunks` gives the frozen chunk count and `chunks_fingerprint` on 6/6 sets.
  - The BM25 token vocabulary is the frozen `bm25_meta` vocab minus exactly 1 on every set. bm25s counts one extra
    reserved entry; this is consistent and does not affect scoring.
  - 66 frozen chunks re-embedded through the same Ollama path match the stored vectors exactly (max |diff| 0.0).
- The question vectors differ from S9's Ollama query vectors by at most 1.8e-7.

**Reader and judge.**
- 381 reader calls, all served by Together, all `finish_reason=stop`, 0 errors. 3 HR_C2000 cells had messages identical
  to another new cell and reused its answer. No new pack is identical to a frozen pack (`packs_index.jsonl`).
- 301 successful judge calls. 6 judge HTTP attempts failed at the network level:
  - 4 recovered within the judge's own 3 attempts.
  - 2 records got no reply at all, which counts as an infrastructure failure, not a malformed reply. I re-ran them once
    with the frozen `judge_records(retry_failures=True)` (`judge_retry.py`), and both were judged.
- 1 new cell needs adjudication (HYBRID_C500, counted [0, 1]). The judge was asked about 3 questions first to
  estimate cost ($0.0031 for 6 cells), and then run on all.

**Statistics** (`report.py` -> `results.json`, `tables.md`).
- The `revision_stats.py` helpers are taken out with `ast` and run unchanged, as S2a does.
- Coverage uses S7's `s7lib.gold_reach`, unchanged: the text method on the exact reader messages. An item counts as
  present when at least 90% of its quote occurs verbatim inside items of the same document; "complete" means every
  item of some bundle is present.
- I added a character-offset method: a gold `[start, end)` counts as covered when at least 90% of it lies inside the
  union of the chunk spans. It is applied to the packed items, the top 8 and the top 40.

## 2. Results

### Accuracy, abstention, pack shape, reader tokens (all 96 questions; `tables.md`)

| arm | chunk | correct /90 | abstain /6 | mean pack chars | mean items in pack | reader prompt tok (all / en / fa) | completion tok | LLM calls/q |
|---|---|---|---|---|---|---|---|---|
| HYBRID_C500_6000 | 500 | **58-59** | 6 | 5,908 | 11.5 | 2,099 / 1,561 / 2,637 | 116 | 1 |
| HYBRID_6000 (frozen) | 1000 | **62-63** | 6 | 5,956 | 6.7 | 2,093 / 1,531 / 2,655 | 123 | 1 |
| HYBRID_C2000_6000 | 2000 | **49** | 6 | 5,961 | 3.7 | 2,086 / 1,524 / 2,648 | 112 | 1 |
| HR_C500_6000 | 500 | **62** | 6 | 5,898 | 11.4 | 2,110 / 1,559 / 2,661 | 124 | 1 |
| HR_6000 (frozen) | 1000 | **69-71** | 6 | 5,958 | 6.6 | 2,111 / 1,553 / 2,669 | 126 | 1 |
| HR_C2000_6000 | 2000 | **51** | 6 | 5,957 | 3.7 | 2,077 / 1,507 / 2,647 | 116 | 1 |

At 2,000 characters the packer's last item was cut in 73 (HYBRID) and 71 (HR) of 96 packs. At 500 it was cut in 39
and 38.

### By tier, language, genre (correct lo-hi / n)

| arm | T1 /30 | T2 /30 | T3 /30 | en /45 | fa /45 | BOOK /30 | LAW /30 | RPT /30 |
|---|---|---|---|---|---|---|---|---|
| HYBRID_C500 | 26 | 18 | 14-15 | 25-26 | 33 | 19 | 20-21 | 19 |
| HYBRID (1000) | 28 | 20-21 | 14 | 27 | 35-36 | 19 | 22 | 21-22 |
| HYBRID_C2000 | 26 | 12 | 11 | 20 | 29 | 15 | 16 | 18 |
| HR_C500 | 28 | 18 | 16 | 28 | 34 | 19 | 22 | 21 |
| HR (1000) | 28 | 23-24 | 18-19 | 33-35 | 36 | 20 | 23-24 | 26-27 |
| HR_C2000 | 26 | 13 | 12 | 24 | 27 | 15 | 18 | 18 |

### Paired comparisons (net = wins - losses, bounds; exact sign-test p range; 95% bootstrap CI in pp)

| A vs B | net | wins/losses (resolved) | sign p | CI questions | CI 20 components |
|---|---|---|---|---|---|
| HYBRID_C500 vs HYBRID (1000) | -5..-3 | 3/7 | 0.23-0.55 | [-13.3, 3.3] | [-13.0, 2.0] |
| HYBRID_C500 vs HR (1000) | -13..-10 | 2/12 | 0.002-0.013 | [-23.3, -3.3] | [-25.5, -5.7] |
| HR_C500 vs HYBRID (1000) | -1..0 | 10/11 | 1.0 | [-11.1, 10.0] | [-10.3, 7.6] |
| HR_C500 vs HR (1000) | -9..-7 | 4/11 | 0.049-0.12 | [-18.9, 0.0] | [-18.1, -2.2] |
| HYBRID_C2000 vs HYBRID (1000) | -14..-13 | 3/16 | 0.003-0.004 | [-25.6, -5.6] | [-24.7, -6.0] |
| HYBRID_C2000 vs HR (1000) | -22..-20 | 3/23 | <=0.0001 | [-34.4, -12.2] | [-37.4, -10.4] |
| HR_C2000 vs HYBRID (1000) | -12..-11 | 5/16 | 0.017-0.027 | [-23.3, -2.2] | [-23.5, -3.9] |
| HR_C2000 vs HR (1000) | -20..-18 | 2/20 | <=0.0001 | [-32.2, -11.1] | [-33.3, -13.4] |
| HR_C500 vs HYBRID_C500 | +3..+4 | 9/6 | 0.45-0.61 | [-4.4, 13.3] | [-2.4, 10.9] |
| HR_C2000 vs HYBRID_C2000 | +2 | 12/10 | 0.83 | [-7.8, 12.2] | [-11.5, 15.3] |
| HYBRID_C2000 vs HYBRID_C500 | -10..-9 | 6/15 | 0.053-0.078 | [-21.1, 0.0] | [-21.1, 1.2] |
| HR_C2000 vs HR_C500 | -11 | 5/16 | 0.027 | [-22.2, -3.3] | [-20.4, -4.2] |
| HR (1000) vs HYBRID (1000), frozen | +6..+9 | 10/4 | 0.049-0.18 | [-1.1, 18.9] | [-1.0, 21.2] |

The resolved wins and losses by tier, language and genre are in `tables.md`. Most losses at both sizes are T2/T3
questions. For example, HR_C2000 vs HR (1000) is T1 +0/-2, T2 +2/-12, T3 +0/-6.

### Gold-bundle coverage (complete / partial / none, of 90)

| arm | in pack (S7 text) | in pack (offset) | in top 8 (offset) | in top 40 (offset) |
|---|---|---|---|---|
| HYBRID_C500 | 55/30/5 | 53/32/5 | 49/33/8 | 65/24/1 |
| HYBRID (1000) | 57/26/7 | 57/25/8 | 58/26/6 | 72/17/1 |
| HYBRID_C2000 | 46/34/10 | 46/34/10 | 61/25/4 | 79/10/1 |
| HR_C500 | 59/28/3 | 57/29/4 | 52/30/8 | 66/23/1 |
| HR (1000) | 63/23/4 | 63/23/4 | 66/21/3 | 75/14/1 |
| HR_C2000 | 51/34/5 | 51/32/7 | 64/24/2 | 80/9/1 |

- The text and offset pack measures agree on 88-90 of 90 questions per arm.
- Coverage broken down by tier, language and genre is in `tables.md` and `results.json` (`coverage`).

### Chunks per pool per size (`results.json` `index.chunks`)

| pool (set) | 500 | 1000 | 2000 | mean chunk chars 500 / 1000 / 2000 |
|---|---|---|---|---|
| RPT-fa (C) | 14,267 | 6,921 | 3,236 | 393 / 767 / 1,569 |
| RPT-en (D_en) | 52,553 | 25,685 | 12,400 | 402 / 804 / 1,604 |
| BOOK-fa | 18,078 | 8,905 | 4,319 | 408 / 820 / 1,649 |
| BOOK-en | 19,997 | 9,661 | 4,876 | 409 / 841 / 1,664 |
| LAW-fa | 3,442 | 1,619 | 758 | 390 / 799 / 1,656 |
| LAW-en | 10,971 | 5,427 | 2,616 | 403 / 792 / 1,564 |
| **total** | **119,308** | **58,218** | **28,205** | |

### GPU time (`index_<S>.jsonl`, `ollama_ledger_<S>.jsonl`, `gpu_phases.jsonl`)

| phase | wall | notes |
|---|---|---|
| probe, 200 chunks, 500 | 3.4 s (59 chunks/s) | nvidia-smi 787 MiB before and after (Ollama's memory already resident) |
| probe, 200 chunks, 2000 | 5.8 s (34 chunks/s) | |
| embed index 500 (119,293 texts, 15.6 M tokens) | 57 min elapsed (19:36-20:33); 3,229 s summed Ollama call time | ran in parallel with the 2000 job and other streams (GPU at 100%), so per-call times include contention |
| embed index 2000 (28,205 texts, 12.2 M tokens) | 30 min elapsed (19:37-20:06); 1,691 s summed Ollama call time | same |
| question vectors (96) | 50.5 s | |
| rerank 1000 / 2000 / 500 (96 q x 50 each) | 96 s / 154 s / 32 s | fp16 cuda, peak allocated 1,271 MB (500: 1,193 MB); model load about 20 s each, not included |

The 2,000-character reranker inputs hit the frozen 512-token `max_length`: 2.35 M padded tokens vs 1.45 M at 1000.
Long chunks are therefore truncated on the chunk side before scoring; this is the frozen behaviour.

## 3. Cost

$0.1798 of the $0.30 cap (`ledger.jsonl`: 381 reader calls, 307 judge HTTP calls). The key's `limit_remaining` was
checked before every batch of at most 50 calls and never went below $3.20.

## 4. Failed, skipped or worth knowing

- **Two index runs crashed** with Ollama HTTP 400 "Only one usage of each socket address", an ephemeral-port
  exhaustion inside Ollama under the shared load (the 500 run on BOOK_fa, the 2000 run on D_en).
  - Both resumed from the per-batch-committed cache. I wrapped `HybridIndex.build` in a retry loop in `index.py`
    without changing any vector computation, and one running 500 job was killed and restarted with it.
  - Effect on the logs: the `wall_s` of those restarted sets in `index_<S>.jsonl` covers only the resumed part. Use
    the ledger totals above. Partial logs are in `logs/*_crashed.log` and `logs/*_killed.log`.
- **Rerank phases.** Instead of one combined phase there were three, one per size, each with all 96 questions, because
  the 500 index finished later. Scores are per question and do not depend on phase grouping.
- **Frozen packer knobs were not adapted to chunk size.** `top_k` stays 8 and `max_candidates` 40. At 2,000
  characters, about 3 whole chunks fit in 6,000 characters, so the packer cuts the 4th in roughly 3/4 of packs. At
  500, the packer fills about 11 items. Packs have the same character length at every size (5,898-5,961).
- **No separate dev smoke test with LLM calls.** The code path was debugged through the 96/96 byte-for-byte
  reproduction at size 1000, which makes no LLM calls, and through the 3-question test cost estimate. Nothing was
  iterated on test.
- The reader prompt-token means are almost equal across sizes, because packs are budgeted in characters.

## 5. Facts for the paper

1. At a fixed 6,000-character pack, 1,000-character chunks were best for both pipelines.
   - HYBRID: 58-59 (500) vs 62-63 (1000) vs 49 (2000).
   - HR: 62 (500) vs 69-71 (1000) vs 51 (2000).
   - Source: `tables.md` and `results.json` `table`.
2. 2,000-character chunks are clearly worse than the frozen setup: HR_C2000 vs HR_6000 net -20..-18, 2 wins / 20
   losses, sign p <= 0.0001, component CI [-33.3, -13.4] pp. HYBRID_C2000 vs HYBRID_6000 is net -14..-13, p <= 0.004.
   Source: `results.json` `compare`.
3. The 2,000-character loss happens at packing, not at retrieval.
   - A complete gold bundle is in the top 40 for 79-80/90 questions at 2000, vs 72-75 at 1000.
   - It is in the top 8 for 61-64 vs 58-66.
   - It is in the 6,000-character pack for only 46-51 vs 57-63, since about 3.7 chunks fit and the last one is cut in
     about 74% of packs.
   - Source: `results.json` `coverage` and `last_item_cut`.
4. 500-character chunks cost 3-9 questions against 1,000.
   - HR_C500 vs HR_6000 net -9..-7, sign p 0.049-0.12, component CI [-18.1, -2.2].
   - HYBRID_C500 vs HYBRID_6000 net -5..-3, p >= 0.23.
   - The reranker's gain over plain HYBRID shrinks at both other sizes: +3..+4 at 500 and +2 at 2000, vs +6..+9 at
     1000.
   - Source: `results.json` `compare`.
5. The losses come almost entirely from T2/T3. T1 is 26-28/30 at every size and pipeline. Source: `results.json`
   `groups` and `wins_losses_resolved_by_group`.
6. Indexing cost scales with chunk count.
   - The six pools hold 119,308 / 58,218 / 28,205 chunks at 500 / 1000 / 2000.
   - Embedding the new indexes through Ollama bge-m3 took 57 min (500) and 30 min (2000) of elapsed GPU time, run in
     parallel on a shared RTX 4060.
   - Sources: `results.json` `index.chunks` and `index.gpu`, `ollama_ledger_<S>.jsonl`.
