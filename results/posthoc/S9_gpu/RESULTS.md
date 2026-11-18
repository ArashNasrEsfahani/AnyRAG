# S9 GPU: post-hoc experiments with light local GPU work (November 2026)

These experiments are post-hoc and exploratory (COMMON.md). The author approved light local GPU work for this stream (RTX 4060 Laptop, 8 GB). That replaces COMMON.md's "No GPU" section here. The rules followed were: one GPU job at a time; OMP/OPENBLAS/MKL threads = 2; below-normal process priority (`gpuenv.py`); and peak memory probed on one item before each full run. The frozen worktree @ 77924d4 was imported read-only and never edited. No test question was used for tuning.

## 1. What was run

### Job 1: Ollama bge-m3 vs HF bge-m3 (`align.py`; outputs `align_results.json`, `align_vectors.npz`, `align_lists.jsonl`, logs in `logs/align_*.log`)
- **The two paths, as in the frozen code.**
  - **Ollama path (HYBRID/HR).** Model `bge-m3:latest`, digest 7907646…, through the frozen `OllamaClient.embed`, with num_ctx 8192 and truncate on.
    - Query: `normalize(q)`, one text per call, as in `HybridIndex.query_vector`.
    - Chunk: `hybrid.index_text(chunk)`, in batches of 16, as the index was built.
  - **HF path (LEX8/LEX5B).** The frozen `m3multi.M3Encoder`, BAAI/bge-m3 snapshot 5617a9f…, fp16 on cuda.
    - Query: `normalize(q)` at max_length 128, the LEX5B query path. It was also run at 512.
    - Chunk: max_length 512, batch 16, which is how the lex8 chunk artifact was made.
  - Both outputs are L2-normalised before comparison.
- **Items encoded.** The 156 questions (96 test + 60 dev) and a seeded sample of 1,000 chunks: `random.Random(0).sample` over all 58,218 chunks of the 6 corpus sets.
- **Stored vectors.** These were also compared: the HybridIndex `embeddings.npy` (Ollama) against `lex8/idx/chunks/<set>.npz` (HF, f16). This was done for the sample and for all 58,218 chunks.
- **Top-40 dense lists (96 test questions, pool scope).** The HYBRID dense channel (Ollama query · stored Ollama chunks) against the LEX dense channel (HF query@128 · stored HF chunks).
- **Counterfactual packs.** The frozen HYBRID_6000, HYBRID_15500, HR_6000 and HR_15500 retrieval and packing was re-run twice.
  - **Run (a), reproduction.** The fresh Ollama query vector with the stored Ollama chunk vectors. This reproduced the saved pack's `messages_sha256` in **96/96** cells for all 4 arms. The HYBRID trace order was also reproduced 96/96.
  - **Run (b), counterfactual.** The same with the HF vectors (query@128 and chunks) in the dense channel. HR's reranker was rerun on GPU for both (a) and (b).
- **LEX5B.** Its F2 MaxMean top 40 was recomputed with its own HF dense channel and then with the Ollama dense channel. The HF run reproduced the saved `f2_top40` 96/96. The selector call was not rerun, so for LEX5B only the retrieval stage can be compared.

### Job 2: HYBRID_AGENT (`hybrid_agent.py`, `judge.py`, `analyze.py`; outputs `runs/HYBRID_AGENT/{dev,test}/`, `judge/HYBRID_AGENT/`, `results.json`, `results_tables.md`)
- **Code.** `hybrid_agent.py` is a copy of `S3_agentic/agents.py`. The only change is that `search(query)` is backed by `HrEnv`, which makes HR's own calls:
  - `HybridIndex.search(query, doc_ids=pool scope, k=50, depth=40)`: BM25 + Ollama bge-m3 dense, equal-weight RRF k=60.
  - `rerank.rerank_hits(query, top 50, bge-reranker-v2-m3 fp16 cuda, keep=8)`.
  - The top 8 chunks are rendered exactly as ITER_BM25 renders its chunks.
- **Unchanged from S3.** Prompts, tools, the 5-round limit, 3 calls per round, the forced finish, `finish(chunk_ids)`, `pack_evidence` at 15,500, the frozen answer call and the GLM-5.3-flash pin (together, no fallbacks).
- **Caveat: the prompt still describes BM25.** Because the prompts were kept word for word, the system prompt still describes the tool as "keyword (BM25) search … Matching is by words".
- **Per-search GPU logging.** Each search records the Ollama embed wall time and the reranker's CUDA-synchronised ms in `trace.gpu`.
- **Dev smoke.** 5 dev questions, the same 5 ITER_BM25 used (`runs/HYBRID_AGENT/dev`). No prompt change was made. The pinned judge gave 4/5. These runs were also the cost estimate: about $0.001 per question.
- **Test.** One run on all 96 test questions: 0 errors, process wall 1,303 s (`logs/test_HYBRID_AGENT.log`).
- **Judge.** The frozen pinned judge (`judge_multi`, gpt-oss-120b, deepinfra/bf16, `--require-pin`, 3 attempts), in batches of at most 45 calls with a key check before each. All 96 cells were judged; 0 are `needs_adjudication`.
- **Analysis.** S3's procedure, with the frozen arms loaded through `report_test.py` and ITER_BM25 read from S3's folder. It gives diff bounds and sign-test ranges over unresolved cells, and a bootstrap over questions and over the 20 source-family components (10,000 resamples, seed 0). The contrast without the 26 closed-book-correct questions (`S7_analysis/contam.json`) leaves 64 answerable questions.
- **GPU time per question.** This means different things per arm:
  - HYBRID_AGENT: the sum over searches of embed and rerank time.
  - HR: query embed (from the run ledger) + `trace.rerank.ms`.
  - HYBRID/SETR: query embed.
  - LEX5B: `t_encode_ms`.

## 2. Results

### Job 1: alignment (`align_results.json`)

| comparison (cosine per item) | n | mean | min | 1st pct |
|---|---|---|---|---|
| questions: Ollama vs HF@128 (as deployed) | 156 | 0.999982 | 0.999902 | 0.999926 |
| test questions only | 96 | 0.999981 | 0.999902 | 0.999911 |
| questions: Ollama vs HF@512 | 156 | 0.999982 | 0.999902 | 0.999926 |
| chunk sample: fresh Ollama vs fresh HF@512 | 1,000 | 0.999974 | 0.996514 | 0.999916 |
| chunk sample: stored HYBRID vs stored lex8 | 1,000 | 0.999974 | 0.996513 | 0.999913 |
| all 58,218 stored chunks: HYBRID vs lex8 | 58,218 | 0.999973 | 0.985450 | 0.999889 |
| fresh Ollama vs stored HYBRID (reproducibility) | 1,000 | 1.000000 | 1.000000 | 1.000000 |
| fresh HF vs stored lex8 (fp16 storage) | 1,000 | 0.999999 | 0.999901 | 0.999995 |

- **Question truncation.** The longest question is 90 tokens, so HF's 128-token query cap truncates nothing; @128 and @512 give identical results.
- **Chunk truncation.** Of the sampled chunks, 1 is longer than 512 tokens (max 616). HF truncates such chunks at 512 and Ollama does not (num_ctx 8192). This is the cause of the few low minima, including 0.985 over all chunks.

Top-40 dense lists, 96 test questions (`align_results.json` → `lists`):

| measure | value |
|---|---|
| overlap@40 | mean 39.77 (99.4%), min 39, identical sets 74/96 (fa 39.71, en 39.83) |
| overlap@10 | mean 9.90; top-1 identical 96/96 |
| Spearman of the two scorings over the union of both top-40s | mean 0.9986, min 0.9910 |
| Kendall tau of ranks on the intersection | mean 0.986, min 0.941 |
| Spearman over the whole pool scope | mean 0.99997, min 0.99990 |

Would a test pack change if the other path's dense vectors were used? (Exact answer-message sha256 against the saved pack. The reproduction was 96/96 in every row.)

| arm | packs that would change | of which the chunk set changes | order only |
|---|---|---|---|
| HYBRID_6000 | 16/96 | 4 | 12 |
| HYBRID_15500 | 36/96 | 10 | 26 |
| HR_6000 | 1/96 | 1 | 0 |
| HR_15500 | 7/96 | 6 | 1 |
| LEX5B (retrieval stage only, Ollama dense instead of HF) | F2 top-40 order identical in 8/96; same top-40 set mean overlap 39.8/40; top-24 shown to the selector identical in order 35/96, as a set 86/96 | – | – |

- **HYBRID fused order.** The order was identical in only 11/96 questions (mean overlap 39.82/40).
- **HR candidate set.** HR's top-50 candidate set was identical in 66/96 questions, and its reranked top 40 was in identical order in 71/96.
- **Why near-identical vectors still move the packs.** The two models agree almost perfectly, but swaps between nearly tied neighbours inside the dense channel move chunks within RRF / MaxMean. Packs are sensitive to order, especially HYBRID_15500, which packs the first 40 in fused order.

### Job 2: HYBRID_AGENT (`results_tables.md`, `results.json`; machine strict verdicts, 90 answerable questions; a..b = bounds over unresolved frozen cells)

| arm | all /90 | T1 /30 | T2 /30 | T3 /30 | fa /45 | en /45 | without closed-book-correct /64 | abstention /6 |
|---|---|---|---|---|---|---|---|---|
| **HYBRID_AGENT** | **77** | 28 | 24 | 25 | 39 | 38 | 55 | 6 |
| ITER_BM25 (S3) | 73 | 27 | 23 | 23 | 34 | 39 | 50 | 6 |
| HYBRID_6000 | 62..63 | 28 | 20..21 | 14 | 35..36 | 27 | 44..45 | 6 |
| HR_6000 | 69..71 | 28 | 23..24 | 18..19 | 36 | 33..35 | 50..51 | 6 |
| LEX5B_6000 | 68..69 | 29 | 23..24 | 16 | 36..37 | 32 | 51..52 | 6 |
| HYBRID_15500 | 68 | 28 | 22 | 18 | 35 | 33 | 49 | 6 |
| HR_15500 | 69..70 | 28 | 21..22 | 20 | 38 | 31..32 | 50..51 | 6 |

Paired comparisons, HYBRID_AGENT minus the other arm (90 answerable questions; sign p unadjusted):

| vs | net | wins/losses | sign p | 95% CI pp (questions) | 95% CI pp (components) | T1 | T2 | T3 | fa | en |
|---|---|---|---|---|---|---|---|---|---|---|
| ITER_BM25 | +4 | 7/3 | 0.344 | [-2.2, 11.1] | [-6.0, 14.9] | +1 | +1 | +2 | +5 | -1 |
| HR_6000 | +6..+8 | 15/8 … 16/9 | 0.152-0.307 | [-3.3, 18.9] | [-5.4, 15.9] | +0 | +0..+1 | +6..+7 | +3 | +3..+5 |
| HR_15500 | +7..+8 | 14/6; 14/7 | 0.115-0.189 | [-2.2, 18.9] | [0.0, 16.0] | +0 | +2..+3 | +5 | +1 | +6..+7 |
| HYBRID_6000 | +14..+15 | 20/5; 20/6 | 0.004-0.009 | [5.6, 26.7] | [3.5, 24.5] | +0 | +3..+4 | +11 | +3..+4 | +11 |
| LEX5B_6000 | +8..+9 | 15/7; 16/7 | 0.093-0.134 | [-1.1, 20.0] | [-3.6, 18.6] | -1 | +0..+1 | +9 | +2..+3 | +6 |

Without the 26 closed-book-correct questions (64 answerable questions):

| vs | net | wins/losses | sign p | 95% CI pp (questions) | 95% CI pp (components) |
|---|---|---|---|---|---|
| ITER_BM25 | +5 | 7/2 | 0.180 | [-1.6, 17.2] | [-5.9, 20.5] |
| HR_6000 | +4..+5 | 11/6; 11/7 | 0.332-0.481 | [-6.2, 20.3] | [-8.9, 20.4] |
| HR_15500 | +4..+5 | 8/3; 8/4 | 0.227-0.388 | [-4.7, 17.2] | [-2.9, 18.4] |
| HYBRID_6000 | +10..+11 | 14/3; 14/4 | 0.013-0.031 | [3.1, 29.7] | [1.7, 28.6] |
| LEX5B_6000 | +3..+4 | 10/7; 11/7 | 0.481-0.629 | [-7.8, 18.8] | [-6.8, 18.3] |

Cost and time (all 96 questions; latency and GPU time are descriptive):

| arm | query LLM tokens / q | x HYBRID_6000 | LLM calls / q | GPU ms / q mean (median) | latency s median (mean) |
|---|---|---|---|---|---|
| HYBRID_AGENT | 9,070 | 4.09 | 3.62 | 1,512 (1,137) | 10.5 (13.0) |
| ITER_BM25 | 9,285 | 4.19 | 3.64 | 0 | 8.8 (9.9) |
| HYBRID_6000 | 2,216 | 1.00 | 1.00 | 205 (157) | 1.8 (2.8) |
| HR_6000 | 2,237 | 1.01 | 1.00 | 1,331 (1,205) | 3.2 (4.0) |
| LEX5B_6000 | 4,511..5,131 | 2.04..2.32 | 2.00 | 38 (24) | 4.0 (4.5) |
| HR_15500 | 5,287 | 2.39 | 1.00 | 1,328 (1,187) | 3.6 (4.7) |

- **Agent behaviour.** 2.27 searches per question over 218 searches in all. Rounds: mean 2.50, histogram {2: 72, 3: 12, 5: 12}. Every evidence set came from `finish` (96/96).
- **Evidence size.** Median 2,830 characters and mean 3,123, out of the 15,500 allowed.
- **Failures.** 0 system failures and 0 truncated answers.
- **Per search.** Ollama embed took 77 ms and the reranker 589 ms on average (50 pairs, cuda fp16, 0 CPU fallbacks). Peak torch allocation was 1,271 MB; nvidia-smi showed 1,962 MiB with Ollama and the reranker both loaded.
- **Latency comparability.** The agent latencies were measured in this session; the frozen ones come from the October test session.

### Total GPU wall time
- **Job 1.** About 425 s of process wall: probe 27.6 s, encode 80.4 s, lists run 1 164.9 s, lists run 2 152.3 s. The lists step was run a second time to add the chunk-set/order split, after deleting the first output, now kept in `logs/align_lists_v1.jsonl`.
- **Job 2.** Process wall was 90 s for the dev smoke and 1,303 s for test. GPU-side compute inside those runs was 7.5 s (dev) and 145.1 s (test), summed from `trace.gpu`. Each process also spent about 23–25 s loading the reranker.
- **Total.** About 1,820 s (about 30 min) of GPU-holding process wall time. The GPU was actually busy for about 10.5 min: 425 s (job 1, an upper bound because it includes CPU work) + 153 s (job 2 compute) + about 48 s of model loads. The rest of job 2's wall time was spent waiting for OpenRouter.

## 3. Cost spent (`ledger.jsonl`, 457 calls; every call has `usage.cost`)

| part | calls | USD |
|---|---|---|
| HYBRID_AGENT dev smoke (agent + answer) | 16 | 0.0048 |
| HYBRID_AGENT test | 348 | 0.1097 |
| judge dev (5) + test (96; 4 decided by rule, 88 model calls) | 93 | 0.0122 |
| **total** | **457** | **$0.127** of the $0.40 cap |

- Ollama and the reranker are local and have no API cost. Job 1 made no API calls.
- The key had $4.73 remaining before job 2 and $4.41 after; S8 was spending in parallel. The $2.00 floor was checked every 5 questions and before each judge batch, and was never reached.

## 4. Failed, skipped or caveats
- **Nothing failed.** 0 errors, 0 unresolved judge cells for HYBRID_AGENT, 0 CPU fallbacks of the reranker.
- **The prompt still says BM25.** HYBRID_AGENT's prompt still describes the search tool as BM25 keyword search. This is deliberate: the prompts were kept identical so that only the backend differs. A prompt that describes semantic search might change how the agent writes queries; that was not tried.
- **Only GLM-5.3-flash, with machine verdicts.** There are no human checks and no other readers, so these numbers are not comparable with the paper's human-merged ones. It is one run, and the reader has no seed.
- **LEX5B counterfactual.** Only the retrieval stage was recomputed (F2 order and the top 24 shown). Whether the selector would pick different bundles needs an LLM call and was not done.
- **The comparisons are post-hoc and exploratory.** There are 5 comparisons plus 5 more without the closed-book-correct questions, and the p values are unadjusted. Only HYBRID_AGENT vs HYBRID_6000 is below 0.05 (0.004–0.009; 0.013–0.031 without the closed-book-correct questions).

## 5. Facts for the paper
1. **Same model.** Ollama `bge-m3:latest` (used by HYBRID/HR) and HF BAAI/bge-m3 (used by LEX5B) give the same embedding up to numerical noise. Per-item cosine is mean 0.99998, minimum 0.99990, 1st percentile 0.99993 over the 156 questions. Over 1,000 sampled chunks it is mean 0.99997, 1st percentile 0.99992, minimum 0.9965; the minimum is a chunk of more than 512 tokens, which HF truncates and Ollama does not. They can be called the same model, with that truncation footnoted (`align_results.json`).
2. **Dense lists agree; fused orders do not always.** The two paths' top-40 dense lists agree on 99.4% of chunks: mean overlap@40 39.77, top-1 identical 96/96, Kendall tau 0.986 on the shared items. But the small rank swaps propagate through fusion. With HF vectors in the dense channel, the packs would change in 16/96 HYBRID_6000 cells (4 with a different chunk set), 36/96 HYBRID_15500 (10), 1/96 HR_6000 and 7/96 HR_15500, against a 96/96 exact reproduction of the saved packs (`align_results.json` → `lists`).
3. **Hybrid search tool vs BM25 tool.** Giving S3's iterative GLM-5.3-flash agent the real HR retrieval as its search tool (BM25 + bge-m3, RRF, bge-reranker-v2-m3 top 50 → 8) gave **77/90** machine-strict, against 73/90 with the BM25-only tool.
   - Against ITER_BM25: +4, 7/3, p 0.34. Against HR_6000: +6..+8, p 0.15–0.31. Against LEX5B_6000: +8..+9, p 0.09–0.13.
   - Against HYBRID_6000: +14..+15, 20/5–20/6, p 0.004–0.009. This comparison is post-hoc and unadjusted (`results.json`).
4. **Where the gain is.** The gain over ITER_BM25 is on Persian (+5; 39/45 vs 34/45); English is −1. Against the single-pass arms it is mostly T3 (+5 to +11) and English. Without the 26 closed-book-correct questions it scored 55/64, against 50 for ITER_BM25 and 50–51 for HR_6000 (`results_tables.md`).
5. **Cost.** 9,070 query LLM tokens per question, 4.09× HYBRID_6000 and the same as ITER_BM25. Plus 1.5 s of GPU time per question (2.27 searches × (77 ms embed + 589 ms rerank)). Median latency 10.5 s, against 3.2 s for HR_6000. The stream billed $0.127 (`results.json`, `ledger.jsonl`).
