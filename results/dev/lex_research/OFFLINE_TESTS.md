# LEX research, part 2: offline retrieval tests with bge-m3 sparse, multi-vector and diversity (2026-09-26)

**Scope.** The 54 answerable dev questions. Metric: lex4 decisive-span retention with source identity, in the 6,000-char pack and in the top-24 candidates. No LLM or API calls; spend $0.
- **Code.** Encoder: `ragbench/retrieval/m3multi.py` (branch `upgrades`, commit 03b5e93). Harness in this folder: `count.py`, `encode.py`, `evaluate.py`.
- **Data.** Tables: `OFFLINE_TABLES.md`. Numbers: `results.json`. Per question: `rows.jsonl`. Index-time vectors: `m3/*.pkl`. Step log: `PROGRESS.md`. Literature: `RESEARCH.md`.
- **Declared first.** Arms, the diversity rule (doc-decay 0.5, then MMR λ 0.7) and the 1 : 0.3 : 1 weights were written in `PROGRESS.md` before the run. No constant was tuned on these results.
- **Baselines were reused, not re-run.**
  - HYBRID-intact is the deterministic fused order. It matches `lex6/routing_offline` exactly (max diff 0.000).
  - HYBRID+rerank is the SAVED dev run's reranked order (`retrieved_units[].meta.chunk`). The reranker was not run again.
- **Caveat.** This dev set has been seen many times. One question moves the mean by 0.019. Differences of 1–3 questions are noise.

## 0. Packer check: two packers, and the saved numbers reproduce

| order | chunk packer (the saved arms' packer) | intact packer (the lex5/lex6 replay packer) | saved evidence |
|---|---:|---:|---:|
| HYBRID | 0.720 (32 full) | 0.729 (32) | 0.720 |
| HYBRID+rerank | **0.806 (35)** | 0.766 (32) | **0.806** |

The chunk packer is `contracts.pack_evidence` over the top-40 chunk hits, cutting the last item. It reproduces the saved retention of both baselines exactly, so its table (1b below) is the fair comparison.

The intact-package packer costs the reranker 0.04. Earlier replays that scored HYBRID+rerank at 0.80 with the intact packer (idea_test) were close to this. **The packer choice matters as much as some ranking changes do.** A strong ranker should keep the chunk packer.

## 1. Results (chunk packer; cells give mean retention with questions at 100% in brackets)

| arm | pack | top-24 | fa | en | T1 | T2 | T3 | T3 all gold docs in pack /18 | better/worse vs rerank | better/worse vs HYBRID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 0.720 (32) | 0.886 (43) | 0.722 | 0.718 | 0.926 | 0.623 | 0.611 | 12 | 6/13 | – |
| **HYBRID+rerank** | **0.806 (35)** | 0.906 (41) | **0.877** | 0.736 | 0.981 | **0.748** | **0.690** | 13 | – | 13/6 |
| a0 sparse alone | 0.685 (27) | 0.896 (43) | 0.747 | 0.622 | 0.870 | 0.605 | 0.579 | 11 | 4/13 | 7/12 |
| **a1 RRF(BM25, dense, m3-sparse)** | **0.763 (34)** | 0.921 (44) | 0.772 | **0.755** | 0.981 | 0.669 | 0.639 | **14** | 6/10 | **4/0** |
| a2 RRF(dense, sparse) | 0.705 (28) | 0.902 (44) | 0.741 | 0.669 | 0.870 | 0.626 | 0.618 | 11 | 5/14 | 6/7 |
| b1 ColBERT rescore of the fused top-80 | 0.730 (31) | 0.904 (43) | 0.735 | 0.726 | 0.926 | 0.640 | 0.624 | 11 | 7/12 | 5/4 |
| b2 ColBERT alone (whole pool) | 0.730 (31) | 0.901 (43) | 0.735 | 0.726 | 0.926 | 0.640 | 0.624 | 11 | 7/12 | 5/4 |
| b3 RRF(BM25, dense, ColBERT) | 0.738 (33) | 0.904 (43) | 0.759 | 0.718 | 0.926 | 0.651 | 0.639 | 12 | 6/11 | 2/0 |
| c1 m3-all (1 : 0.3 : 1) rescore of bm ∪ dn ∪ sp | 0.734 (32) | 0.899 (43) | 0.759 | 0.710 | 0.926 | 0.623 | 0.655 | 12 | 7/12 | 4/2 |
| c2 m3-all rescore of the top-80 | 0.734 (32) | 0.899 (43) | same as c1 | | | | | 12 | 7/12 | 4/2 |
| **c3 RRF(BM25, dense, sparse, ColBERT)** | 0.760 (34) | **0.923 (46)** | 0.772 | 0.749 | 0.981 | 0.669 | 0.630 | 13 | 6/10 | 4/1 |
| c4 RRF(BM25, c1) | 0.746 (33) | 0.886 (42) | 0.765 | 0.727 | 0.926 | 0.660 | 0.653 | 14 | 7/12 | 3/0 |
| d MMR 0.7 on a1 | **0.767 (34)** | **0.926 (44)** | **0.790** | 0.744 | 0.981 | 0.663 | 0.657 | 14 | 6/9 | 5/2 |
| d doc-decay 0.5 on a1 | 0.749 (32) | 0.901 (43) | 0.753 | 0.745 | 0.926 | 0.669 | 0.653 | **16** | 6/12 | 7/3 |
| d doc-decay 0.5 on c3 | 0.748 (32) | 0.893 (43) | 0.759 | 0.736 | 0.926 | 0.637 | 0.681 | **16** | 5/11 | 7/3 |

- The intact-packer table and every diversity variant are in `OFFLINE_TABLES.md` §1a/§1b.
- Under the intact packer the best arm is b3 (0.754), and the reranker scores 0.766.

## 2. Query-time latency per stage (p50 / p95 over 54 questions)

| stage | p50 ms | p95 ms |
|---|---:|---:|
| BM25 + dense ranking (CPU, query vector given) | 3.4 | 14.8 |
| bge-m3 query encode on the GPU, fp16: one pass gives dense + sparse + ColBERT | 13.2 | 27.3 |
| the same on the CPU, fp32, 2 threads | 182.6 | 254.1 |
| sparse channel: whole-pool CSR product and top-40 (CPU) | 1.4 | 6.4 |
| ColBERT MaxSim over the ~80 fused candidates, fp16 vectors in RAM (CPU) | 46.6 | 75.8 |
| **cross-encoder rerank of 50 candidates on the GPU (saved HYBRID+rerank trace)** | **3,938** | **5,545** |

**Index-time cost.**
- All 58,218 pool chunks were encoded in 12 minutes on the RTX 4060: sparse weights, ColBERT MaxSim against every question, and HF dense.
- Sparse weights are small.
- Full ColBERT vectors would need about 27 GB in fp16 (58k chunks × about 225 tokens × 1,024 × 2 bytes). Only the 2,501 candidate-union vectors were kept.

**Query-time cost.**
- a1 adds one bge-m3 query encode plus a sparse product: about 15 ms on the GPU or about 185 ms on the CPU.
- The HF dense vector from that pass could replace the Ollama query embedding, so the net cost can be about 0.
- That is roughly 1.01–1.1x HYBRID's p50 of 1.62 s, against about 1.7x for the reranker.
- Zero LLM tokens are added.

## 3. What the numbers say

1. **Nothing tested beats HYBRID+rerank on pack retention.**
   - The best arms are MMR-on-a1 (0.767), a1 (0.763) and c3 (0.760), all about 0.04 below 0.806.
   - They are +0.04 over HYBRID. a1 is **4 better / 0 worse** than HYBRID: a free, lossless lift.
2. **In English the m3 arms match or beat the reranker; in Persian they do not.**
   - English: a1 0.755 against 0.736.
   - Persian: a1 0.772 against 0.877.
   - Four of a1's ten losses to the reranker are RPT-fa T2/T3: RPT-fa-T2-001, T2-003, T3-002 and T3-003.
3. **The losses are ordering, not recall.**
   - In 9 of a1's 10 losses to the reranker, a1's top 24 already holds at least as much of the evidence as the reranker's pack (in 6 of them, all of it). The evidence sits below the roughly 6-item pack cut.
   - c3 has the best top-24 of any arm: **0.923 with 46/54 full**, against the reranker's 0.906 with 41 full.
   - So the m3 channels are a better **candidate generator** than the reranker, but a worse **orderer** of the first six.
4. **ColBERT (bge-m3 multi-vector) is not worth its storage here.**
   - Alone or as a rescorer it scores 0.730, barely above HYBRID. As an RRF channel it adds nothing over sparse (c3 against a1).
   - The paper's 1 : 0.3 : 1 score fusion (c1/c2, 0.734) is worse than RRF with BM25.
   - The published MIRACL-fa gain ("All" 63.4 against dense 60.9, `RESEARCH.md`) does not show up at pack level on our corpora.
5. **Diversity trades T1/T2 for T3.**
   - Doc-decay raises T3 all-docs-in-pack to 16/18, against 13 for the reranker and 12 for HYBRID, but drops T1 and T2 precision.
   - MMR is neutral to slightly positive (0.767).
   - Neither is a win on its own.
   - Doc-decay is the strongest T3 document-coverage tool found so far without an LLM.

## 4. Verdict and the proposed LEX7

**Verdict: no.** No combination from a–d beats HYBRID+rerank at near-HYBRID query cost.

What does hold up:
- **a1 (BM25 + dense + bge-m3 sparse, RRF)** is a safe, nearly free upgrade of HYBRID: +0.043, 0 losses, English at or above the reranker.
- **c3 is the best top-24 candidate set measured on this dev set.**

The remaining gap is about 0.04, or about 2–3 questions. It is concentrated in the Persian report pool, and it comes from ordering the first six items.

**Proposed LEX7** (all extra work is at index time; no cross-encoder):

1. **Index time (free at query time).**
   - bge-m3 sparse weights for every chunk: 12 minutes on the GPU for all six pools.
   - **Contextual chunk prefixes**, one index-time LLM call per document or section. Each prefix gives the company or author, the document title, the period or year, and the section, in the document's language. The prefix is prepended to the BM25, sparse and dense text.
   - This goes after the RPT-fa failure mode: the wrong year's or a sibling report's chunks outrank the right one (`idea_test` §DOC, `ROUTING_OFFLINE`). Anthropic reports a 49% cut in top-20 failures without a reranker (`RESEARCH.md` #2).
   - It is the one untested lever aimed at the losses above.
2. **Query time.**
   - One bge-m3 pass gives the dense and sparse query vectors.
   - RRF of BM25 top-40, dense top-40 and sparse top-40. ColBERT is dropped.
   - LEX6's document ranking by centroid (S5): reserve the best package of the top-2 documents only when they are absent from the first 3,000 chars. This is the T3 lever, measured at 17/18 with gold documents.
   - Pack with the **chunk packer**, not the intact packer.
3. **Optional, gated.**
   - A tiny-budget selector over the c3 top-12 snippet view, capped at about 1,000 tokens so it stays at or below 1.5x the reranker arm's 2,215 tokens.
   - Only if steps 1–2 leave a gap. LEX5 showed that selection over this candidate set is where retention above 0.83 comes from, but at 3–7k tokens.

**Next cheapest test.**
- Write the contextual prefixes for RPT-fa first (97 documents; about $0.02 with a small model at index time).
- Re-encode that pool: about 2 minutes on the GPU, plus a BM25 rebuild.
- Replay a1 and a1 + S5 reservation on the 54 questions.
- Go/no-go: at least 0.806 overall, and Persian at least 0.877 or within 1 question of it.
