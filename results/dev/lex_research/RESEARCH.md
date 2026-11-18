# LEX research: how to beat a cross-encoder without using one

Written 2026-09-26. Every number below links to its source. When the setup differs from ours, the caveat is stated. "n/f" means not found in the sources checked. Numbers I estimated myself, not taken from a paper, are marked **(est.)**.

## 1. Summary

The strongest published signal for our setup is the **bge-m3 paper itself**. It reports MIRACL nDCG@10 for its "All" mode, which combines dense, sparse and multi-vector scores with weights 1 : 0.3 : 1. For Persian (fa) that score is 63.4, against 60.9 for dense alone and 45.1 for sparse alone. Averaged over languages, "All" scores 71.5 and dense 69.2 ([bge-m3](https://arxiv.org/html/2402.03216v4)). Three other papers put bge-reranker-v2-m3 on MIRACL-fa at 60.4 to 61.9, averaged over languages at 67.6 to 69.3 ([Querit](https://arxiv.org/html/2606.19037), [jina-reranker-v3](https://arxiv.org/html/2509.25085), [LAMAR](https://arxiv.org/html/2607.22042)). Those numbers use different first-stage retrievers, and bge-m3 trained on MIRACL training data, so this is not a head-to-head result. Still, on Persian, bge-m3's multi-signal fusion sits in the same range as the reranker, with no cross-encoder involved. It is the first thing to test.

The second lever is **index-time LLM enrichment**, which our token gate does not charge for:
- **Contextual chunk prefixes.** Anthropic reports that they cut the top-20 retrieval failure rate by 35% (embeddings only) and by 49% (embeddings plus BM25). Adding a reranker brought the cut to 67% ([Anthropic](https://www.anthropic.com/news/contextual-retrieval)).
- **doc2query-style synthetic questions.** docTTTTTquery raised MS MARCO MRR@10 from 18.6 to 27.2 for about 9 ms of extra query time ([castorini](https://github.com/castorini/docTTTTTquery)).

The third lever is **tuned convex-combination fusion instead of RRF**, which beat RRF on every dataset in Bruch et al. ([arXiv 2210.11934](https://arxiv.org/html/2210.11934)).

For T3 (cross-document) questions, **query-adaptive MMR or diversity in pack selection** is cheap, but the evidence is mixed: DF-RAG reports +4–10% F1 ([DF-RAG](https://arxiv.org/abs/2601.17212)), while ARAGOG found no gain from MMR ([ARAGOG](https://arxiv.org/abs/2404.01037)).

**LLM listwise reranking** (RankGPT, RankZephyr) matches or beats cross-encoders. But on our 1,000–2,000-character chunks it costs tens of thousands of tokens per query, which fails the 1.5x token gate.

## 2. Evidence by family

### 2.1 Learned sparse / lexical
| Method | Reported gain (incl. vs cross-encoder) | Query-time cost | Persian / multilingual | Source |
|---|---|---|---|---|
| bge-m3 sparse (lexical weights) | MIRACL nDCG@10: fa 45.1 vs BM25 28.7; avg 53.9 vs BM25 31.9. On MLDR (long docs), sparse 62.2 beats dense 52.5 | Comes from the same forward pass as dense; inverted-index lookup | fa reported | [bge-m3](https://arxiv.org/html/2402.03216v4) |
| SPLADE-v3 (English) | MS MARCO MRR@10 40.2; BEIR-13 avg 51.7. MiniLM cross-encoder on top: overall effect "close to 0". DeBERTaV3 cross-encoder does beat it on most query sets | Sparse inverted index, low ms | English only (no multilingual v3 in the report) | [SPLADE-v3](https://arxiv.org/html/2403.06789) |
| MILCO (multilingual LSR, 2025) | MIRACL avg 72.3; fa 62.3 (vs bge-m3 sparse 45.1, bge-m3 dense 60.9); +34.1% over bge-m3 sparse | about 0.44 ms/query at pruning p=60 (vs 1.29 ms for Qwen3-Embed-0.6B); 10x smaller index | fa reported; open weights | [MILCO](https://arxiv.org/html/2510.00671v2), [code](https://github.com/thongnt99/milco) |

### 2.2 Late interaction (multi-vector)
| Method | Reported gain | Query-time cost | Persian / multilingual | Source |
|---|---|---|---|---|
| bge-m3 multi-vector (ColBERT-style) | MIRACL fa 62.0 vs dense 60.9; avg 70.5 vs 69.2. Used to **rerank the top-200 from dense** ("considering its heavy cost") | One query encode (shared) plus MaxSim over 200 candidates' stored token vectors: cheap on CPU for ≤200 chunks **(est.)**; storage-heavy | fa reported | [bge-m3](https://arxiv.org/html/2402.03216v4) |
| Jina-ColBERT-v2 | BEIR avg 53.1 (ColBERTv2 49.6, BM25 44.0); MIRACL avg 62.3; fa 56.3 | Matryoshka 64-dim loses about 1.6% nDCG | fa reported, below bge-m3 | [Jina-ColBERT-v2](https://arxiv.org/html/2408.16672) |
| ColBERTv2 + PLAID | Up to 7x faster (GPU) and 45x faster (CPU) than vanilla ColBERTv2; tens of ms on GPU, tens to a few hundred ms on CPU at 140M passages | Index-time engine only | English model | [PLAID](https://arxiv.org/abs/2205.09707) |
| Reference: cross-encoder baseline | bge-reranker-v2-m3 on MIRACL: fa 60.35, avg 67.64 (Qwen3-Emb-0.6B top-100; first stage fa 53.33, avg 59.87). fa 61.92, avg 69.32 (jina-emb-v3 top-100; first stage fa 54.13). fa 61.7, avg 69.1 (LAMAR table). BEIR 56.51. RusBEIR: bge-m3 61.13 goes to 65.85 with the reranker; BM25+reranker 59.87 | 0.6B cross-encoder over top-k | fa reported | [Querit](https://arxiv.org/html/2606.19037), [jina-reranker-v3](https://arxiv.org/html/2509.25085), [LAMAR](https://arxiv.org/html/2607.22042), [RusBEIR](https://arxiv.org/html/2504.12879) |

The official bge-reranker-v2-m3 card says the model reranks bge-m3's top-100 on MIRACL, but it gives its numbers only as images ([HF card](https://huggingface.co/BAAI/bge-reranker-v2-m3)). No MIRACL-fa figure for bge-m3 plus bge-reranker-v2-m3 appears as text: **n/f**.

### 2.3 Index-time enrichment (free under our token gate)
| Method | Reported gain | Query-time cost | Persian / multilingual | Source |
|---|---|---|---|---|
| Anthropic Contextual Retrieval (50–100-token LLM prefix per chunk, applied to both the embedding and BM25) | Top-20 failure rate falls from 5.7% to 3.7% (contextual embeddings, -35%) and to 2.9% (+ contextual BM25, -49%). **With the reranker (Cohere)** it falls to 1.9% (-67%). So the prefix alone gets about two-thirds of the way to prefix+reranker. Indexing cost about $1.02 per million document tokens (with caching) | Zero extra query-time cost | Language-agnostic (prompt the LLM in the document's language) | [Anthropic](https://www.anthropic.com/news/contextual-retrieval) |
| Contextual retrieval vs late chunking (2025 study) | Contextual retrieval "preserves semantic coherence more effectively" at higher compute cost. Late chunking is more efficient but "sacrifice[s] relevance and completeness" | Zero | n/f | [Reconstructing Context](https://arxiv.org/abs/2504.19754) |
| docTTTTTquery / doc2query | MS MARCO MRR@10: BM25 18.6, +doc2query 21.8, +docTTTTTquery 27.2. Latency 55 to 64 ms | +9 ms | English T5 (an LLM can generate Persian questions instead) | [castorini](https://github.com/castorini/docTTTTTquery) |
| Doc2Query-- (filter hallucinated expansions) | Up to +16% effectiveness over Doc2Query; -23% query time; -33% index size | Faster than doc2query | Method is language-agnostic | [Doc2Query--](https://arxiv.org/abs/2301.03266) |
| HyPE (index-time hypothetical questions, question-to-question matching) | Claims up to +42 pts context precision and +45 pts claim recall (small-scale evaluation) | No runtime LLM call | n/f | [HyPE](https://arxiv.org/abs/2607.29402) |
| Proposition indexing (Dense X) | Contriever: Recall@5 +9.3 pts (+22.5% relative), Recall@20 +7.7. GTR: Recall@20 +2.7. Under a 500-token QA budget: +2.7 to +4.1 EM | Bigger index; retrieves smaller units | English | [Dense X](https://arxiv.org/html/2312.06648) |
| RAPTOR (recursive summary tree) | +20% absolute accuracy on QuALITY with GPT-4 | Retrieves over tree nodes: cheap | Language-agnostic via LLM | [RAPTOR](https://arxiv.org/abs/2401.18059) |
| Late chunking (Jina) | About +1.5 to +1.9 nDCG absolute (+2.7 to +3.6% relative) over naive chunking on BEIR subsets. Hurts where relevant text sits inside unrelated text | Zero | Needs a long-context embedder | [Late chunking](https://arxiv.org/html/2409.04701) |

### 2.4 Query-side cheap tricks (PRF)
| Method | Reported gain | Query-time cost | Persian / multilingual | Source |
|---|---|---|---|---|
| RM3 (lexical PRF) | Gains are "unstable"; RM3 "often fails to generalize and even underperforms vanilla BM25" on BEIR. It also hurts BERT rerankers | Second BM25 pass | Language-agnostic but tokenizer-sensitive | [GPRF](https://arxiv.org/html/2510.25488v1), [Li et al. TOIS](https://dl.acm.org/doi/10.1145/3570724) |
| Vector PRF (Rocchio / average on dense) | Helps dense retrievers "over several evaluation metrics" when the query keeps at least 50% weight and feedback is shallow (few docs) | One extra ANN search | Model-agnostic | [Li et al. 2108.11044](https://arxiv.org/abs/2108.11044) |
| Offline PRF | Moves PRF to index time (pseudo-queries), so retrieval is single-pass; effectiveness kept on TREC DL | Single pass | n/f | [Offline PRF](https://arxiv.org/abs/2308.10191) |
| PromptPRF | LLM extracts query-independent features **offline**; small dense models approach large ones on TREC DL/BEIR (no single headline number in the abstract) | No runtime LLM | n/f | [PromptPRF](https://arxiv.org/abs/2503.14887) |

### 2.5 Multi-signal fusion
| Method | Reported gain | Query-time cost | Persian / multilingual | Source |
|---|---|---|---|---|
| Convex combination (TM2C2) vs RRF | MS MARCO NDCG@1000: 0.454 vs 0.425 (RRF k=60). HotpotQA 0.699 vs 0.675; FEVER 0.744 vs 0.721. "CC outperforms RRF in in-domain and out-of-domain." RRF is sensitive to its parameters. The alpha weight converges with under 5% of the training queries | ~0 | Method-level | [Bruch et al.](https://arxiv.org/html/2210.11934) |
| bge-m3 weighted "All" (dense : sparse : multi-vector = 1 : 0.3 : 1) | MIRACL: fa 63.4 vs dense 60.9 (+2.5); avg 71.5 vs 69.2 (+2.3). Dense+Sparse alone: fa 62.3, avg 70.4. MLDR (long docs) used different weights (0.15 : 0.5 : 0.35): 65.0 vs dense 52.5 | One encode plus rescoring of candidates | fa reported | [bge-m3](https://arxiv.org/html/2402.03216v4) |

### 2.6 MMR / diversity for multi-document questions
| Method | Reported gain | Query-time cost | Persian / multilingual | Source |
|---|---|---|---|---|
| DF-RAG (query-adaptive MMR / gMMR) | +4–10% F1 over vanilla cosine RAG on reasoning-intensive QA; recovers up to 91.3% of an 18-pt oracle headroom | Test-time tuning; no fine-tuning (the abstract does not say whether it needs LLM calls) | n/f | [DF-RAG](https://arxiv.org/abs/2601.17212) |
| Dartboard (relevant information gain) | SOTA on RGB QA versus methods that trade off relevance and diversity separately (no number in the abstract) | Cheap set selection | n/f | [Dartboard](https://arxiv.org/abs/2407.12101) |
| ARAGOG (negative result) | "MMR and Cohere rerank did not exhibit notable advantages over a baseline Naive RAG" | — | — | [ARAGOG](https://arxiv.org/abs/2404.01037) |
| DPPs for RAG (2026) | Density plus diversity via DPP (numbers n/f in what I read) | Cheap | n/f | [DPP-RAG](https://arxiv.org/abs/2604.03240) |

### 2.7 LLM rerankers with small budgets
| Method | Reported gain | Per-query cost | Persian / multilingual | Source |
|---|---|---|---|---|
| RankGPT (GPT-4 listwise, window 20, step 10, top-100) | BEIR-8 avg 53.68 vs monoT5-3B 51.36, Cohere-v2 49.45, BM25 43.42. Mr.TyDi avg 62.93 vs mmarcoCE 60.28 | $0.596 per query (GPT-4, top-100); $0.098 (top-30) | Mr.TyDi (no fa in Mr.TyDi) | [RankGPT](https://arxiv.org/html/2304.09542) |
| RankZephyr (7B, open) | DL19 0.7816, DL20 0.8159; on par with RankGPT-4 | Sliding window 20/10 over top-100, i.e. many calls | English-trained | [RankZephyr](https://arxiv.org/html/2312.02724) |
| FIRST (single-token listwise) | About 50% faster inference while keeping ranking quality; gains across BEIR | One window forward pass, 1 output token | English | [FIRST](https://arxiv.org/abs/2406.15657) |
| Setwise | Fewer LLM inferences and prompt tokens than pairwise and listwise at similar effectiveness | Still several calls | n/f | [Setwise](https://arxiv.org/abs/2310.09497) |

**(est.) Token arithmetic for our setup:** 50 chunks × 1,000–2,000 characters is about 12k–40k tokens of input (Persian tokenizes worse than English). The reader pack is 6,000 characters, roughly 2k–4k tokens. Any listwise pass over top-50 is therefore **3x to 10x or more** the per-question tokens and fails the 1.5x gate. What could fit: one FIRST-style pass over the top-10 to 15 with passages cut to about 60–100 tokens each (about 1k–1.5k tokens). That still adds an LLM round-trip to latency, so p95 is at risk.

## 3. Ranked shortlist for LEX
Constraints: no query-time cross-encoder; index time is free; latency and tokens at most 1.5x; Persian and English; must help T3.

1. **bge-m3 tri-signal fusion (dense + sparse + multi-vector) as the core scorer.** On MIRACL-fa it matches or exceeds published bge-reranker-v2-m3 numbers, using the same encoder we already run.
   - *Test:* score the top-200 hybrid candidates with bge-m3 multi-vector MaxSim. Fuse dense, sparse and multi-vector, keeping BM25 as a fourth signal. Compare Recall@k and pack-hit on the dev set against HYBRID and HYBRID+CE.
2. **Contextual chunk prefixes at index time.** The LLM writes each prefix in the document's language, naming the entity (company), period, section and document type. The prefix is added to both the lexical and the dense representation. This should matter most on the 97-report Persian pool, where chunks omit the company name and year.
   - *Test:* re-index with prefixes and measure the failure@k reduction, overall and per tier.
3. **Tuned convex-combination fusion instead of RRF.** Bruch et al. find it beats RRF and needs few tuning queries.
   - *Test:* per-signal normalisation (min-max or theoretical-min), then grid-search the weights on the dev split only. Report on test to avoid overfitting to 54 questions.
4. **Index-time synthetic questions (doc2query--/HyPE) as an extra field or signal.** This turns question-to-passage matching into question-to-question matching, at zero query-time LLM cost.
   - *Test:* generate 3–5 Persian or English questions per chunk and filter them doc2query-- style (drop questions whose own retrieval score is low). Add them as a separate BM25 field and a max-over-questions dense score, fused in item 3.
5. **Document-aware pack assembly for T3.** Use query-adaptive MMR, or a per-document quota such as "at least 2 documents if the second document's score is within x of the first". Also aggregate chunk scores per document (max or sum of the top chunks).
   - *Test:* adaptive versus always-on (ARAGOG saw no MMR gain on naive QA). Measure the T3 "2+ gold documents in pack" rate and T1 regression separately.
6. **Cheap query-side boosts, gated.** Options: Rocchio dense PRF (query weight at least 0.5, top-3 feedback docs); offline PRF / PromptPRF features; or MILCO as a stronger multilingual sparse signal (fa 62.3 vs bge-m3 sparse 45.1).
   - *Test:* dense PRF latency (one extra ANN search) and whether MILCO adds over bge-m3 sparse inside the fusion. Treat RM3 as a likely negative.

Not recommended as the core: LLM listwise reranking (token gate). RAPTOR and propositions are optional index-time extras worth a later ablation for T2/T3, but they change pack granularity, and the 6,000-character pack must be re-budgeted.

## 4. Caveats (where published gains may not transfer)
- **The MIRACL comparison is not controlled.** bge-m3 was fine-tuned with MIRACL training data ([bge-m3](https://arxiv.org/html/2402.03216v4)). The reranker numbers use other first stages (Qwen3-Emb, jina-emb-v3). Wikipedia passages are also not company reports or laws. The +2.5 fa gap may shrink or reverse on our pool.
- **Recall is not answer accuracy.** Our gate is +3/54 answers. Anthropic's gains are failure@20, and the Bruch and bge-m3 gains are nDCG. The 6,000-character pack holds only about 3–6 chunks, so rank-1 to 5 precision matters more than Recall@20.
- **Anthropic's numbers are proprietary.** They use Voyage/Gemini embeddings and a Cohere reranker on English codebases, fiction and papers; nothing in them is Persian. The reranker still added a further 18-pt cut in failure rate (-49% to -67%), so a cross-encoder remains a strong control to beat.
- **SPLADE-v3's parity with the MiniLM cross-encoder is English, MS MARCO-trained** ([SPLADE-v3](https://arxiv.org/html/2403.06789)). DeBERTa cross-encoders still win there, and bge-reranker-v2-m3 is stronger than MiniLM.
- **MLDR weights differ from MIRACL weights.** Sparse dominates long documents. Our 1,000–2,000-character chunks sit between the two, so weights must be tuned, not copied.
- **Persian lexical traps.** Zero-width non-joiner (ZWNJ), Arabic vs Persian yeh and kaf, and Persian/Arabic/Latin digits all hit BM25, sparse and doc2query alike. Normalise before indexing and querying.
- **doc2query hallucinates.** Filtering (doc2query--) matters ([Doc2Query--](https://arxiv.org/abs/2301.03266)). Generated Persian questions may be translation-flavoured.
- **Small-sample overfitting.** 54 test questions make +3 about 5.6 pts, so tune only on dev, and pre-register the fusion weights.
- **The MMR evidence is mixed** (DF-RAG positive; ARAGOG null), and diversity can push the single gold chunk out of the pack on T1.
- **HyPE's +42/+45-pt claims come from a small evaluation;** treat them as directional.
- **Not found:** a published MIRACL-fa figure for bge-m3 + bge-reranker-v2-m3 as text; any Persian doc2query or contextual-retrieval study; late-chunking results in Persian.
