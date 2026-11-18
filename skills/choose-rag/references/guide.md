# Plain-English decision guide

From *What Reaches the Reader: A Comparison of Retrieval-Augmented Generation Setups in Persian and English*
(Arash Nasr Esfahani, University of Tehran, in preparation). This is the paper's recommendation table
(Table “Practical recommendations with supporting evidence and confidence” in the journal manuscript) rewritten in plain words, with the
context a practitioner needs to use it. Evidence ids in brackets point into `evidence.json`.

## Contents
1. What the study tested (and therefore what it can say)
2. Confidence labels
3. The decision table, row by row
4. Findings outside the table that still matter
5. Where the study does not reach
6. Glossary

## 1. What the study tested

- **Documents.** Six search pools of natural documents: company reports, non-fiction books and statutes, each
  genre once in Persian and once in English. 155 documents, about 38 million characters, 58,218 chunks.
  Born-digital text only, no scans. [bench-size, bench-pools]
- **Questions.** 156 questions; 96 in the one-shot test, 90 of them answerable (45 Persian, 45 English).
  Three tiers of 30 test questions each: T1 single fact in one passage; T2 several parts of one document;
  T3 at least two documents. Each question is annotated with the evidence bundles that suffice. [bench-tiers, bench-annotation]
- **Pipelines.** BM25 + dense retrieval (bge-m3) fused by reciprocal rank fusion ("hybrid"); hybrid + a
  cross-encoder reranker (bge-reranker-v2-m3, "HR"); Lex-RAG, hybrid + one LLM selection call; and an
  adapted set selector (SetR-p+fill). All use 1,000-character chunks. Evidence budgets from 3,000 to
  24,000 characters, with 6,000 and 15,500 the frozen cells. Post hoc: search agents, long context,
  routing, chunk sizes, query rewriting, a second embedding model, cross-language questions. [pipelines]
- **Readers and judge.** One primary reader (GLM-5.3-flash) answers every question; five more readers
  answer the same packs. An LLM judge (gpt-oss-120b) with a strict rubric; a second judge (Mistral Large)
  on part of the arms; one human reviewer on 30 questions.
- **Statistics.** Paired exact sign tests. With 90 questions only differences of about 11 answers are
  detectable (about 8 of 45 within one language half). A difference of 5 needs about 400 to 460
  questions. [power]
- **Fixed in advance vs post hoc.** Only the frozen test (the first three table rows below) was fixed in
  advance (committed in the author's repository, not registered with a third party), and its gate
  failed (inconclusive). Everything else is post hoc on the same 90 questions.
  [frozen-gate]

## 2. Confidence labels

- **Strong**: the difference is larger than the test can detect (about 11 answers, or about 8 in one
  language half) and holds across the reader draws where they exist; or the claim is a measured cost; or
  the question tier leaves no room for a difference.
- **Tentative**: the point estimates agree in direction but lie within the noise of the test.
- **Measured**: a reported number or comparison that the paper's guide table does not grade.

A range such as "62–63" covers answers the judge could not resolve.

## 3. The decision table

There is **no single default**. The suggested starting point is hybrid retrieval followed by a
cross-encoder reranker: it needs no LLM tokens and was ahead of plain hybrid in all nine pairings of reader
draws, but its lead (6 to 9 answers of 90) is not significant and is absent in the Persian half. [ce-vs-hybrid]

| Situation | What to do | Why | Confidence |
|---|---|---|---|
| Single-fact questions (T1) | Hybrid retrieval at a modest budget; no reranker or selection call. | Every pipeline answers 27–29 of 30. | Strong [g-t1] |
| Within-document questions (T2) | Hybrid with a cross-encoder reranker. | Reranker and Lex-RAG 23–24 of 30, plain hybrid 20–21. | Tentative [g-t2] |
| Cross-document questions (T3) | Reranker first. Add an iterative search agent over the hybrid index if ~4x the tokens and ~10 s are acceptable. | Reranker and SetR 18–19 vs hybrid 14 of 30; agent 25 of 30 but not significantly ahead of the reranker. | Tentative [g-t3] |
| Retrieval channel | Run BM25 and dense, fuse them. If only one can run, dense with a larger budget. | Hybrid 62–63, dense 53, BM25 48 of 90 at 6,000 chars; gaps shrink at 15,500. | Strong vs BM25 alone [g-channel-bm25]; Tentative vs dense [g-channel-dense] |
| Chunk size | About 1,000 characters (200 overlap). | Reranked pipeline 69–71 at 1,000, 62 at 500, 51 at 2,000: few large chunks fit the pack. | Strong vs 2,000 at a 6,000-char budget with this packer [g-chunk-2000]; Tentative vs 500 [g-chunk-500] |
| Query rewriting | Do not add HyDE or multi-query. | 61 and 58–59 vs 62–63, at an extra LLM call; fewer complete bundles in Persian. | Tentative [g-rewrite] |
| Embedding model | bge-m3 or multilingual-e5-large; spend effort on a reranker. | Dense-only ties; e5 4–5 lower inside hybrid, not significant. Two models tried. | Tentative [g-embed] |
| Persian budgets | Set budgets in tokens, not characters. | Persian costs ~1.5–1.8x the tokens per character for five of six readers, 3.0x for one (Qwen3). | Strong [g-fa-tokens] |
| Persian spelling and digits | Normalise letters (Arabic ي/ك to ی/ک), non-joiners and digits before BM25. | Without it BM25 loses 5–8 of 45 complete bundles; retrieval only. | Tentative [g-fa-norm] |
| Question language differs from documents | Dense channel required, reranker after the fusion; never BM25 alone. | BM25 finds 2 and 8 of 45 bundles, dense 35 and 28; fusion loses its lead; reranked pipeline 53–54 vs 69–71 same-language. | Strong for the BM25 collapse [g-xl-bm25]; Tentative for fusion vs dense [g-xl-fusion] |
| Tight budget (3,000 to 6,000 characters) | A reranking or selection step: a cross-encoder, or one LLM selection call if no GPU. | Removing the selection call costs ~15 answers at 3,000 (one run) and 5–10 at 6,000. Cross-encoder lead over hybrid at 6,000 not significant. | Strong at 3,000 for the selection call, one run [g-tight-sel3000]; Tentative at 6,000 and for the cross-encoder [g-tight-sel6000, g-tight-ce] |
| Loose budget (15,500 characters or more) | Plain hybrid; reranking and selection add little. | Hybrid rises from ~50 to 72 correct as the budget grows; the reranked pipeline gains less. Costs ~2.4x the tokens of 6,000. | Tentative [g-loose] |
| Latency-sensitive | One pass: hybrid, with or without a reranker. No selection calls or agents. | Median latency 1.8 s hybrid, 3.2 s reranked, 4.0 s Lex-RAG, 5.5 s SetR, 10.5 s agent (measured under load; only the order is reliable). | Tentative [g-latency] |
| No GPU for a reranker | Hybrid with a larger budget, or one LLM selection call over the fused candidates. | Hybrid at 15,500 matches Lex-RAG at 6,000 with 2.5x the text. | Tentative [g-nogpu] |
| Questions answerable from memory | Do not count on retrieval for them; build the evaluation without them. | The reader answers 26 of 90 with no evidence (mostly English classics and EU law; 0 of 30 company reports). | Strong [g-memory] |
| Search agents | Treat the BM25-tool agent's lead with caution. | Its lead over the reranker lies in memory-answerable questions; the hybrid-index agent keeps a lead without them (55 vs 50–51 of 64). | Tentative [g-agents] |

Note on the tight-budget row: the paper labels it "Tight budget (6,000 characters)"; its strong
evidence is at 3,000 characters (one run) and its tentative evidence at 6,000.

## 4. Findings outside the table that still matter

- **Answers are lost before the reader.** Given only the gold evidence (~570 characters) the reader answers
  83 of 90; the best single-pass pipelines reach ~70. For 15–18 questions the complete evidence never
  reaches the top 40 candidates; for 11–15 more it is dropped when the pack is built. So measure what
  reaches the reader (complete evidence bundles in the pack), not only accuracy. [reader-bounds, funnel, partial-misses] (Measured)
- **Do not stack** an LLM selection call on top of a cross-encoder: no measurable gain. [selection-over-ce] (Tentative)
- **Long context is not a substitute for retrieval**: putting the top-ranked documents into a long
  prompt scored 24 of 90; reading whole documents works only when the right documents are already known. [long-context] (Measured)
- **Routing** between a cheap and a wide pipeline by guessed difficulty saved about 4% of tokens. [routing] (Measured)
- **Lexical-only agents** (grep, BM25 tools) gained in the English half only; the hybrid-index agent
  gained in both. Cause unknown: Persian or the Persian documents. [agents-language] (Measured)
- **An agent without a GPU.** Of the tested options that need no GPU, the BM25-tool agent had the highest
  point estimate (73 of 90 against 68–69 for a selection call and 62–63 for plain hybrid; 23 of 30
  cross-document questions against 16 and 14), all within the noise. Its gain lay in the English half
  (+12 against plain hybrid; −2 to −1 in Persian), and its lead over the reranker lay in memory-answerable
  questions. For English documents it is an optional extra; for Persian documents it is not recommended.
  It was not tested on questions in another language than the documents, and neither was any agent.
  [agent-bm25-nogpu, agents-language, g-agents] (Tentative; not in the guide table's no-GPU row)
- **Cost.** Selection calls roughly double the tokens; agents use ~4x the tokens and ~3.6 calls per
  question; a 15,500-character pack costs ~2.4x the tokens of 6,000. [cost-latency, budget-tokens] (Measured)
- **Noise.** Re-running the reader on identical packs moves a pipeline's score by about ±2 answers; the
  ranking of close pipelines changes with the reader draw, the judge, and how evenly a human review covers
  the pipelines. Review all compared setups on the same questions. [reader-noise, two-judges, human-asymmetry] (Measured)

## 5. Where the study does not reach

Say so plainly when a user's situation is here, and mark any advice you give as general practice, not
evidence from this study.

- Languages other than Persian and English (Arabic, Urdu and other Arabic-script languages share the
  normalisation issue in kind, but were not tested).
- Corpora much larger than ~38 M characters / ~58 k chunks, or many more documents per pool; index
  scaling, ANN recall loss and sharding were not studied.
- GraphRAG, knowledge graphs, structured/tabular data, SQL, code search, multimodal or scanned documents
  (OCR), chat history or conversational retrieval, multi-turn agents with memory.
- Other embedding models, rerankers or readers beyond those named; fine-tuned retrievers; ColBERT-style
  late interaction; very long-context readers used as the main design.
- Answer formats beyond short factual answers (long-form summaries, generation of reports), and
  unanswerable-question handling beyond the six test cases (all pipelines abstained correctly).
- Latency under production load: the study's latencies were measured under contention; only the order is
  informative.
- Absolute accuracy on your data: every number is from 90 questions with one primary reader.

## 6. Glossary

- **Hybrid retrieval**: run BM25 (keyword) and dense (embedding) search; merge the two ranked lists with
  reciprocal rank fusion (RRF).
- **Cross-encoder reranker**: a model that scores each (question, passage) pair jointly and reorders the
  candidates; bge-reranker-v2-m3 in the study; runs locally on a GPU; no LLM tokens.
- **LLM selection call**: one extra LLM call that picks which candidate passages go into the pack.
- **Evidence budget / pack**: the maximum amount of evidence text the reader sees per question, and the
  passages actually placed in it.
- **Complete evidence bundle**: a set of passages that together suffice to answer the question.
- **Iterative search agent**: the reader issues several searches (up to five in the study) before answering.
