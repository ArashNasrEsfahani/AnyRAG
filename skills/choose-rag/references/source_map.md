# Sources: cited by title

`evidence.json` cites the journal manuscript (`paper_journal/main.pdf`, 38 pages, in preparation, with no
appendix and no supplement) by the **titles** of its tables, figures and sections, not by their numbers, because
the numbers can change before publication (they already changed once, from the preprint build to the journal
manuscript). Each card's `source.paper` holds the citation in English and Persian, for example:

    Table “Practical recommendations with supporting evidence and confidence”;
    Sect. “Reliability of the rankings and practical guidance”

Table and figure titles are the captions without the "Post hoc." prefix and without the explanatory second
sentence. Section titles are the subsection titles as printed.

Some numbers on the cards were cut from the manuscript when it was shortened, but they remain in the public
results. Those cards also cite the folder that holds them, after the paper titles:

    Sect. “Confirmatory test”; Data release: anyrag-results/posthoc/S5_stats

`anyrag-results` is https://huggingface.co/datasets/ArashNasr/anyrag-results. The inventory of the pools
(documents, characters, chunks) is in the benchmark dataset, https://huggingface.co/datasets/ArashNasr/anyrag-benchmark,
cited as `anyrag-benchmark/corpus_manifest`. `source.file` names the result files the numbers were checked
against. `scripts/choose.py` prints `source.paper` as it stands.

## In the paper

| Content | Cited as |
|---|---|
| question tiers | Table “Definition and examples of the three question tiers” |
| frozen test accuracy, by language half and tier | Table “Test accuracy of four pipelines at two evidence budgets” |
| retrieval components (channel, rewriting, chunk size, embedding) | Table “Retrieval components varied one at a time” |
| evidence funnel, second-judge accuracy | Table “Complete evidence bundles in the candidate list and in the pack, with accuracy under two judges” |
| question language swapped | Table “Retrieval when each question is asked in the other language” |
| recommendations (the guide) | Table “Practical recommendations with supporting evidence and confidence” |
| the four pipelines | Fig. “The four pipelines” |
| budgets and tokens | Fig. “Accuracy at five evidence budgets (left) and against tokens spent per question (right)” |
| sections | Sect. “Benchmark”, “Retrieval and selection pipelines”, “Readers and judging”, “Experimental setup”, “Confirmatory test”, “Retrieval components and where answers are lost”, “Evidence budget and iterative or long-context retrieval”, “Persian and English”, “Reliability of the rankings and practical guidance” |

## Only in the data release

| Content | Cited as (Data release: …) |
|---|---|
| search pools: documents, characters, chunks | `anyrag-benchmark/corpus_manifest` |
| paired statistics of the frozen test, cost and latency, the complete human review | `anyrag-results/test` |
| the earlier, unequal human review | `anyrag-results/test/superseded_asymmetric_review` |
| human verdicts | `anyrag-results/human_review` |
| judges against the human | `anyrag-results/posthoc/S1_evaluator` |
| pack variants, gold-evidence run, packers without an LLM | `anyrag-results/posthoc/S2a_packs` |
| run-to-run noise of the reader and the selection call | `anyrag-results/posthoc/S2b_variance` |
| search agents | `anyrag-results/posthoc/S3_agentic`, `anyrag-results/posthoc/S9_gpu` (hybrid-index agent, latency) |
| long context and routing | `anyrag-results/posthoc/S4_route_long` |
| equivalence, power, pooled model, per-reader contrasts | `anyrag-results/posthoc/S5_stats` |
| closed book, listwise reranker | `anyrag-results/posthoc/S6_baselines` |
| misses by evidence, memory-answerable questions, channels by language, token ratio | `anyrag-results/posthoc/S7_analysis` |
| the selection call at each budget, selection over the reranker | `anyrag-results/posthoc/S8_factorial` |
| retrieval channels | `anyrag-results/posthoc/S10_retrieval` |
| query rewriting, chunk size, embedding model | `anyrag-results/posthoc/S11_rewrite`, `S12_chunks`, `S13_embed` |
| language halves, tokens per character by reader | `anyrag-results/posthoc/S14_bilingual` |
| cross-language questions, spelling variants | `anyrag-results/posthoc/S15_crossling` |

The mapping lives in `work/site/gen_evidence.py` (`REF` and `CITE`); regenerate `evidence.json` from it when a
title changes.
