# Sources: cited by title

`evidence.json` cites the journal manuscript (`paper_journal/main.pdf`, in preparation) by the **titles** of its
tables, figures and sections, not by their numbers, because the numbers can change before publication (they
already changed once, from the preprint build to the journal manuscript). Each card's `source.paper` holds the
citation in English and Persian, for example:

    Table “Practical recommendations with supporting evidence and confidence”;
    Sect. “Reliability of the rankings and practical guidance”

Table and figure titles are the captions without the "Post hoc." prefix and without the explanatory second
sentence. `source.file` names the result files the numbers were checked against. `scripts/choose.py` prints
`source.paper` as it stands.

| Content | Cited as |
|---|---|
| question tiers | Table “Definition and examples of the three question tiers” |
| search pools | Table “Search pools, sources and question counts” |
| pipelines and settings | Table “The four pipelines and their settings” |
| frozen test accuracy | Table “Test accuracy of four pipelines at two evidence budgets” |
| retrieval components | Table “Retrieval components varied one at a time” |
| evidence funnel | Table “Complete evidence bundles in the candidate list and in the pack, with accuracy under two judges” |
| selection step | Table “Effect of the selection step at 6,000 characters” |
| agents, long context, routing | Table “Search agents, long-context reading and routing” |
| memory-answerable questions | Table “Main contrasts with and without questions answerable from memory” |
| tokens per character by reader | Table “Reader prompt tokens per character of evidence in Persian and English” |
| question language swapped | Table “Retrieval when each question is asked in the other language” |
| language halves | Table “Contrasts split by language half” |
| run-to-run noise | Table “Run-to-run noise of the reader and the selection call” |
| pooled model | Table “Mixed-effects logistic model pooling six readers” |
| recommendations (the guide) | Table “Practical recommendations with supporting evidence and confidence” |
| paired statistics, per-reader, cost | Tables “Paired comparisons on the 90 answerable test questions”, “Per-reader contrasts for six readers”, “Query-time cost of each pipeline with the primary reader” |
| judges | Tables “First judge against human review on reviewed answers”, “A second judge against the first judge and the human” |
| token ratio, channels | Tables “Persian-to-English token ratio and tokens at equal budgets”, “Gold evidence found by each retrieval channel in the top 40” |
| where answers are lost; budgets | Fig. “Evidence in the pack for correct and wrong answers”; Fig. “Accuracy at five evidence budgets” |
| sections | Sect. “Benchmark”, “Retrieval and selection pipelines”, “Experimental setup”, “Confirmatory test”, “Retrieval components and where answers are lost”, “Evidence budget and iterative or long-context retrieval”, “Persian and English”, “Reliability of the rankings and practical guidance” |
| appendices | App. “Annotation and construction”, “Extended results of the frozen test”, “Human review and judges”, “Statistical definitions and variance” |

The mapping lives in `work/site/gen_evidence.py` (`REF` and `CITE`); regenerate `evidence.json` from it when a
title changes.
