---
name: choose-rag
description: Recommends a retrieval-augmented generation (RAG) setup for the user's own project from the evidence of a controlled Persian-English RAG study, with a confidence label and paper source for every part. Use whenever someone asks which RAG pipeline, retrieval method (BM25, dense, hybrid), reranker, LLM selection step, chunk size, embedding model, evidence or context budget, query rewriting (HyDE) or agentic / iterative search to use, or wants to design, review, debug or improve a RAG or document question-answering pipeline, search chatbot or enterprise search over documents, in any language, Persian included (e.g. "کدام RAG", "بازیابی", "بازرتبه‌ساز", "اندازهٔ قطعه", "جست‌وجوی سازمانی"). Runs a short interview, applies the study's decision rules with a bundled script, and says plainly where the study's evidence ends.
---

# choose-rag

Help the user pick a RAG setup for their project using the evidence of one study:
*What Reaches the Reader: A Comparison of Retrieval-Augmented Generation Setups in Persian and English*
(Arash Nasr Esfahani, University of Tehran, in preparation). The study compared retrieval channels, chunk
sizes, rerankers, LLM selection, evidence budgets, query rewriting, embedding models and search agents on
natural Persian and English documents, at fixed evidence budgets.

The value of this skill is that every recommendation is traceable to a measured result with an honest
confidence label. That only holds if you stay inside the evidence. Do not add components, numbers or
claims of your own and present them as the study's; when you add general advice, label it as such.

## Files

- `scripts/choose.py`: the decision rules as a deterministic script (Python 3 standard library). Use it
  rather than applying the rules by hand; the rules have order-dependent overrides that are easy to get
  wrong.
- `references/chooser.json`: the 8 questions, the option cards, the 45 rules, the test profiles.
- `references/evidence.json`: 65 evidence items (claim, numbers, source, confidence, caveat).
- `references/guide.md`: the paper's decision table in plain English, what the study tested, and the
  list of situations it does not cover. Read section 5 when the project looks out of scope.
- `references/evaluation_recipe.md`: how the user can check the recommendation on their own questions.
- `scripts/sign_test.py`: the exact sign test used in the recipe.
- `references/source_map.md`: how sources are cited (journal manuscript tables and sections by title).

Paths are relative to this skill's folder. Run scripts with the folder's absolute path, for example
`python <skill-dir>/scripts/choose.py ...` (use `python3` where `python` is not Python 3).

## Workflow

### 1. Infer, then interview

First read what the user already told you (project description, repository, documents, constraints) and
fill in as many of the 8 answers as you can. Then ask only for what is missing or uncertain, in a short
conversational message, not a form. Confirm your inferences in the same message ("I'm assuming English
documents and questions, no GPU; correct me if not"). Two rounds of questions at most; "not sure" is
always an acceptable answer and the rules have a sensible default for it.

The questions (option ids in brackets; `python scripts/choose.py --questions` prints the full wording):

1. **q1 Question type**: single facts in one passage [t1]; facts combined within one document [t2];
   across documents [t3]; a mix [mix]; not sure [unsure].
2. **q2 Language**: English docs and questions [en]; Persian [fa]; both, each question in its documents'
   language [both]; questions in a different language from the documents [cross]; not sure [unsure].
3. **q3 Evidence budget** the reader may see per question: tight ~3,000 characters (~860 tokens English,
   ~1,420 Persian) [tight]; moderate ~6,000 (~1,530 / ~2,660 tokens) [moderate]; generous 15,500+
   (~3,680 / ~6,530 tokens) [generous]; not sure [unsure]. If the user thinks in tokens or in "top-k
   chunks", convert (about 810 characters per 1,000-character chunk on average).
4. **q4 Latency**: under ~2 s [fast]; a few seconds [few]; ~10 s is fine [slow]; not sure [unsure].
5. **q5 GPU for a cross-encoder reranker** (bge-reranker-v2-m3 runs locally, no LLM tokens): [yes] [no] [unsure].
6. **q6 Extra LLM calls per question**: none [none]; one extra [one]; several if accuracy improves [many]; not sure [unsure].
7. **q7 Could the model already know the answers?** private or recent documents [private]; public,
   well-known texts (classic books, major laws) [public]; not sure [unsure].
8. **q8 Persian spelling varies?** (Arabic ي/ك vs Persian ی/ک, missing non-joiners, mixed digits):
   [clean] [mixed] [unsure]. Ask only if Persian may be involved (q2 is fa, both, cross or unsure).

Useful mappings: a hosted-API-only stack or CPU-only server means q5 = no. "Compare clauses across
contracts", "what changed between versions", "summarise across reports" are t3. A support bot answering
from FAQ articles is mostly t1, unless answers need several sections. A chat UI with streaming usually
tolerates "a few seconds"; a search-as-you-type box is "fast".

### 2. Run the rules

```
python <skill-dir>/scripts/choose.py --q1 t3 --q2 fa --q3 moderate --q4 slow --q5 yes --q6 many --q7 private --q8 mixed
```

Unanswered questions default to `unsure`. Add `--lang fa` for Persian card text, `--format json` for
machine-readable output, `--no-evidence` to drop the evidence details. The output lists the six pipeline
slots, the extras, what to avoid, the trade-offs triggered by the answers, every evidence item with its
confidence, source and caveat, and the rules that fired.

If you cannot run code (some chat surfaces), apply `references/chooser.json` by hand: start every slot
from rule r01, then go through `rules` in order; a rule fires when every `when` key matches (list =
any-of, `!key` = none-of, `q1..q8` = answers, other keys = the slot's current value); `set` overwrites,
`add` appends if absent, `remove` deletes. Say that you applied the rules by hand.

### 3. Present the recommendation

Write it for the user, in their language (Persian if they wrote in Persian; use `--lang fa` for the card
text). Do not paste the raw script output; turn it into this structure:

1. **Recommended pipeline**: one line per slot, in this order: retrieval channels, chunking, reranking
   or selection, evidence budget, search agent, normalisation. Use concrete settings (e.g. "BM25 +
   bge-m3 dense, merged with reciprocal rank fusion; ~1,000-character chunks with 200 overlap"). A tiny
   flow line such as `query -> BM25 + dense -> RRF -> reranker -> 6,000-char pack -> reader` helps.
2. **Also do**: the extras.
3. **Avoid**: the avoid items, each with a few words of why.
4. **Evidence**: for each slot and avoid item, the key number, its confidence label exactly as given
   (**Strong**, **Tentative** or **Measured**, with its scope, e.g. "Strong against BM25 alone,
   Tentative against dense alone"), and the paper source (journal manuscript tables and sections by title,
   as the script prints them; numbers are not used because they may change before publication). Keep it compact: a table or one line each.
5. **Trade-offs**: the trade-off cards, which explain tensions between the user's own answers. If none
   fired, mention the main cost of the recommendation (tokens, latency) in one line.
6. **What the study does not cover here**: anything in the project outside the study's scope (see
   below), with any general advice clearly marked "general practice, not from this study".
7. **Caveat and how to check it**: always include this: the evidence rests on 90 test questions, one
   primary reader (GLM-5.3-flash) and one judge, and most analyses are post hoc; the data do not settle a
   single default. Then give the 4-step check from `references/evaluation_recipe.md` in short form
   (≥50 own questions, closed-book run to drop memory-answerable ones, run A and B on the same questions
   with blind scoring, `scripts/sign_test.py` on the discordant pairs) and name the specific A-vs-B
   comparison that matters most for this user (usually the slot whose confidence is Tentative and whose
   cost is highest).

Keep the whole answer readable in a couple of minutes. Numbers are "of 90 questions" or "of 30 per tier"
unless stated; ranges like 62–63 cover answers the judge could not resolve. Do not convert Tentative into
"recommended" language such as "clearly better"; Tentative means the point estimates agree but are within
noise.

### 4. Follow-ups

If the user changes an answer, rerun the script rather than editing the old result. If they ask "why",
quote the evidence item (claim, numbers, caveat) from `references/evidence.json`. If they ask about a
component the study did not test, say so before giving any general view.

## Staying inside the evidence

The study tested: Persian and English; company reports, non-fiction books and statutes; 155 documents,
~38 M characters, ~58 k chunks; bge-m3 and multilingual-e5-large embeddings; bge-reranker-v2-m3; BM25;
RRF fusion; budgets 3,000 to 24,000 characters; one primary reader plus five replay readers; short
factual answers.

When the project goes beyond this, say so explicitly and keep the study's recommendation only for the
parts it covers. Common cases (details in `references/guide.md` section 5):

- other languages: the channel, chunk and budget findings were measured on Persian and English only; for
  another Arabic-script language, normalisation is plausible by analogy but untested;
- much larger corpora (millions of documents), sharded or approximate indexes;
- GraphRAG, knowledge graphs, tables, SQL, code, images, scanned PDFs / OCR;
- multi-turn chat with conversation memory, long-form or summarising answers;
- other rerankers, embeddings, fine-tuned retrievers, very long-context readers.

For these, you may add your general knowledge, under a heading such as "General practice (not from
this study)", and suggest the evaluation recipe to test it.

## Example

User: "I'm building a Q&A bot over ~2,000 internal English policy PDFs. We only have API access to
GPT-class models, no GPU. Answers in under 5 seconds. Most questions need a couple of sections of one
policy."

Inferred: q1 t2, q2 en, q4 few, q5 no, q7 private. Ask: budget (q3) and whether one extra LLM call is
acceptable (q6). With q3 moderate, q6 one:

```
python <skill-dir>/scripts/choose.py --q1 t2 --q2 en --q3 moderate --q4 few --q5 no --q6 one --q7 private
```

gives hybrid + 1,000-char chunks + one LLM selection call over the fused candidates + ~6,000 characters
+ no agent, with a trade-off: a larger budget (~15,500 characters) for plain hybrid reached similar
accuracy (68 vs 68–69 of 90) without the extra call. Present it as above, and suggest comparing those
two on their own questions.
