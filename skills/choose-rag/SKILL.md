---
name: choose-rag
description: Recommends a retrieval-augmented generation (RAG) setup for the user's own project from the evidence of a Persian-English RAG comparison study, with a confidence label and paper source for every part. Use whenever someone asks which RAG pipeline, retrieval method (BM25, dense, hybrid), reranker, LLM selection step, chunk size, embedding model, context budget, query rewriting (HyDE) or agentic search to use, or wants to design, review, debug or improve a RAG, document Q&A or enterprise-search pipeline, e.g. "which RAG should I use for this project", "review my RAG setup", "choose retrieval for my repo", in any language, Persian included ("کدام RAG", "بازیابی", "بازرتبه‌ساز", "اندازهٔ قطعه"). Reads the user's codebase and documents with a read-only scanner, asks only what it cannot infer, applies the study's decision rules with a bundled script, suggests file-by-file changes to an existing stack, and says plainly where the evidence ends.
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

- `scripts/scan_project.py`: a read-only scanner for the user's project folder. It infers the eight
  chooser answers (with confidence and file:line evidence), the document corpus, the existing RAG stack
  and the resources, and with `--recommend` runs the rules and lists the changes to make in the user's
  files. Python 3 standard library only.
- `scripts/choose.py`: the decision rules as a deterministic script (Python 3 standard library). Use it
  rather than applying the rules by hand; the rules have order-dependent overrides that are easy to get
  wrong.
- `references/chooser.json`: the 8 questions, the option cards, the rules, the test profiles.
- `references/evidence.json`: 65 evidence items (claim, numbers, source, confidence, caveat).
- `references/guide.md`: the paper's decision table in plain English, what the study tested, and the
  list of situations it does not cover. Read section 5 when the project looks out of scope.
- `references/evaluation_recipe.md`: how the user can check the recommendation on their own questions.
- `scripts/sign_test.py`: the exact sign test used in the recipe.
- `references/source_map.md`: how sources are cited (journal manuscript tables, figures and sections by title,
  plus folders of the public data release for numbers cut from the manuscript).
- `scripts/test_choose.py`, `scripts/test_scan.py`: self-tests (33 rule profiles; 4 synthetic projects).

Paths are relative to this skill's folder. Run scripts with the folder's absolute path, for example
`python <skill-dir>/scripts/choose.py ...` (use `python3` where `python` is not Python 3).

## Workflow

Use step 1 when there is a project to look at (you are working inside a repository, the user names a
folder, or documents are attached). Use step 2 alone when there is no codebase. Both end in step 3.

### 1. Scan the project, then ask only what is missing

**Run the scanner** on the project root (the current working directory unless the user names another):

```
python <skill-dir>/scripts/scan_project.py <project-dir>            # slot table + profile
python <skill-dir>/scripts/scan_project.py <project-dir> --json     # same, machine-readable
```

It is safe to run on any folder: it only reads files, never executes or imports project code, never uses
the network and writes nothing. It skips `.git`, `node_modules`, virtual environments, build and cache
folders, binary files and `.env` files, and it caps the files and bytes it reads (the output says when a
cap was hit). Tell the user in one line that you scanned their project locally.

The output has, for each slot q1..q8, a **value** (a chooser option id), a **confidence** (high, medium,
low, unknown) and the **evidence** (file:line or counts), plus `ask` (slots that are unknown or low) and
`confirm` (medium or high). It also lists the corpus (folders, file types, count, size, rough characters,
script mix such as "Persian/Arabic script 92%, Latin 8%"), the query side (language of test questions,
UI strings, prompts), the stack (vector DB, BM25, embedding model, reranker, framework, fusion, query
rewriting, agent, Persian normalisation, chunk_size, overlap, top_k, context budgets) and the resources
(GPU hints, LLM providers, latency hints).

**If Python is not available**, do the same with your own file tools and say so: Glob for document
folders and types (pdf, md, html, docx, txt, json) outside code folders; Read a few documents and estimate
the share of Persian/Arabic versus Latin letters; Grep the code and config for `faiss|chroma|qdrant|
weaviate|pgvector|milvus|lancedb|elasticsearch|opensearch`, `rank_bm25|bm25s|BM25`, `bge-m3|e5|
text-embedding|SentenceTransformer`, `reranker|CrossEncoder|rerank`, `langchain|llama_index|haystack|dspy`,
`chunk_size|chunk_overlap|top_k|k=|n_results`, `torch.cuda|cuda|cu12|nvidia`, `openai|anthropic|
openrouter|ollama|vllm`, `stream=True|websocket|autocomplete`, and test-question files. Fill the same
slot table with honest confidence levels.

**Show the profile and ask.** Give the user a compact table of what you inferred (slot, value, confidence,
one piece of evidence) and the stack in a line or two. Then ask **only** the slots in `ask`, in one short
conversational message, and confirm the rest in a single sentence ("I read your documents as Persian with
mixed ي/ی spelling and a GPU in your Dockerfile; correct me if not"). Two rounds at most; "not sure" is
always an acceptable answer. Things to keep in mind:

- q3 inferred from top_k x chunk size is what the code sends **now**; the question is what the reader
  **may** see, so phrase it as a confirmation.
- q5 from `torch.cuda` alone is weak (code often falls back to CPU); ask whether production has a GPU.
- q6 inferred from an existing agent or rewriting step only shows extra calls are tolerated today.
- q1 and q7 are rarely visible in code; ask them unless test questions or the README settle them.
- If the documents are not in the repository (a bucket, a database), ask where they are or ask q2/q8
  directly; offer to scan a local copy.
- Anything the scanner lists under "Outside the study" goes into the scope section of your answer.

### 2. Interview (no codebase)

Read what the user already told you and fill in as many of the 8 answers as you can, then ask only for
what is missing, in a short message, confirming your inferences in the same message. The questions
(option ids in brackets; `python scripts/choose.py --questions` prints the full wording):

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

### 3. Run the rules

With a scanned project, pass the confirmed answers back to the scanner; it runs `choose.py` and maps the
recommendation onto the user's files:

```
python <skill-dir>/scripts/scan_project.py <project-dir> --recommend --q1 t2 --q4 few --q6 one
```

Flags override the scan; unanswered slots keep the scanned value, or `unsure`. Without a project, or to
see the full evidence cards, run the rules directly:

```
python <skill-dir>/scripts/choose.py --q1 t3 --q2 fa --q3 moderate --q4 slow --q5 yes --q6 many --q7 private --q8 mixed
python <skill-dir>/scripts/choose.py --json scan.json --q4 few     # scan.json from scan_project.py --json
```

Add `--lang fa` for Persian card text, `--format json` for machine-readable output, `--no-evidence` to
drop the evidence details. The output lists the six pipeline slots, the extras, what to avoid, the
trade-offs triggered by the answers, every evidence item with its confidence, source and caveat, and the
rules that fired.

If you cannot run code (some chat surfaces), apply `references/chooser.json` by hand: start every slot
from rule r01, then go through `rules` in order; a rule fires when every `when` key matches (list =
any-of, `!key` = none-of, `q1..q8` = answers, other keys = the slot's current value); `set` overwrites,
`add` appends if absent, `remove` deletes. Say that you applied the rules by hand.

### 4. Present the recommendation

Write it for the user, in their language (Persian if they wrote in Persian; use `--lang fa` for the card
text). Do not paste the raw script output; turn it into this structure:

1. **Recommended pipeline**: one line per slot, in this order: retrieval channels, chunking, reranking
   or selection, evidence budget, search agent, normalisation. Use concrete settings (e.g. "BM25 +
   bge-m3 dense, merged with reciprocal rank fusion; ~1,000-character chunks with 200 overlap"). A tiny
   flow line such as `query -> BM25 + dense -> RRF -> reranker -> 6,000-char pack -> reader` helps.
2. **Changes to your code** (only when an existing stack was detected): the `changes` list from
   `--recommend`, file by file, each with its location, the change and its confidence. Lead with
   `change` items, then `scope` and `optional`, and end with one line naming what already matches
   (`keep`). Examples: "`rag/config.py:12` sets `chunk_size=2000`: consider about 1,000 characters with
   200 overlap (Tentative, at a 6,000-character budget)"; "`app/retrieve.py:40`: add a cross-encoder
   reranker after your Chroma query and re-score the top ~50 candidates (Tentative)". Do not add changes
   the list does not support; your own engineering advice (how to wire it in their framework) is fine
   but label it general practice. Before suggesting edits, read the cited lines so the advice fits the
   code; edit files only if the user asks.
3. **Also do**: the extras.
4. **Avoid**: the avoid items, each with a few words of why.
5. **Evidence**: for each slot, change and avoid item, the key number, its confidence label exactly as
   given in `evidence.json` and the option card (**Strong**, **Tentative** or **Measured**, with its
   scope, e.g. "Strong against BM25 alone, Tentative against dense alone"), and the paper source
   (journal manuscript tables and sections by title, and any data release folder, as the script prints
   them; table numbers are not used because they may change before publication). Keep it compact: a
   table or one line each.
6. **Trade-offs**: the trade-off cards, which explain tensions between the user's own answers. If none
   fired, mention the main cost of the recommendation (tokens, latency) in one line.
7. **What the study does not cover here**: anything in the project outside the study's scope (see
   below and the scanner's "Outside the study" list), with any general advice clearly marked "general
   practice, not from this study".
8. **Caveat and how to check it**: always include this: the study tested Persian and English only, on
   155 documents, with 90 test questions, one primary reader (GLM-5.3-flash) and one judge, and most
   analyses are post hoc; the data do not settle a single default. Then give the 4-step check from
   `references/evaluation_recipe.md` in short form (≥50 own questions, closed-book run to drop
   memory-answerable ones, run A and B on the same questions with blind scoring, `scripts/sign_test.py`
   on the discordant pairs) and name the specific A-vs-B comparison that matters most for this user
   (usually the change whose confidence is Tentative and whose cost is highest).

Keep the whole answer readable in a couple of minutes. Numbers are "of 90 questions" or "of 30 per tier"
unless stated; ranges like 62–63 cover answers the judge could not resolve. Do not convert Tentative into
"recommended" language such as "clearly better"; Tentative means the point estimates agree but are within
noise.

### 5. Follow-ups

If the user changes an answer, rerun the script rather than editing the old result. If they ask "why",
quote the evidence item (claim, numbers, caveat) from `references/evidence.json`. If they ask about a
component the study did not test, say so before giving any general view. If they ask you to apply the
changes, edit their code one change at a time and keep their framework's idioms.

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
- other rerankers, embeddings (for example OpenAI embeddings), fine-tuned retrievers, very long-context
  readers.

For these, you may add your general knowledge, under a heading such as "General practice (not from
this study)", and suggest the evaluation recipe to test it.

## Examples

**Repository.** User, inside their repo: "Review my RAG setup." Run `scan_project.py .`. Suppose it finds
5 Persian statute files with 40% Arabic ي/ك forms (q2 fa high, q8 mixed high), langchain + Chroma with
multilingual-e5, `chunk_size=2000` in `app/rag/config.py:2`, `k=4` in `app/retrieve.py:8` (q3 moderate,
medium), an OpenAI reader and no GPU hints. Ask q1, q4, q5, q6 and q7 in one message and confirm Persian,
the mixed spelling and the ~6,500-character pack. With the answers, run
`scan_project.py . --recommend --q1 t2 --q4 few --q5 yes --q6 none --q7 public` and present the pipeline,
then the changes: add a BM25 channel next to the Chroma search and fuse with RRF (Strong against BM25
alone, Tentative against dense alone); `chunk_size=2000`: consider about 1,000 characters (Tentative);
add a cross-encoder reranker after the Chroma query in `app/retrieve.py` (Tentative); normalise Persian
before BM25 and on queries (Tentative); the pack size already fits.

**No codebase.** "I'm building a Q&A bot over ~2,000 internal English policy PDFs. We only have API
access to GPT-class models, no GPU. Answers in under 5 seconds. Most questions need a couple of sections
of one policy." Inferred: q1 t2, q2 en, q4 few, q5 no, q7 private. Ask: budget (q3) and whether one
extra LLM call is acceptable (q6). With q3 moderate, q6 one:

```
python <skill-dir>/scripts/choose.py --q1 t2 --q2 en --q3 moderate --q4 few --q5 no --q6 one --q7 private
```

gives hybrid + 1,000-char chunks + one LLM selection call over the fused candidates + ~6,000 characters
+ no agent, with a trade-off: a larger budget (~15,500 characters) for plain hybrid reached similar
accuracy (68 vs 68–69 of 90) without the extra call. Present it as above, and suggest comparing those
two on their own questions.
