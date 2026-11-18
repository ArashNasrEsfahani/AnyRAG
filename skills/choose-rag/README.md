# choose-rag: an Agent Skill for picking a RAG setup

`choose-rag` helps you choose a retrieval-augmented generation (RAG) setup for your own project. Point it
at your repository: it reads your code and documents locally, infers what it can (document language and
spelling, current chunk size and top-k, vector DB, BM25, embedding model, reranker, GPU, LLM provider,
latency hints, question types) and asks only what it cannot tell. Without a codebase it asks up to eight
short questions instead. It then runs the decision rules of a research study and gives you a recommended
pipeline and, for an existing stack, the changes to make file by file. Each part comes with its
evidence, a confidence label (Strong / Tentative / Measured) and the paper source, plus what to avoid,
the trade-offs, and a recipe to check the result on your own questions.

The evidence comes from *What Reaches the Reader: A Comparison of Retrieval-Augmented Generation Setups
in Persian and English* (Arash Nasr Esfahani, University of Tehran, in preparation). It rests on 90 test
questions, one primary reader and one judge, and most analyses are post hoc. The skill says so every
time, and it tells you when your situation is outside what the study tested.

The same folder works in Claude (Claude Code and the Claude apps) and in OpenAI Codex. Both read
`SKILL.md` with `name` and `description` frontmatter. `agents/openai.yaml` holds optional Codex UI
metadata, which Claude ignores.

## Contents

```
choose-rag/
  SKILL.md                      instructions for the agent
  README.md                     this file
  agents/openai.yaml            optional Codex metadata
  references/
    chooser.json                8 questions, option cards, 50 rules, 33 test profiles
    evidence.json               65 evidence items with confidence labels and sources
    guide.md                    plain-English decision guide and scope limits
    evaluation_recipe.md        paired evaluation with a sign test, on your own data
    source_map.md               how sources are cited (journal manuscript, by title)
  scripts/
    scan_project.py             read-only project scanner: infers the 8 answers, the stack, the changes
    choose.py                   the rules as a deterministic script (Python 3 standard library)
    test_choose.py              reproduces the 33 test profiles and sweeps all 43,200 answer combinations
    test_scan.py                runs the scanner on 4 synthetic projects built in a temporary folder
    sign_test.py                exact sign test for the evaluation recipe
```

## Install

### Claude Code

Personal, for all projects, straight from GitHub:

```bash
git clone --depth 1 https://github.com/ArashNasrEsfahani/AnyRAG.git && mkdir -p ~/.claude/skills && cp -r AnyRAG/skills/choose-rag ~/.claude/skills/
```

Or from this folder:

```bash
# macOS / Linux
mkdir -p ~/.claude/skills && cp -r choose-rag ~/.claude/skills/
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.claude\skills" | Out-Null
Copy-Item -Recurse choose-rag "$env:USERPROFILE\.claude\skills\"
```

For one project only, copy the folder to `<project>/.claude/skills/choose-rag/`. Then start Claude Code
in your project and ask a RAG question ("Which RAG setup should I use for this project?"), or type
`/choose-rag`.

### Claude apps (claude.ai, desktop)

1. Use `choose-rag.zip` (next to this folder), or zip the `choose-rag` folder yourself so that the zip
   contains `choose-rag/SKILL.md`.
2. Turn on code execution, then go to Settings > Capabilities > Skills and upload the zip. (Menu names
   may differ by plan and version.)
3. Ask, for example, "Which RAG setup should I use for ...?"

If code execution is off, the skill still works: the agent applies the rules from `chooser.json` by hand
and says so.

### OpenAI Codex (CLI, IDE extension)

```bash
git clone --depth 1 https://github.com/ArashNasrEsfahani/AnyRAG.git && mkdir -p ~/.codex/skills && cp -r AnyRAG/skills/choose-rag ~/.codex/skills/
# or, from this folder:
mkdir -p ~/.codex/skills && cp -r choose-rag ~/.codex/skills/
```

Codex scans `$CODEX_HOME/skills` (by default `~/.codex/skills`, which is also where its own
skill-installer puts skills) and `~/.agents/skills`. The current Codex documentation lists
`~/.agents/skills/<name>/` for personal skills and `<repo>/.agents/skills/<name>/` for a repository, so
this also works:

```bash
mkdir -p ~/.agents/skills && cp -r choose-rag ~/.agents/skills/
```

Restart Codex. Invoke the skill explicitly with `$choose-rag` (or pick it from `/skills`), or just ask a
RAG question and Codex will pick it from the description.

### Check the install

```bash
python ~/.claude/skills/choose-rag/scripts/test_choose.py     # or the Codex path
python ~/.claude/skills/choose-rag/scripts/test_scan.py
```

They should print `Profiles: 33 of 33 passed` with a sweep of 0 problems, and `Scan checks: 38 of 38 passed`.

## Point it at your repo

Open Claude Code or Codex in your project folder and ask:

> Which RAG setup should I use for this project?

or "Review my RAG setup", "Choose retrieval for my repo", `/choose-rag` (Claude Code), `$choose-rag`
(Codex). The agent runs the scanner on the folder, shows you what it inferred with a confidence for each
answer, asks only the unknown or uncertain ones, and then gives the recommendation with changes such as:

```
[change] app/rag/config.py:2: Your chunk_size=2000: consider about 1,000 characters with 200 overlap.
         Confidence: Tentative (at a 6,000-character budget with this packer). Evidence: g-chunk-2000 [tentative], chunks [measured]
[change] app/retrieve.py:8: Add a cross-encoder reranker (e.g. bge-reranker-v2-m3) after your Chroma retrieval ...
[keep]   app/retrieve.py:8: Your pack of about 6,480 characters fits the recommended ~6,000.
```

The scanner is safe to run on any folder: it only reads files (standard library, no dependencies), never
executes or imports your code, never uses the network and writes nothing. It skips `.git`,
`node_modules`, virtual environments, build and cache folders, binary files and `.env` files, and caps
what it reads (2,500 files, 256 KB per text file, 64 MB in total). Your documents and code stay on your
machine; the agent sees only the scan summary and the lines it chooses to read. You can run it yourself:

```bash
python scripts/scan_project.py path/to/your/repo                 # slot table, corpus, stack, resources
python scripts/scan_project.py path/to/your/repo --json          # machine-readable; "answers" feeds choose.py
python scripts/scan_project.py path/to/your/repo --recommend --q1 t2 --q4 few   # + recommendation and changes
```

It infers: document folders, types, count, size, rough characters and script mix (Persian/Arabic, Latin,
other); the language of test questions, UI strings and prompts; vector DBs (FAISS, Chroma, Qdrant,
Weaviate, pgvector, Milvus, LanceDB, Elasticsearch/OpenSearch), BM25 (rank_bm25, bm25s, Elasticsearch),
embedding models (bge-m3, e5, OpenAI, sentence-transformers), rerankers (bge-reranker, Cohere, cross-encoders),
frameworks (LangChain, LlamaIndex, Haystack, DSPy), chunk_size / overlap, top-k and context budgets; GPU
hints, LLM providers (OpenAI, Anthropic, OpenRouter, Ollama, vLLM, ...) and latency hints; FAQ versus
cross-document question wording. Code-side inferences are heuristics: the confidence column says how far
to trust each one, and the agent confirms them with you.

## Using the script directly

```bash
python scripts/choose.py --questions                     # the interview
python scripts/choose.py --q1 t3 --q2 fa --q5 yes        # unanswered = "not sure"
python scripts/choose.py --json '{"q1":"t2","q2":"en"}' --format json
python scripts/choose.py --q1 t2 --q2 fa --lang fa       # Persian card text
python scripts/sign_test.py results.csv                  # question_id,a,b[,exclude]
```

## Example conversation

> **User:** We're building a Q&A assistant for our law firm. It answers from EU regulations plus our own
> contract templates, all in English. We only use a hosted LLM API, no GPU. Which RAG setup should we use?
>
> **Assistant:** A few things I'm assuming from your message, correct me if they're wrong: English
> documents and questions; no GPU, so no local cross-encoder reranker; and the model may already know
> parts of the EU regulations from training, since they are public. Three quick questions:
> 1. Do lawyers mostly ask for one provision, combine several parts of one document, or compare across
>    documents (for example, a template against a regulation)?
> 2. How long can an answer take: under 2 seconds, a few seconds, or ~10 seconds?
> 3. Is one extra LLM call per question acceptable, or several?
> (And if you know how much evidence text you want to send per question, tell me; otherwise I'll use
> the default.)
>
> **User:** A mix, honestly. A few seconds is fine. One extra call is OK. No idea on the budget.
>
> **Assistant:** *(runs `choose.py --q1 mix --q2 en --q4 few --q5 no --q6 one --q7 public`)*
>
> **Recommended pipeline**
> `question -> BM25 + dense (bge-m3) -> RRF fusion -> one LLM selection call -> ~6,000-character pack -> reader`
>
> | Part | Setting | Confidence | Source (journal ms., by title) |
> |---|---|---|---|
> | Retrieval | BM25 + dense, fused with reciprocal rank fusion. Hybrid 62–63 vs dense 53 vs BM25 48 of 90 | Strong against BM25 alone, Tentative against dense alone | Guide table; components table |
> | Chunks | ~1,000 characters, 200 overlap (2,000-char chunks: 51 vs 69–71) | Strong against 2,000 at this budget; Tentative against 500 | Guide table; components table |
> | Selection | One LLM call that picks passages from the fused candidates (no GPU needed); removing it cost ~15 answers at 3,000 chars and 5–10 at 6,000 | Tentative at 6,000 (your budget); Strong only at 3,000 (one run) | Guide table; budget figure; data release (posthoc/S8_factorial) |
> | Budget | ~6,000 characters | Tentative | Guide table; budget figure |
> | Agent | None | Tentative | Guide table |
> | Normalisation | Standard tokenisation (the paper's normalisation findings are Persian only) | Measured | |
>
> **Also do:** build your evaluation without questions the model answers from memory, found with a
> closed-book run (Strong: the reader answered 26 of 90 test questions with no evidence, mostly
> well-known English texts and EU law). Count how often the pack holds the complete evidence, not just
> accuracy. Either bge-m3 or multilingual-e5-large is fine.
>
> **Avoid:** BM25 alone (Strong); 2,000-character chunks (Strong); HyDE or multi-query rewriting
> (Tentative); stuffing whole documents into a long prompt (24 of 90, Measured); routing questions by
> guessed difficulty, which saved ~4% of tokens (Measured).
>
> **Trade-off:** instead of the selection call you could send plain hybrid ~15,500 characters: similar
> accuracy (68 vs 68–69 of 90), no extra call, but 2.5 times the text (Tentative).
>
> **Not covered by the study:** legal cross-references, citations and amendment chains were not studied
> as such; the statutes in the study were searched as plain text. *General practice, not from this
> study:* keep article numbers and headings in each chunk so answers can cite them.
>
> **Caveat:** this rests on 90 test questions, one primary reader and one judge; most analyses are post
> hoc. Check it on your own data: take 50+ real questions, drop the ones the model answers closed-book,
> run the selection-call setup and plain hybrid at 15,500 characters on the same questions, score blind,
> and run `scripts/sign_test.py`. If p ≥ .05, choose the cheaper one.

## Licence and citation

Data and text follow the AnyRAG release licences (code MIT, data CC BY 4.0). Cite the paper:
Arash Nasr Esfahani, *What Reaches the Reader: A Comparison of Retrieval-Augmented Generation Setups in
Persian and English*, University of Tehran, in preparation.
