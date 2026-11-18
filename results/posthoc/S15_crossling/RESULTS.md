# S15: cross-lingual query swap and Persian orthographic perturbation (post-hoc, Nov 2026)

These runs are post-hoc and exploratory (COMMON.md); no test question was used for tuning. The frozen code @77924d4 was
imported read-only. Every number below comes from `results_tables.md` / `results.json` (written by `analyze.py`), and
from `rows.jsonl`, `translations.jsonl`, `answers.jsonl`, `judge/`, the ledgers, `embed_runs.jsonl` and `retrieve_X2.log`.
Scores are machine-strict verdicts of the pinned judge, so they cannot be compared with the paper's human-merged numbers.

## 1. What was run

**Code.** `s15lib.py` reuses S10's verified retrieval path (`s10lib`: frozen `HybridIndex.search`, HYBRID `_offer` +
`final_pack`, `build_answer_messages`) and S9's HR path:
- HR: fused top 50 at depth 40, then `rerank_hits` with bge-reranker-v2-m3 (fp16, cuda), keeping 40, then the same packer.
- Recall@40 is S7's span rule (`t5_channels.gold_cov`/`bundle_cls`, copied).
- Pack reach is S7's text rule (`s7lib.gold_reach`, imported).

**Verification for the original queries** (`results_tables.md` §Verification; 96 questions):
- BM25 and DENSE 6,000 packs: sha256 equal to S10's packs 96/96.
- HYBRID fused top 40: equal to the frozen list 96/96. Pack sha equal to the frozen HYBRID pack 96/96.
- HR top 40: equal to the frozen list 96/96. Pack sha equal to the frozen HR_6000 pack 96/96.
- Fresh Ollama vectors of the original questions against S9's saved `q_oll`: cosine min 0.99999988 (`embed_runs.jsonl`).
- Dense recall here is exact, not S7's lower bound.

**Experiment X: cross-lingual query swap.**
- *Translation (`translate.py`).*
  - Model: GLM-5.3-flash, pinned as in reader_pins/MANIFEST (together), with the frozen OpenRouterReader.
  - Prompt: a plain faithful-translation prompt ("translate faithfully and completely … keep every name, number, date,
    year and unit as given … do not answer"; prompt version s15-translate-v1). It was fixed on 5 dev questions
    (`dev_translations.jsonl`) and not changed after that.
  - Coverage: all 96 test questions (90 answerable + 6 unanswerable), forward translation plus a back-translation
    (check only).
  - Number check: the set of numbers is identical in source and translation for 96/96, and every call ended with
    finish reason "stop".
  - Latin script: 20 of the 45 answerable English-pool questions keep a name or acronym in Latin script in their Persian
    translation (e.g. GDPR, Mowi, "Consumer Rights Act 2015"). These are reported separately as "en_noLatin" (25 questions).
  - No fa→en translation contains Persian script.
- *Encoding (`embed.py`).* Each question's translation was encoded as `normalize(t)` with the frozen Ollama bge-m3 path
  (one text per call).
- *Retrieval (`retrieve.py`).* Each question keeps its original pool and scope, and q_lang is set to the translated
  language. Arms: BM25-only, DENSE-only, HYB (RRF) and HR. For each arm: top-40 recall and reach of the 6,000-char pack.
- *End to end.*
  - **HR_6000_TRANS**: the HR 6,000 pack built for the translated query, with the frozen answer prompt. The answer
    language is the language asked, i.e. the translated one.
  - **CLOSEDBOOK_TRANS**: S6's closed-book prompt (`build_answer_messages(query, None)`) with the translated question.
  - Dev smoke: 5 dev questions for each arm (`dev_answers.jsonl`, `dev_judge/`). Then one test run.
- *Judging.* The pinned judge (gpt-oss-120b, deepinfra/bf16, temperature 0, seed 42, no fallbacks, 3 attempts) ran through
  the frozen `judge_multi.judge_records`.
  - Each record carries the question's **original** setting (fa-fa / en-en). The judge therefore sees the original
    question and the same reference packet as for the frozen arms; only the answer differs.
  - The v2 rubric accepts either language.
  - 2 of 192 cells are `needs_adjudication` and are reported as bounds.
- *Comparisons.* Against frozen HR_6000 and S6 CLOSEDBOOK, using the revision_stats helpers as S7 runs them: sign range
  over unresolved cells, and bootstrap over questions and over the 20 components with 10,000 resamples, seed 0.

**Experiment P: orthographic perturbation** (45 answerable Persian test questions; `variants.py`).
- Variants:
  - Arabic ي/ك for Persian ی/ک.
  - ZWNJ removed; ZWNJ replaced by a space.
  - All digits set to Persian, to Arabic-Indic or to Latin.
  - "combined" = Arabic ي/ك + ZWNJ removed + Arabic-Indic digits.
- Arms:
  - BM25: frozen tokenizer.
  - BM25RAW: plain Unicode tokenizer on both sides (S7 raw_both).
  - BM25RAWQ: plain query tokenizer on the frozen index (S7 raw_query).
  - DENSE: Ollama on `normalize(variant)`, the repo path.
  - DENSERAW: the variant sent to Ollama as written.
  - HYB: the repo hybrid.
  - HYBRAW: BM25RAW + DENSERAW with the same RRF.
- Measures: recall@40 and 6,000-pack reach. No reader calls.

## 2. Results

### X. Retrieval, original vs translated query (answerable; complete gold bundle; `results_tables.md`)

| pool lang (n) | arm | recall@40 C orig → trans | gold items orig → trans | 6,000-pack reach C orig → trans |
|---|---|---|---|---|
| fa (45) | BM25 | 37 → **2** | 85 → 13 /106 | 25 → 0 |
| fa | DENSE | 38 → **35** | 98 → 87 | 26 → 21 |
| fa | HYB (RRF) | 39 → **33** | 98 → 85 | 33 → **19** |
| fa | HR | 39 → 35 | 99 → 87 | 34 → 29 |
| en (45) | BM25 | 29 → **8** | 88 → 27 /124 | 20 → 4 |
| en | DENSE | 33 → 28 | 104 → 93 | 21 → 16 |
| en | HYB | 33 → 27 | 98 → 92 | 24 → 16 |
| en | HR | 36 → 28 | 102 → 94 | 29 → 20 |
| en_noLatin (25) | BM25 / DENSE / HYB / HR | 15→3 / 19→16 / 19→17 / 21→17 | | 14→0 / 14→11 / 14→10 / 18→15 |

- **BM25 collapses on cross-lingual queries.**
  - The BM25 list is empty for 11 Persian-pool and 14 English-pool translated queries, and shorter than 40 for 23 and 17.
  - The 8 English-pool bundles it still finds come mostly from Latin-script names left in the Persian translations: 3 of
    25 in en_noLatin.
- **Dense holds up.** It loses 3 (fa) and 5 (en) complete bundles at recall@40.
- **RRF hybrid falls below dense-only on cross-lingual queries.**
  - Recall@40: fa 33 vs 35 (HYB-only 0, DENSE-only 2); en 27 vs 28 (2 / 3).
  - Pack reach in the Persian pool: HYB 19 vs DENSE 21 (1 / 3).
  - Reach loss from the original query: HYB 33 → 19 (−14) against DENSE 26 → 21 (−5).
  - With original queries, HYB was above DENSE in reach (fa 7 / 0).
  - The cross-encoder rescues most of it: HR reaches 29 (fa) and 20 (en), against 21 and 16 for dense alone.

### X. End to end (machine strict, 90 answerable; a..b = bounds over unresolved cells)

| arm | all /90 | fa-pool /45 | en-pool /45 | RPT-fa | RPT-en | BOOK-fa | BOOK-en | LAW-fa | LAW-en | abstention /6 |
|---|---|---|---|---|---|---|---|---|---|---|
| HR_6000 (frozen) | 69..71 | 36 | 33..35 | 15 | 11..12 | 9 | 11 | 12 | 11..12 | 6 |
| **HR_6000_TRANS** | **53..54** | 28 | 25..26 | 10 | 8..9 | 6 | 10 | 12 | 7 | 6 |
| CLOSEDBOOK (S6) | 26 | 6 | 20 | 0 | 0 | 3 | 9 | 3 | 11 | 3 |
| **CLOSEDBOOK_TRANS** | **22..23** | 4..5 | 18 | 0 | 1 | 2 | 7 | 2..3 | 10 | 4 |

By tier (fa pool / en pool): HR_6000_TRANS T1 14/15, T2 7/7, T3 7/3..4, against HR_6000 T1 13/15, T2 14/9..10,
T3 9/9..10.

| contrast | group | net | better/worse | sign p | 95% CI pp, questions | 95% CI pp, components |
|---|---|---|---|---|---|---|
| HR_6000_TRANS vs HR_6000 | all | −18..−15 | 4/20 | 0.001–0.004 | [−30.0, −6.7] | [−30.6, −6.5] |
| | fa pool | −8 | 2/10 | 0.039 | [−31.1, −4.4] | [−35.1, 0.0] |
| | en pool | −10..−7 | 2/10 | 0.013–0.092 | [−37.8, 0.0] | [−33.3, −4.2] |
| CLOSEDBOOK_TRANS vs CLOSEDBOOK | all | −4..−3 | 4/8 | 0.39–0.58 | [−12.2, 4.4] | [−8.6, 2.8] |
| | fa pool | −2..−1 | 1/3 | 0.63–1.0 | | |
| | en pool | −2 | 3/5 | 0.73 | [−17.8, 6.7] | [−11.1, 6.1] |
| HR_6000_TRANS vs CLOSEDBOOK_TRANS | en pool | +7..+8 | 13/6 | 0.12–0.17 | | |

- **The closed-book gap follows the source, not the query language.**
  - English-source questions asked in Persian: 18/45 answered closed-book, against 20/45 asked in English.
  - Persian-source questions asked in English: 4..5/45, against 6/45.
  - Cell by cell, en pool: 15 correct both ways, 5 lost, 3 gained. Fa pool: 3 both, 3 lost, 1 gained (1 unresolved).
- **Abstention.** The closed-book model said NOT_FOUND on 24 Persian-pool and 10 English-pool answerable questions.
- **HR_6000_TRANS, pack reach × verdict.** Complete-bundle packs were answered correctly 46 of 49 times; partial packs
  4/29; none 3/12.
- **Unanswerable questions.** HR_6000_TRANS abstained correctly on 6/6, CLOSEDBOOK_TRANS on 4/6.

### P. Orthographic perturbation (45 answerable Persian questions; complete bundles /45 [Δ vs original])

| variant (questions changed) | BM25 (repo) | BM25RAW | BM25RAWQ | DENSE (repo) | DENSERAW | HYB (repo) | HYBRAW |
|---|---|---|---|---|---|---|---|
| original — recall@40 / reach | 37 / 25 | 35 / 25 | 35 / 25 | 38 / 26 | 38 / 26 | 39 / 33 | 38 / 32 |
| Arabic ي/ك (45) | 37 / 25 | **30 [−5] / 16 [−9]** | **28 [−7] / 18 [−7]** | 38 / 26 | 38 / 27 | 39 / 33 | 40 [+2] / 26 [−6] |
| ZWNJ removed (37) | 36 [−1] / 26 | 34 [−1] / 21 [−4] | 33 [−2] / 22 [−3] | 38 / 26 | 38 / 26 | 39 / **30 [−3]** | 39 / 31 |
| ZWNJ → space (37) | 37 / 26 | 32 [−3] / 22 [−3] | 37 [+2] / 24 | 38 / 26 | 38 / 26 | 39 / 32 [−1] | 37 [−1] / 30 |
| Persian digits (0) | unchanged | | | | | | |
| Arabic-Indic digits (27) | 37 / 25 | 35 / 25 | 35 / 25 | 38 / 26 | 38 / 26 | 39 / 33 | 38 / 32 |
| Latin digits (27) | 37 / 25 | 36 [+1] / 28 [+3] | 36 [+1] / 24 | 38 / 26 | 38 / 26 | 39 / 33 | 39 / 32 |
| combined (45) | 36 [−1] / 26 | **29 [−6] / 15 [−10]** | **27 [−8] / 17 [−8]** | 38 / 26 | 37 [−1] / 26 | 39 / 30 [−3] | 39 [+1] / 25 [−7] |

- **The repo path is robust.** BM25 with normalisation, dense and the repo hybrid lose at most 1 complete bundle at
  recall@40 under every variant.
  - `normalize()` maps ي/ك and every digit form back, so their lists are identical to the original's (45/45).
  - The hybrid's pack reach drops by 3 when ZWNJs are removed: `normalize()` keeps ZWNJ, and removing it merges words.
- **Without normalisation BM25 loses 5–8 bundles** under Arabic ي/ك and under "combined" (BM25RAW −5/−6, BM25RAWQ −7/−8).
  Dense on raw text loses at most 1.
- **Un-normalised hybrid.** Its recall@40 stays at 37–40, because dense compensates. Its pack reach still drops by 6–7
  (32 → 26 / 25): the noisy BM25 votes push gold chunks out of the 6,000-char pack.
- **Original Persian digits.** No test question had non-Persian digits to start with. Only Latin digits help raw BM25 (+1).

## 3. Cost and GPU time

**API spend.** $0.078 of the $0.35 cap (`ledger.jsonl` + `ledger_judge_*.jsonl`). All reader calls were served by
Together, and all judge calls by DeepInfra.

| part | calls | USD |
|---|---|---|
| translation, forward (101 = 96 test + 5 dev) | 101 | 0.0051 |
| back-translation | 101 | 0.0049 |
| HR_6000_TRANS answers (96 + 5 dev) | 101 | 0.0383 |
| CLOSEDBOOK_TRANS answers (96 + 5 dev) | 101 | 0.0096 |
| judge (dev 7 + test 143 model calls; the rest by rule) | 150 | 0.0202 |

- **Tokens per question.** One LLM call per question in each arm. HR_6000_TRANS: 2,115 prompt tokens per question (fa
  pool 2,650, en pool 1,580) and 130 completion tokens; median latency 2.3 s. CLOSEDBOOK_TRANS: 207 prompt and 136
  completion tokens. Translation costs one extra call per question.
- **Key floor.** `limit_remaining` was checked before every batch of at most 50 calls (`floor_checks.jsonl`) and stayed
  between $3.34 and $4.02.

**GPU** (in parallel with other streams, no lock; VRAM at most 3.75 GB used of 8 GB before loading).
- Ollama encoding of 453 query texts: 198 s, including model reload.
- Reranker over 192 queries × 50 pairs: 72 s of rerank wall time, plus about 25 s to load the model.
- The whole retrieval process for X and P took about 5 min. The dev smoke took about 1 min.

## 4. Failed, skipped, caveats

- **Skipped: HYBRID_6000 end to end for the worst orthographic variant.** The trigger was ambiguous. The repo HYBRID moved
  by 0 bundles at recall@40 (−3 at pack reach). Only un-normalised BM25 moved by 5 or more. The coordinator asked to skip
  steps that were not done.
- **Skipped from FRAMING item 3.** Diacritics added or stripped, and the 30 Persian dev questions for recall. Neither was
  in the brief.
- **Unresolved cells.** 2 of 192 judge cells are needs_adjudication (HR_6000_TRANS 1, CLOSEDBOOK_TRANS 1). They are
  counted as bounds and were not retried.
- **Latin script in translations.** 20/45 Persian translations of English-pool questions keep Latin-script names or
  acronyms. This is a natural output of the "keep names as given" prompt, and realistic, but it inflates BM25 and HYB on
  the English pool. Compare the en_noLatin rows.
- **Answer and judge language.** The answer language is the asked (translated) language. The judge sees the original
  question in its own language (see §1), so the judge prompt differs from the frozen one only in the candidate answer's
  language.
- **Repository hygiene, please clean up.**
  - My first `retrieve.py` run had no HF_HOME set, so the reranker started downloading into the frozen worktree. It
    created `D:\KareOne\RAG-Paper-upgrades\work\models\hf\` (22 MB, partial `bge-reranker-v2-m3`).
  - That folder did not exist before 20:06 today, and it is git-ignored (`/work/` in .gitignore). HEAD is still 77924d4
    and no tracked file changed.
  - My attempt to delete it was refused by the permission system. **Please delete
    `D:\KareOne\RAG-Paper-upgrades\work\`.** Later runs set HF_HOME to `D:\KareOne\RAG Paper\work\models\hf` and
    HF_HUB_OFFLINE=1, as S9 did.
- **Name clash in `analyze.py`.** `import judge` there resolves to S10's judge.py, because S10 is first on sys.path. The
  rids are therefore read from the verdict rows, not recomputed. All 190 resolved cells matched.
- **Scope.** One run, one reader, machine verdicts only, post-hoc. The p values are unadjusted.

## 5. Facts for the paper

1. **BM25 collapses on cross-lingual queries; dense holds.** When a question is asked in the other language against its
   own pool, BM25 finds a complete gold bundle in the top 40 for 2/45 (Persian pool) and 8/45 (English pool), down from
   37 and 29. bge-m3 dense keeps 35/45 and 28/45, down from 38 and 33 (`results_tables.md`, experiment X retrieval).
2. **Default RRF hybrid falls below dense-only on cross-lingual queries.** Persian pool: recall@40 33 vs 35, and
   6,000-char pack reach 19 vs 21. Its pack reach falls by 14 bundles against 5 for dense alone. Adding the cross-encoder
   (HR) restores most of it: reach 29 (`results_tables.md`, fusion vs dense table).
3. **End-to-end cost of the swap.** HR_6000 with translated questions scores 53..54/90, against 69..71 with the original
   questions: −18..−15, 4/20, sign p 0.001–0.004, component CI [−30.6, −6.5] pp. The drop is similar in both pools: fa
   −8, en −10..−7 (`results.json` X_compare).
4. **The closed-book memory gap follows the source, not the query language.** English-source questions asked in Persian
   are answered closed-book 18/45 times, against 20/45 in English. Persian-source questions asked in English: 4..5/45,
   against 6/45. Overall −4..−3, p 0.39–0.58 (`results_tables.md`, closed-book rows).
5. **Orthographic noise only matters without normalisation.** With the repo's `normalize()`, BM25, dense and the hybrid
   lose at most 1 complete bundle at recall@40 under Arabic ي/ك, ZWNJ or digit changes. Without it, BM25 loses 5–8 of 45
   (Arabic ي/ك, combined), and the un-normalised hybrid's pack reach drops by 6–7 even though dense keeps its recall
   (`results_tables.md`, experiment P).
