# Idea test: keyword-count ranking (KW) and document-first retrieval (DOC), 2026-09-26

**Where things are**
- Code: `RAG-Paper-upgrades` branch `upgrades`, commit d866a97.
  - `ragbench/approaches/kwdoc.py` holds the method.
  - HYBRID gets two flags: `kw` and `doc=<k>`.
- Harness: this folder.
  - `idea_a.py` does the keywords, the retrieval replay and the part-A tables.
  - `idea_b.py` does the answers; `b_report.py` builds the part-B tables.
- Step log: `PROGRESS.md`. The question list and every rule were declared there before the first run.
- Spend: **$0.040** in total. The budget file moved from 5.4019 to 5.4421.
  - 24 keyword calls: $0.0011.
  - 72 answer cells.
  - 72 judge model calls.

**The two methods**

**KW (keyword-count ranking).**
- One `openai/gpt-oss-20b` call per question writes ~10 keywords. It is pinned to amazon-bedrock, with reasoning low,
  1 attempt and a 6 s deadline. The prompt is domain-neutral and asks for "words a passage that ANSWERS the question
  would contain". The model returned 12.2 keywords per question (capped at 16).
- Each chunk of the pool is scored by the number of **distinct** keywords it contains. A keyword counts on either of
  two matches:
  - **Exact:** the folded token sequence, where fold handles Persian/Arabic letters, digits and harakat, and a
    half-space is a word break. Solid spellings also match (`X‌Y` = `XY`).
  - **Similar (a):** every content token has a chunk token with the same light fa/en stem, or a shared prefix of at
    least max(5, shorter length − 2) characters. Numbers must match exactly.
  - **(b) embedding-similar words were skipped.** They would need embeddings of the whole pool vocabulary.
- Ties are broken by BM25.
- The index is built once per corpus set and kept in memory: token postings, solid-bigram postings, and the vocabulary
  by prefix and by stem. It is not scanned per keyword. Build time is 3.2 s for the 6,921 chunks of the 97-document
  RPT-fa set; ranking takes about 13 ms per question.
- **HYBRID+KW** = RRF (k=60) of BM25 top 40, dense top 40 and KW top 40.

**DOC (document-first).**
- Pool documents are ranked by idf-weighted keyword coverage: the sum of ln(1 + N/df) over the keywords found in the
  document.
- HYBRID+KW then runs inside the top-k documents.
- Fallback (declared before the run): the whole pool is used when no chunk in the chosen documents holds at least
  min(⌈n/2⌉, 2) keywords.

**Questions**
- The 12 pilot questions plus 12 new ones: the -001 question of the 6 cells the pilot did not cover and of the 6 RPT
  cells.
- 12 are Persian and 12 English, with 8 per tier.
- By domain: RPT 12 (6 of them in the 97-document RPT-fa pool), BOOK 6, LAW 6.

**Caveat.** There are 24 questions. A difference of 1–2 questions, or about 0.05 in mean retention, is noise.

## A. Retrieval (no reader)

Every arm is packed by the same packer: HYBRID-intact, with intact packages in the arm's order, exact dedupe, exact
rendered cost and 6,000 characters. Retention is decisive-span retention with source identity.

| arm | span retention | 100% retained /24 | T3: both docs in pack /8 | top-24 retention | top-24 100% /24 |
|---|---:|---:|---:|---:|---:|
| HYBRID | 0.748 | 14 | 3 | 0.898 | 20 |
| HYBRID+rerank | **0.799** | **16** | **4** | 0.897 | 18 |
| KW | 0.601 | 11 | 2 | 0.765 | 14 |
| HYBRID+KW | 0.663 | 12 | 2 | **0.909** | 19 |
| DOC-1 | 0.490 | 8 | 0 | 0.621 | 10 |
| DOC-2 | 0.622 | 12 | 2 | 0.784 | 16 |
| DOC-3 | 0.642 | 12 | 2 | 0.816 | 17 |
| DOC-5 | 0.642 | 12 | 2 | 0.816 | 17 |

Retention by group, as mean retention (with the number of questions at 100%):

| group (n) | HYBRID | +rerank | KW | HYBRID+KW | DOC-3 |
|---|---:|---:|---:|---:|---:|
| fa (12) | 0.74 (7) | 0.82 (8) | 0.61 (6) | 0.65 (6) | 0.61 (6) |
| en (12) | 0.76 (7) | 0.78 (8) | 0.59 (5) | 0.67 (6) | 0.67 (6) |
| T1 (8) | 1.00 (8) | 1.00 (8) | 0.88 (7) | 1.00 (8) | 1.00 (8) |
| T2 (8) | 0.65 (3) | 0.68 (4) | 0.52 (2) | 0.52 (2) | 0.46 (2) |
| T3 (8) | 0.59 (3) | 0.72 (4) | 0.41 (2) | 0.47 (2) | 0.47 (2) |
| RPT-fa, 97 docs (6) | 0.67 (3) | 0.83 (4) | 0.58 (3) | 0.58 (3) | 0.50 (3) |

The full breakdown (per domain, pilot vs new, per question) is in `tables_a.md`.

The DOC step, where the gold documents are the documents of the decisive spans:

| k | gold-doc recall | all gold docs chosen /24 | T3: both chosen /8 | fallback fired |
|---|---:|---:|---:|---:|
| 1 | 0.646 | 12 | 0 | 0 /24 |
| 2 | 0.812 | 18 | 4 | 0 /24 |
| 3 | 0.854 | 20 | 6 | 0 /24 |
| 5 | 0.854 | 20 | 6 | 0 /24 |

**DOC-best is DOC-3.** The declared rule is highest retention with ties going to the smaller k; DOC-3 and DOC-5 tie
at 0.642.

**Cost of each step:**
- Keyword call: p50 839 ms, p95 1,551 ms, with 262 prompt and 185 completion tokens.
- KW scoring and ranking: about 13 ms.
- DOC step: under 1 ms.

The offline reranker ran on the CPU (about 62 s per question). That is not a latency figure; part B has the real one.

## B. Answers

- Reader: `z-ai/glm-5.3-flash`, pinned to together, with the pilot's settings.
- Judge: v2 (`judge_multi`, `openai/gpt-oss-120b`), strict.
- Packing: HYBRID's own packer in every arm, so the answers compare like with like.
- Reuse: the HYBRID and HYBRID+rerank answers for the 12 pilot questions come from `lex5_pilot` (same code path,
  reader, pin and settings). Everything else is new.
- Latency: the whole timed query. For the KW arms, the part-A keyword call is added; both arms reused those same
  keywords.

| arm | correct /24 | fa | en | T1 | T2 | T3 | shown retention | p50 s | p95 s | keyword / retrieval / answer p50 ms | LLM tokens/q | $/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| HYBRID | 12 | 5 | 7 | 6 | 2 | 4 | 0.728 | 1.62 | 3.69 | – / 114 / 1,583 | 2,169 | 0.00035 |
| HYBRID+rerank | **13** | 6 | 7 | 6 | 1 | 6 | **0.828** | 2.74 | 5.59 | – / 965 (GPU) / 1,741 | 2,215 | 0.00037 |
| HYBRID+KW | 10 | 6 | 4 | 6 | 1 | 3 | 0.726 | 2.40 | 4.37 | 839 / 120 / 1,356 | 2,638 | 0.00037 |
| DOC-3 | 11 | 6 | 5 | 7 | 1 | 3 | 0.684 | 2.46 | 4.52 | 839 / 95 / 1,447 | 2,635 | 0.00031 |

**Ratios against HYBRID.**
- HYBRID+KW: LLM tokens 1.22x, p50 1.48x, p95 1.18x.
- DOC-3: tokens 1.21x, p50 1.52x.
- Against HYBRID+rerank, both KW arms are about 0.9x on latency and 1.19x on tokens.

The per-question grid is in `tables_b.md`.

## Verdicts

**KW: no.**
- On its own, KW loses retention to HYBRID: 0.60 against 0.75.
- Fused into HYBRID it still loses: 0.66, with 12 questions at 100% against 14.
- Answers are no better: 10 correct against HYBRID's 12 and HYBRID+rerank's 13. That gap is inside the noise, but it
  points the wrong way.
- It costs about 450 extra LLM tokens and about 0.8 s per question. That is 1.5x HYBRID's p50, which is at the edge of
  the LEX latency gate.
- The one place it is level is the top-24 candidates: 0.909 against 0.898, with 19 questions at 100% against 20. It
  neither adds nor removes much of what a selector would see.

**DOC: no, at any k.**
- Retention at the best k (DOC-3) is 0.64 against 0.75.
- Answers: 11 against 12 and 13.
- It never beats HYBRID+KW, which is the same search without the document restriction. The restriction only removes
  evidence.
- HYBRID+rerank remains the strongest arm on both retention and answers.

**Where DOC fails.**
- **The 97-document pool.** The gold document ranks 7th on RPT-fa-T2-002 and 10th on RPT-fa-T2-001. On RPT-fa-T3-002
  the three gold documents rank 11th, 16th and 41st.
- **Why the wrong documents win.** Coverage scoring rewards large documents that happen to contain many generic
  keywords (years such as ۱۴۰۰, and words such as "number" or "users"). The same company's report from the wrong year
  (cafebazaar 1398) or a sibling report (ads-report 1400) wins. There is no length normalisation, and year keywords
  have low df but match many documents.
- **Two-document (T3) questions.** Even k=3 captures both gold documents on only 6 of 8. k=1 by construction never
  can, and DOC-1 scores 0/8 on T3.
- **The fallback never fired (0/24 at every k).** "Some chunk holds at least 2 keywords" is almost always true once
  keywords include years or the entity name, so it is not a useful "found nothing" signal.
- **Small pools (BOOK, LAW; 10–13 documents).** DOC-3 changes little, because 3 documents is already about 25% of the
  pool.

**KW failure cases** (keywords against the wording of the decisive span):
1. **RPT-fa-T2-002** (installs in 1399).
   - Keywords: «کاربران کافه‌بازار», «نصب برنامه», «تعداد نصب», «Cafe Bazaar users».
   - The span reads: «کاربران **بازار** در سال 1399، بیش از 670 میلیون بار بازی، نصب و بروزرسانی کرده‌اند…».
   - The report calls itself «بازار», not «کافه‌بازار», and puts «نصب» next to «بازی/بروزرسانی», never in «نصب
     برنامه». The best span chunk holds 5 keywords, but mostly generic ones such as «سال ۱۳۹۹» and «چند بار» (a
     question phrase that similar-matched «بار»). KW retention is 0, against 0.5 for HYBRID.
2. **RPT-en-T2-002** (Glovo rider risk).
   - Keywords: "rider classification policy/decision/adjustment/update" (invented phrasings).
   - The spans say "risks in Spain in relation to the model operated by Glovo until August 2021", "appropriate
     provisions", "contingent liability of € 440 to 770 million" and "employment-based model".
   - Three of the seven spans' chunks contain 0 keywords. Retention is 0.4, against 1.0 for HYBRID.
3. **LAW-fa-T3-001** (minimum age for a newspaper licence).
   - Keywords: «حداقل سنی», «امتیاز روزنامه», «متقاضی امتیاز».
   - The spans are «درخواست کننده اجازه باید دارای شرایط ذیل باشد», «سن او از سی سال کمتر نباشد» and «دارا بودن
     حداقل 25 سال».
   - The law says «درخواست کننده اجازه» (1334) and «سن … کمتر نباشد». Every span chunk has at most 1 keyword, and the 6
     English keywords hit nothing in a Persian pool. KW retention is 0; HYBRID's is 1.0.
4. **LAW-en-T2-001** (UK bribery, foreign official, British citizen).
   - The keywords name the case: "UK citizen", "French ministry", "Government contract".
   - The decisive provisions are phrased generically: "close connection with the United Kingdom", "a British citizen",
     "no act or omission … takes place in the United Kingdom". Six of the nine span chunks contain 0 keywords.
     Retention is 0.33, against 0.78.
5. **BOOK-en-T1-001** (Faraday's candle cup).
   - The decisive chunk does contain 4 keywords (candle, wick, flame, melting sides≈).
   - Dozens of other chunks of the same book contain as many or more (candle 163 chunks, flame 110, combustion 111).
     Count-ranking cannot separate them, and BM25 as a tie-breaker is too weak. KW-alone retention is 0; HYBRID+KW
     recovers it (1.0).

**The common cause.**
- A 20B model, asked for answer-side vocabulary, mostly paraphrases the question. It also invents phrasings the
  documents do not use, and adds cross-language keywords that are dead weight in single-language pools.
- A distinct-count score then favours chunks rich in generic entity, year and topic words, not the specific sentence.

## What to try next (if anything)
1. **Keep KW only as a candidate-recall channel, never as a ranker or a document filter.** It is level with HYBRID on
   top-24 retention, so it could feed a selector.
2. **Weight keywords by idf within the pool, and exclude year and entity keywords from the chunk score.** A pure count
   treats «۱۴۰۰» like the decisive term.
3. **Try doc-level prefiltering only for very large pools (≥ 50 documents), and only as a soft prior.** One option:
   add a document-rank channel to RRF instead of a hard restriction, with length-normalised coverage (BM25-style).
4. **Drop the "other language" instruction for single-language pools.** 0 hits in every such case.
5. **Similar-word (b) with embeddings is not worth building on this evidence.** The misses are paraphrase-level
   («درخواست کننده اجازه» against «متقاضی امتیاز»), and word-level similarity would not bridge them.
6. **HYBRID+rerank remains the arm to beat.** On these 24 questions it leads on retention (0.80), questions at 100%
   (16), T3 (4/8 both docs) and answers (13/24).
