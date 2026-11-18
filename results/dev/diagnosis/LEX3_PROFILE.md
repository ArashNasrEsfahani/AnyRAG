# LEX3 profile: where the time goes, and what makes HYBRID+rerank accurate

Dev run `work/datasets_v2/devrun` (60 questions, reader gemma4:e4b-it-qat, judge gpt-oss-120b). Read-only analysis from
saved traces (`runs/dev/*/{LEX3,HYBRID,HYBRID+rerank,LEX2}/runs_*.jsonl`), verdicts (`RESULTS_DEV.json` cells) and gold
units (`questions/*_dev.jsonl` evidence unit_ids). No model calls. Three light offline timings (CPU only, read-only KB
`work/kb/kb.sqlite`) split the lexical stage into load / scan / windows. Scripts: session scratchpad (`prof1.py`,
`bench_lex.py`, `acc*.py`).

Code: `RAG-Paper-upgrades/ragbench/approaches/lex3/{approach.py,core.py}`, `approaches/lex/search.py`,
`retrieval/hybrid.py`, `evidence.py`.

Caveat up front: 3 questions per cell, one replicate, a 4B reader and a single judge pass. Several flips below happen
with the SAME gold coverage and near-identical answers (reader or judge noise). Treat per-question mechanisms as
hypotheses. The latency findings are solid: they come from per-stage timers in every trace.

---

## 1. LEX3 latency, stage by stage

The per-stage timers are `trace.timing`. Model time is the sum of `trace.calls[].wall_ms`. Code time is total minus model time.

### All 60 questions (seconds)

| stage | what it does | p50 | p95 | mean | share of total |
|---|---|---:|---:|---:|---:|
| question | BM25 + dense over the pool (query embed) + question-only routing | 1.4 | 2.5 | 1.43 | 3.6% |
| **interpret (call 1)** | LLM: entities, periods, facets | 7.0 | 9.4 | 6.89 | 17.2% |
| route | card BM25 + chunk votes, shortlist 8 | 0.0 | 0.2 | 0.06 | 0.1% |
| vocab | phrase mining from the top 24 hits (+ one fact-label SQL per hit) | 4.4 | 12.7 | 4.95 | 12.3% |
| **pick (call 2)** | LLM: phrase IDs per facet | 5.4 | 10.1 | 5.83 | 14.5% |
| **lexical** | Reports load + regex scan of EVERY unit of the 8 shortlisted docs + windows | 10.3 | 34.7 | 12.28 | 30.6% |
| fuse | RRF over regions | 0.0 | 0.0 | 0.01 | 0.0% |
| blocks | build_blocks | 0.1 | 0.4 | 0.16 | 0.4% |
| followup | per missing requirement: hybrid search + another full lexical pass | 0.0 | 16.4 | 3.46 | 8.6% |
| answer (call 3) | LLM answer | 4.8 | 11.2 | 5.10 | 12.7% |
| **total** | | **36.6** | **66.7** | 40.2 | |

- **Model time vs code time:** model p50 18.1 s (mean 17.8), code p50 20.5 s (mean 22.4, p95 43.4). **More than half of
  LEX3's latency is Python code, not the model.** Code that could be cut or cached: lexical 12.3 s, vocab 5.0 s and
  follow-up 3.5 s, about 21 s of the mean.
- **Calls and tokens** (60 each, no cache hits):

| call | wall p50 / p95 s | prompt tokens p50 (mean) | completion tokens p50 (mean) |
|---|---|---:|---:|
| 1 interpret (`plan`) | 6.9 / 9.3 | 363 (364) | 142 (142) |
| 2 pick | 5.3 / 10.1 | 1,164 (1,200) | 79 (81) |
| 3 answer | 4.8 / 11.2 | 1,491 (1,564) | 50 (74) |

  Call 1 is short in and long out: 142 generated tokens of JSON is what makes it take 7 s. Call 2 spends 1.2k prompt tokens
  on the phrase list.
- **Degraded modes:** 59 questions ran with no degraded mode and 1 ran with `pick_failed`. Routing ran on all 60 questions, because every pool has more than 8 docs. Follow-up ran on
  20 of the 60 questions (35 retrievals), and only 7 of those 20 ended correct.

### Per pool (p50 / p95 seconds)

| pool (docs) | question | interpret | vocab | pick | lexical | followup | answer | total | model p50 | code p50 |
|---|---|---|---|---|---|---|---|---|---:|---:|
| BOOK-en (11) | 1.3/1.6 | 6.2/8.2 | 0.9/1.4 | 4.5/7.3 | 11.0/19.2 | 0.0/7.9 | 2.8/7.4 | 29.0/44.1 | 13.3 | 14.6 |
| BOOK-fa (9) | 1.5/1.8 | 7.9/9.2 | 6.8/11.4 | 6.7/8.1 | 14.4/41.5 | **15.1**/18.8 | 5.3/13.2 | **56.9**/81.7 | 20.1 | **38.3** |
| LAW-en (10) | 1.4/1.9 | 5.8/8.6 | 2.9/5.4 | 4.0/6.8 | 7.3/13.8 | 0.0/9.7 | 4.5/10.7 | 32.7/50.2 | 14.5 | 16.8 |
| LAW-fa (13) | 1.4/1.6 | 6.6/9.1 | 1.9/1.9 | 4.6/7.4 | 2.3/3.6 | 0.0/4.3 | 6.2/10.4 | 25.3/31.7 | 17.2 | 6.2 |
| RPT-en (15) | 1.5/3.5 | 7.1/9.5 | 5.1/7.5 | 6.2/15.7 | **25.3**/38.0 | 0.0/6.2 | 6.4/7.0 | **56.9**/70.3 | 21.3 | 33.1 |
| RPT-fa (97) | 2.2/2.6 | 8.6/11.4 | **11.3**/17.5 | 5.5/10.7 | 6.7/10.1 | 0.0/0.0 | 3.7/7.7 | 38.2/49.1 | 18.4 | 20.3 |

The 97-document RPT-fa pool is **not** the slow one. The lexical scan grows with the number of **units inside the 8
shortlisted docs**, not with pool size. The RPT-en shortlist holds 51,777 units (8.1 M characters of annual-report text),
the BOOK-fa shortlist about 14k, and the RPT-fa shortlist about 5k. RPT-fa's cost is in vocab instead: its KB carries dense fact labels,
so the phrase miner runs 24 fact-label SQL queries, about 60 ms each when warm and more when cold.

### Why `lexical` costs 10–35 s (offline re-timing of the same terms over the same shortlist)

| question | shortlist units | terms (literal) | load+fold | regex scan | scan, literal terms only | windows | trace lexical |
|---|---:|---:|---:|---:|---:|---:|---:|
| RPT-en-T1-001 | 51,777 | 14 (7) | 2.4 s | 9.6 s | 3.8 s | 0.06 s | 18.6 s |
| BOOK-fa-T1-001 | 13,674 | 26 (13) | 2.1 s | 9.8 s | 4.1 s | 0.18 s | 58.3 s |
| BOOK-fa-T1-003 | 14,229 | 20 (10) | 2.5 s | 8.3 s | 3.6 s | 0.93 s | 14.2 s (+13.2 s follow-up) |
| RPT-fa-T1-002 | 4,926 | 24 (12) | 1.1 s | 0.9 s | 0.3 s | 0.07 s | 3.8 s |

1. **Every unit × every term.** `search.search()` runs every term's regex over every unit of the shortlist. Terms that
   miss as a phrase then retry each content word (AND fallback).
2. **Morphological twins double the scan.** `MorphTerm.match()` re-runs the literal term first, and there is one twin per
   literal term. The scan takes 9.6 s with the twins and 3.8 s with literal terms only.
3. **Reports are rebuilt per question and per follow-up.** `S.Reports(self.con)` is created fresh, so units are re-read
   from SQLite and re-folded each time. `units` has **no index on doc_id**, so each doc load is a full-table scan of
   251k rows in a 1 GB file. The folded text is already stored in `units.text_fold`, and the code folds again anyway.
   This costs 2.1–2.6 s warm per pass, much more when the page cache is cold, which is the likely cause of the 58 s outlier.
4. **Follow-up repeats all of it.** Each follow-up builds a new `Reports` and scans again. BOOK-fa follow-up has a p50 of 15 s.

---

## 2. HYBRID+rerank vs HYBRID vs LEX3: evidence shape

Answerable questions (54). "Gold" = the question's evidence unit_ids.

| arm | items/q | median item chars | items < 300 chars | evidence chars (median) | gold units covered | gold in evidence | gold at item 1 | docs/q | items from non-gold docs /q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 6.6 | 845 | 2% | 5,999 | 0.752 | 47/54 | 38 | 1.65 | 0.76 |
| HYBRID+rerank | 6.6 | 855 | 2% | 6,000 | **0.798** | **50/54** | 37 | 1.72 | 0.70 |
| LEX3 | 6.1 | 810 | **14%** | **5,357** | 0.745 | 48/54 | 37 | 1.54 | 0.57 |
| LEX2 | 9.0 | 441 | 20% | 5,926 | 0.484 | 37/54 | 15 | 2.28 | - |

- **What the reranker does (bge-reranker-v2-m3, top 50, keeps 40, packs top-8 into 6,000 chars):** it moved **23 gold
  chunks from RRF rank ≥ 8 into the packed top 8, across 16 questions**. It pushed 16 gold chunks the other way, but the
  net effect on first-gold rank is 13 better, 7 worse and 34 unchanged. Coverage rises from 0.752 to 0.798, and three
  more questions get some gold into evidence.
- **Reranked evidence shape:** it is the same as HYBRID's. There are about 6.6 whole chunks of about 850 chars each, the
  budget is filled and the last item is cut. There are almost no fragments.
- **LEX3 evidence differs in four ways.**
  - It holds fewer and smaller items: 14% are under 300 chars, some are 40–98-char headings or table slivers, and about
    1,000 chars of the budget go unused.
  - The block cap of 6 fills with pieces of one region. Several blocks share the same RRF and ranks, and 3,670 candidates
    were dropped at the cap across the 54 questions.
  - It shows fewer distractor docs: 0.57 items per question from non-gold docs, against 0.70 for HYBRID+rerank.
  - Its gold coverage roughly equals HYBRID's.
- **The LEX lexical channel adds no recall.** Of 86 gold-bearing LEX3 evidence blocks, **every one also had a BM25 or
  dense rank from the question channels**. 76 had all three channels, 7 had bm25+dense only and 3 had lex plus one
  question channel. None came from lex alone, and none came from follow-up alone. The lexical scan works as a
  **rescoring vote** on candidates the hybrid already had. It never finds anything new.

### Where the reranker changed the outcome (HYBRID → HYBRID+rerank)

| question | change | mechanism |
|---|---|---|
| BOOK-en-T1-001 | 0 → 1 | Promoted gold chunks from RRF ranks 22, 9 and 13, giving 5 gold items instead of 2, so the reader got the full explanation |
| BOOK-en-T1-003 | 0 → 1 | Gold (b220) moved to #1 and a second gold chunk came up from RRF rank 17 |
| BOOK-en-T2-002 | 0 → 1 | Gold coverage 0.5 → 1.0. Chunks from RRF ranks 10 and 9 covered the third sub-question (the street boys) |
| LAW-fa-T3-002 | 0 → 1 | Brought the 1334 bill's Art. 9 from RRF rank 8 to #3, so both laws were now in evidence |
| LAW-fa-T3-003 | 0 → 1 | Same gold both times, with the second-law chunk moved up from #4 to #3. The reader got it right this time (borderline) |
| RPT-fa-T3-002 | 0 → 1 | Pulled the 1400 ads report (53 M installs) up from RRF 13 to #4. Coverage 0.25 → 0.75 |
| BOOK-en-T3-003 | 1 → 0 | Demoted the Franklin gold (RRF 3 → 9), so one book filled the evidence and the reader answered NOT_FOUND |
| LAW-en-T1-003 | 1 → 0 | Promoted other Acts (Data Protection, Modern Slavery), and the answer added a wrong second consent authority |
| LAW-fa-T2-001 | 1 → 0 | Pulled in a direct-taxes distractor, and the answer gained an unsupported extra claim |
| LAW-fa-T2-003 | 0 → 0 | Gold coverage 1.0 → 0.5, with 5 distractor items from the tax laws |
| BOOK-fa-T1-001 | 1 → 0 | Identical evidence at the top and near-identical answer text. This is judge noise |

**The top 3 mechanisms behind HYBRID+rerank's accuracy:**
1. **Deep promotion.** The cross-encoder reads the question and the chunk together, so gold sitting at RRF ranks 8–29 enters
   the packed top 8 (23 chunks in 16 questions). This is how the second document of a cross-doc question arrives
   (LAW-fa-T3-002, RPT-fa-T3-002).
2. **Several gold chunks per question.** The top of the list fills with several full chunks from the answering passage,
   not one, which completes multi-part answers (BOOK-en-T1-001, BOOK-en-T2-002).
3. **Whole ~850-char chunks packed to the full 6,000-char budget.** The reader always sees complete passages, with 2%
   fragments against LEX3's 14%.

Its failure mode is **same-genre distractors**. In LAW, other Acts' penalty and consent clauses score as relevant (LAW-en-T1-003, LAW-fa-T2-001,
LAW-fa-T2-003). It also crowds out the second book (BOOK-en-T3-003).

---

## 3. LEX3 against the hybrids, question by question

**LEX3 wins (LEX3 correct, the hybrid wrong)**

| question | vs | mechanism |
|---|---|---|
| LAW-en-T3-003 | H and HR | Both hybrids had **zero** gold. The lex vote (rank 1) lifted the Bribery Act s.11 chunk from BM25 rank 11 with no dense rank, and the Modern Slavery penalty from dense rank 9. Rescoring the hybrid's own deep candidates is what won |
| LAW-fa-T2-003 | H and HR | Routing and lex kept all 6 items in the named law (Cooperatives 1350), with both gold units at #1 and #2. The hybrids mixed in direct-taxes law and the reader said "pays tax" |
| RPT-en-T3-002 | H and HR | One block reserved per period brought in the 2023 note with the 2022 comparative (€304.3 m). The hybrids had that report but the reader took the 106.5 figure |
| BOOK-en-T1-001 | H | Three gold blocks at the top (lex agrees with bm25 and dense). HYBRID's answer added an unsupported claim |
| BOOK-en-T2-002 | H | Facet-split requirements plus follow-up covered all three sub-questions: 5 gold blocks, coverage 1.0 against 0.5 |
| LAW-en-T1-003, LAW-fa-T2-001 | HR | These are the reranker's distractor losses above. LEX3 stayed inside the named Act |

**LEX3 losses (a hybrid correct, LEX3 wrong)**

| question | vs | mechanism |
|---|---|---|
| BOOK-en-T1-002 | H and HR | The gold table was cut to an **87-char fragment** and the reader answered NOT_FOUND. Fragmenting |
| BOOK-en-T1-003 | HR | The gold was in a **207-char fragment** (b219 and b220 split) and the reader answered NOT_FOUND. Fragmenting |
| LAW-fa-T3-001 | H and HR | 5 of 6 slots went to pieces of the 1334 bill (one of them 166 chars), and Art. 3 was missing. Entity × period requirements created spurious "partial" rows (a 1334 bill paired with 1364), and the follow-up returned 0 new candidates |
| LAW-fa-T3-002 | HR | The 1334 bill filled 5 slots with 98, 87, 231, 325 and 550-char pieces, and Art. 9 was missing. The reranker found it at RRF rank 8 |
| RPT-fa-T3-002 | HR | Missed the 1400 ads report (53 M). The evidence stayed in the annual reports and included a 40-char fragment. The reranker lifted that doc from RRF 13 |
| LAW-fa-T3-003 | HR | Gold was present in every arm and the reader misread Art. 74. The period range filled 1345–1352, giving 8 requirements, a block cap of 8 and 8 follow-up rows |
| BOOK-en-T3-003 | H | The Douglass-age gold (b98) was absent, coverage 0.5. Its item was at RRF rank 25 with no reservation |
| BOOK-fa-T1-001 | H | Same answer as HYBRID+rerank. Judge noise |

**Pattern:** LEX3 wins through **focus and reservation**: it stays in the named document, gives one block per period or
facet, and votes up deep candidates. It loses through **fragmentation and crowding**: tiny blocks, 6 slots filled by one
region or document, and spurious requirements. Its wins never depend on candidates the hybrid lacked.

Confidence split on the question channels. When BM25 and dense agree on the top-1 chunk (24 questions), LEX3 gets 15 right
against 13 for HYBRID and 13 for HYBRID+rerank. When they disagree (36 questions), LEX3 gets 13 against 14 and 15. **LEX3's planning
pays off where the hybrid is already confident, not where it is unsure**, so the obvious "skip call 2 when the hybrid is
confident" gate is **not** supported by this dev run. What LEX3 adds there is reservation and packing, not new recall.

---

## 4. Ideas to make LEX both faster and more accurate

The projected savings use the measured means and p50s above. The current mean is 40.2 s and the p50 is 36.6 s.

| # | idea | evidence | projected latency |
|---|---|---|---|
| 1 | **Rescore, don't rescan.** Run LEX's regex terms only over the units of the hybrid candidates (80 chunks plus their section neighbours), not every unit of 8 docs | 0 of 86 gold blocks were lex-only. The LAW-en-T3-003 win was a lex vote on a BM25-rank-11 candidate, which a rescoring pass keeps | lexical 12.3 → ~0.3 s mean (p95 34.7 → <1 s). Follow-up lexical passes get the same cut. **About −12 s mean, −10 s p50** |
| 2 | **Put the cross-encoder on LEX's fused regions** (rerank the top ~50 candidates before build_blocks), with lex/period votes as a tie-break and **per-document diversity** when the question names one Act | The reranker promoted 23 gold chunks in 16 questions and would have fixed LAW-fa-T3-002 and RPT-fa-T3-002. LEX's focus is what blocks the reranker's LAW distractors | **+4.0 s** (measured `rerank_ms`/q; HR p50 is 5.7 s above H). Note the no-local-GPU rule for the dev phase |
| 3 | **Pack like the hybrid.** Cap evidence by characters, not 6 blocks. Merge blocks of one region, set a minimum block of about 400 chars (expand to the section), and fill the 6,000 budget | 14% of LEX3 items are under 300 chars. Two NOT_FOUND losses had gold in 87 and 207-char pieces. Two LAW-fa losses came from 5 slots filled by one bill. About 1,000 chars of budget go unused | 0 s (blocks costs 0.16 s). The answer prompt grows about 250 tokens, well under +0.5 s |
| 4 | **Drop vocab + call 2** and feed call-1 facet terms plus deterministic phrases (headings and labels of the hybrid top hits) straight into idea 1 | Call 2 exists to choose regex phrases, and lex adds votes, not recall. The `no_pick` flag already exists: run it as a dev ablation first | **−10.8 s mean** (vocab 4.95 + pick 5.83), −1 LLM call, about −1,280 tokens/q. RPT-fa gains most (vocab p50 11.3 s) |
| 5 | **Fix the requirements generator.** No entity × period cross-product when the entity carries its own year (a law's name). No range fill for law years. Follow-up only when a requirement is missing, not partial, and with no second lexical scan | LAW-fa-T3-001, T3-002 and T3-003 had spurious partials and 8 period requirements (1345–1352). Follow-up ran on 20 questions, 7 of them correct, and no gold block came from follow-up alone | followup mean 3.5 s → <0.5 s; **BOOK-fa p50 −15 s**; p95 −16 s |
| 6 | **Cache per pool.** Keep one `Reports` for the process (units folded once), read `units.text_fold` instead of folding again, and add an index on `units(doc_id)` | Measured load+fold is 2.1–2.6 s per pass, and each doc load is a full-table SCAN of 251k rows. Every question and every follow-up rebuilt it | −2.4 s per lexical pass (more when cold). Only needed if the full scan stays |
| 7 | **One match per unit for the morph twins** (return literal and morph hits from a single pass) | Measured scan: 9.6 s with twins, 3.8 s literal-only | −5 s on RPT-en/BOOK if the full scan stays |
| 8 | **Overlap call 1 with the code stages.** The question channels, candidate-unit loading and heading/label harvesting do not depend on call 1 | Call 1 takes 7.0 s p50. The code stages it can hide are about 1.5–3 s after ideas 1 and 6 | −1.5 to −3 s |
| 9 | **One batched fact-label query** in vocab (if vocab stays) | 24 queries at about 60 ms each when warm, and more when cold (the first 6 took 3 s) | −1.4 s warm on RPT-fa, more when cold |

**Combined projection.** Apply ideas 1, 3, 4 and 5: question 1.4 s + call 1 7.0 s + lex rescoring 0.3 s + blocks 0.2 s +
answer about 5 s gives a **p50 of about 14 s**, with 2 calls and about 2.1k tokens per question. That is a p50 ratio of
about 2.3× HYBRID and 1.2× HYBRID+rerank. Adding idea 2 gives **about 18 s**, still about 1.5× HR, and brings in the
reranker's deep-promotion gains on top of LEX's focus and reservation wins. Dropping call 1 as well (the reranker plus
lexical votes from the question's own terms) gives **about 11 s**, the same as HYBRID+rerank. That is the natural
"LEX as a rescoring layer on HYBRID+rerank" arm to test next. Ideas 6, 7 and 9 matter only if the full-document scan is kept.

**What to verify before building:** run a `no_pick` LEX3 dev arm against the 60 questions (idea 4 is the only one with no
ablation behind it), and re-judge the 11 disagreement questions twice to size the judge noise. BOOK-fa-T1-001 and
LAW-fa-T3-003 flip on near-identical answers.
