# LEX4 offline retrieval replay (dev, 60 questions): decisive-span retention

2026-09-26. No LLM/API call of any kind. The only model use was local bge-m3 query embeddings: one batched call per
question per requirement set, 120 calls, 354 vectors. Everything else is CPU replay over the saved HYBRID index and
LEX3's saved call-1 plans. Code: `RAG-Paper-upgrades/ragbench/approaches/lex4/` (`spans.py`, `requirements.py`,
`core.py`). Scripts and outputs: this folder. The step log is in `PROGRESS.md`.

**Caveat.** This dev set has been seen many times, and it has only 54 answerable questions: 18 per tier, 3 per cell.
No constant was tuned. Every constant is the value from the design (R2) or from Astra R1. One rule was fixed before
any result existed: a source is resolved only for a single-entity subject (see §3). A difference of 1–3 questions is
noise at this size.

## 1. Decisive spans (step 1): `decisive_spans.jsonl`

A decisive span is the part of a gold evidence quote (question-v2 `evidence[].quote`) that holds the value or phrase
of the requirement it supports. The quote is split at sentence ends and line breaks, and the span is the minimal run of
segments that holds the matched keys. The keys are the value, the aliases and the number forms. Matching and
containment both use `normalize.fold`, so Persian and Arabic letters and digits are unified, harakat are removed and
ZWNJ counts as a space.

- There are **158 spans** over the 54 answerable questions, and **0 are unresolved**: every quote was found at its
  offsets and every required requirement has a span.
- 63 spans are anchored on a key. 89 are the whole quote, because the value is a paraphrase or a boolean and does not
  appear literally.
- 1 span is a table row. 1 grew to the whole quote. 4 are line windows around bare labels, and these are weak: the
  chart numbers "40" and "97" in RPT-fa-T3-002 bundle 2, "3. Order." and "(a) business, or".
- Span length: p50 113 chars, max 390.
- I spot-checked 10 random spans (seed 11). All 10 hold the supported value or proposition, and the key anchors were
  right (4 درصد, 86, 470, 23.6, دیوار, KPMG).
- Two weaknesses: long whole-quote spans are strict, and a spelling variant ("instalments" vs "installments") defeated
  one key.

## 2. The span-retention metric (step 2): `retention_saved.json`

A span counts as present only if it appears exactly, after normalization, inside ONE shown item. Spans are never
stitched across items. A question's score is that of its best evidence bundle. I rebuilt the shown texts from the
saved records (retrieved prefix, with the last item cut) and checked the length of all 1,524 items: every one matches.

| saved arm | span retention | 100% questions | judge correct if 100% | judge correct if not | r(span, strict) | r(unit-ID, strict) |
|---|---:|---:|---:|---:|---:|---:|
| HYBRID | 0.720 | 32/54 | 22/32 | 0/22 | 0.61 | 0.56 |
| HYBRID+rerank | 0.806 | 35/54 | 21/33 | 2/19 | 0.50 | 0.51 |
| LEX3 | 0.709 | 30/54 | 21/30 | 2/24 | 0.62 | 0.59 |
| LEX2 | 0.513 | 20/54 | 15/20 | 1/34 | 0.71 | 0.70 |
| pooled | | 117/216 | **79/115 (69%)** | **5/99 (5%)** | 0.61 | 0.59 |

- Span retention is close to a necessary condition for a correct answer: 0 of 39 cells with zero spans shown were
  judged correct.
- It correlates with the verdicts a little better than the unit-ID proxy does.
- In 8 cells the unit-ID proxy says "full" but the spans are not all shown.

## 3. The replay (step 3)

**Requirements.** They come from LEX3's saved call-1 plans (`requirements.jsonl`).
- Each facet becomes one requirement: the aspect is the facet name plus its terms, the subject is the entities, and
  the free periods form one constraint group.
- A year that belongs to a name stays in the name.
- There is no entity × period product. Comparisons split along ONE axis, and only when facets × members ≤ 8.
- A source is resolved only when a single-entity subject matches exactly one pool title.
- This gives 126 requirements with 0 fallbacks, and 35 of them are source-restricted.
- The no-planner arm uses question clauses instead: 169 requirements.

**Candidates.** The question hybrid takes BM25 top 40 ∪ dense top 40 over the pool. It reproduces HYBRID's saved
top 8 exactly on **60/60** questions. Each requirement adds BM25 top 20 + dense top 20.

**Packages.** Packages are intact chunks. A cut sentence is completed with at most 300 added chars, up to a 1,200-char
ceiling, and table caption and header lines are attached. Evidence is capped at 6,000 rendered chars, headers
included. No item is ever cut: 0 overruns and 0 cut items.

**Lexical score.** It is computed over concept groups, and binding happens inside one local support unit: a sentence
plus its heading, or a table row plus its headers.

## 4. Results (step 4): `report_tables.json`, `tables.md`

Span retention is the mean over 54 answerable questions. "T3 all docs" counts cross-document questions whose shown
evidence has a decisive span from every needed document. Code latency is per question, over all 60, in ms; it excludes
embedding and includes BM25, dense scoring, packages, lexical matching and selection. The saved arms' "code" column is
latency minus LLM wall, so it includes their query embed, rerank or regex scan. It is not the same measurement.

| variant | span ret. | req ret. | 100% q | T1 | T2 | T3 | fa | en | BOOK | LAW | RPT | T3 all docs | chars | items | <300 items | code p50/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A (Astra R1) | 0.607 | 0.579 | 22/54 | 0.852 | 0.511 | 0.458 | 0.580 | 0.634 | 0.563 | 0.718 | 0.541 | 4/18 | 3633 | 4.1 | 9/221 | 244 / 554 |
| B (Claude R2) | 0.681 | 0.649 | 27/54 | 0.907 | 0.483 | 0.653 | 0.691 | 0.671 | 0.623 | 0.731 | 0.689 | 8/18 | 5835 | 6.3 | 13/342 | 251 / 574 |
| B-nolex | 0.718 | 0.700 | 30/54 | 0.870 | 0.594 | 0.689 | 0.698 | 0.738 | 0.660 | 0.832 | 0.661 | 8/18 | 5832 | 6.3 | 15/343 | 96 / 199 |
| B-noreq (question pool only) | 0.692 | 0.665 | 28/54 | 0.907 | 0.530 | 0.639 | 0.691 | 0.693 | 0.623 | 0.736 | 0.717 | 9/18 | 5793 | 6.2 | 11/335 | 206 / 416 |
| B-plain (no cover) | 0.706 | 0.680 | 28/54 | 0.907 | 0.544 | 0.667 | 0.722 | 0.690 | 0.637 | 0.769 | 0.712 | 9/18 | 5827 | 6.3 | 9/338 | 248 / 581 |
| HYBRID-intact (packing control) | 0.729 | 0.717 | 32/54 | 0.981 | 0.623 | 0.583 | 0.722 | 0.736 | 0.658 | 0.821 | 0.708 | 8/18 | 5787 | 6.1 | 8/330 | 65 / 152 |
| B-noplanner (question clauses) | 0.715 | 0.688 | 27/54 | 0.981 | 0.557 | 0.606 | 0.735 | 0.695 | 0.697 | 0.772 | 0.675 | 7/18 | 5832 | 6.3 | 13/338 | 428 / 820 |
| HYBRID (saved) | 0.720 | 0.708 | 32/54 | 0.926 | 0.623 | 0.611 | 0.722 | 0.718 | 0.603 | 0.821 | 0.736 | 9/18 | 5955 | 6.6 | 7/354 | (8 / 28) |
| HYBRID+rerank (saved) | **0.806** | **0.770** | **35/54** | 0.981 | 0.748 | 0.690 | 0.877 | 0.736 | 0.739 | 0.764 | 0.917 | 9/18 | 5958 | 6.6 | 7/357 | (3,995 / 5,974) |
| LEX3 (saved) | 0.709 | 0.676 | 30/54 | 0.963 | 0.494 | 0.669 | 0.704 | 0.714 | 0.654 | 0.784 | 0.689 | 9/18 | 4915 | 6.1 | 45/327 | (18,775 / 41,400) |
| LEX2 (saved) | 0.513 | 0.496 | 20/54 | 0.731 | 0.410 | 0.398 | 0.586 | 0.440 | 0.495 | 0.647 | 0.397 | 4/18 | 5639 | 9.0 | 95/486 | (3,049 / 9,677) |

The batched requirement embed (question + requirements, mean 3.1 texts) takes p50 184 ms and p95 296 ms. The clause
batch takes p50 196 ms and p95 278 ms.

Paired counts give, per question, better / worse on span retention, and 100% gained / 100% lost:

| pair | better | worse | 100% gained | 100% lost |
|---|---:|---:|---:|---:|
| A vs B | 2 | 8 | 0 | 5 |
| B vs B-nolex | 2 | 8 | 1 | 4 |
| B vs B-noreq | 2 | 3 | 0 | 1 |
| B vs B-plain | 2 | 6 | 1 | 2 |
| B vs HYBRID-intact | 6 | 10 | 1 | 6 |
| B vs B-noplanner | 6 | 10 | 4 | 4 |
| B vs HYBRID+rerank | 4 | 13 | 1 | 9 |
| B-nolex vs HYBRID+rerank | 7 | 12 | 3 | 8 |
| B-nolex vs LEX3 | 10 | 7 | 3 | 3 |
| HYBRID-intact vs HYBRID | 1 | 1 | 1 | 1 |

## 5. Verdicts

**A vs B: B is better, but neither wins.**
- B beats A on 8 questions and loses on 2 (100%: 27 vs 22).
- A's absolute eligibility threshold, together with a fill that only adds eligible packages, leaves the budget unused:
  rendered evidence has a p50 of 3,271 chars. 23 questions get under 3,000 chars, and BOOK-fa-T2-002 gets no evidence
  at all, because nothing reached S ≥ 0.5.
- A is also the worst on T3: 4/18 questions get all their documents.
- B is still **below** the plain controls: HYBRID 0.720/32, HYBRID-intact 0.729/32 and LEX3 0.709/30. It is far below
  HYBRID+rerank (0.806/35), losing to it on 13 questions and winning on 4.

**Does the full-document regex scan contribute unique spans? Almost never: 1 of 100.**
- LEX3's evidence shows 100 decisive spans. 97 are inside a question top-80 chunk. 2 are only in the LEX4 requirement
  lists.
- 1 is in neither: BOOK-en-T2-002 E4, the "bread for the street boys" sentence. It sits inside a single pool chunk that
  neither channel reached. LEX3's block for it carried inherited region ranks (bm25 10, dense 9, lex 6), so it arrived
  through a lexical window or follow-up region. This is Astra's region-vs-span point, and it is real but tiny.
- 0 spans needed text that is not inside any single chunk.

**Does requirement retrieval recover second documents? No.**
- T3 questions with all documents shown: B 8/18, B-noreq 9/18, HYBRID-intact 8/18, and every saved arm except LEX2
  9/18.
- Across all LEX4 variants, the only decisive span that entered final evidence from outside the question top-80 was
  BOOK-fa-T3-003 E1.
- The requirement lists act as a reranking vote, not as new recall.
- B's T3 span retention (0.653) is higher than HYBRID-intact's (0.583). Four of B's six gains over HYBRID-intact are
  T3 questions: BOOK-fa-T3-003, LAW-en-T3-002, RPT-en-T3-002 and RPT-fa-T3-003.
- But B pays for this on T2 (0.483 vs 0.623). Planner requirements, through 3 of the 5 fused lists, push the
  question-channel gold out: LAW-en-T2-001/002/003, RPT-en-T2-002 (6 requirements from a period split that included
  "since then"). The same happens on one T1 question, BOOK-fa-T1-002.

**Does cover-then-fill beat plain packing? No.** B-plain scores 0.706/28 and B scores 0.681/27; B is better on 2
questions and worse on 6. The reservations cover 124 of 126 requirements, yet showing "one eligible package per
requirement" does not show the decisive span more often than filling by fused rank.

**Other ablations:**
- *No lexical list:* removing it helps. B-nolex scores 0.718/30 against B's 0.681/27, and B wins only 2 while losing 8.
  It also cuts code time from 251 to 96 ms. By the design's own rule, the lexical-component claim should be dropped.
- *No planner:* the question-clause arm (0.715/27) is no worse than the planned B.

**Packing.**
- Intact packages remove LEX3's fragmentation: 2–4% of items are under 300 chars, against LEX3's 14%. The evidence
  also uses the budget (≈5,800 rendered chars vs 4,915).
- HYBRID-intact is the same as HYBRID (1 better, 1 worse), because HYBRID's items were already whole chunks.
- So packing explains LEX3's fragment losses, but it does not add span retention over the hybrid.

**Latency.** LEX4's code path is cheap: B has a p50 of 251 ms and a p95 of 574 ms, B-nolex 96 / 199, plus one
≈0.2 s embed batch. LEX3's code path is ≈18.8 s. Latency is not LEX4's problem. Its problem is that retention does not
improve.

## 6. Which variants deserve a paid reader run

R2's go/no-go rule was: *only configurations that do not lose span recall vs LEX3 AND HYBRID+rerank.* **No LEX4
variant passes.**

- The best LEX4-family arm, B-nolex, is at LEX3's level (0.718 vs 0.709; 10 questions better, 7 worse). It is 0.088
  below HYBRID+rerank, with 8 questions losing 100% retention and 3 gaining it.
- Step 2 maps 100% retention to 69% judge-correct and non-100% to 5%. On that mapping, B-nolex would score roughly 3–4
  answers below HYBRID+rerank (net about −3). The accuracy gate needs +3.
- My recommendation: **do not spend reader money on A or B.**
- If the ~$14 must test anything from this family, the only defensible pair is **B-nolex with HYBRID-intact as its
  control**. It would test whether per-requirement reservation helps on T3 (B-nolex T3 0.689 ≈ HYBRID+rerank 0.690),
  but the replay predicts that it loses overall.
- The one positive LEX4 signal is limited to T3, which is where HYBRID+rerank is already as good. The cheaper
  conclusion for the paper: the reranker, not requirement-guided lexical retrieval, is what improves evidence on this
  dev set.

## Files

- `decisive_spans.jsonl` and `decisive_spans_summary.json`: step 1.
- `retention_saved.jsonl` and `.json`: step 2, saved arms and verdicts.
- `requirements.jsonl`, `vectors.npz`, `vectors_keys.json` and `embed_timing.jsonl`: step 3a.
- `replay_rows.jsonl`, `provenance.jsonl`, `replay_checks.json` and `step3b.log`: step 3b.
- `report_tables.json`, `tables.md` and `reqrecall.json`: step 4.
- Scripts: `common.py`, `step1_spans.py`, `step2_retention.py`, `step3a_embed.py`, `step3b_replay.py`,
  `step4_report.py`, `step4b_reqrecall.py`.
