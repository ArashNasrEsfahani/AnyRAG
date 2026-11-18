# Where LEX beats HYBRID, and where it loses (dev win/loss analysis)

2026-09-26. The analysis reads saved results only. It made no model or API calls and re-ran no arm. Scripts, in this folder:
`features.py` → `features.jsonl`; `matrix.py` → `matrix.csv`, `matrix_long.jsonl`, `arms.json`; `analyze.py` →
`analysis.json` and `ANALYSIS_TABLES.md` (every pair × feature cell, with question ids). Step log: `PROGRESS.md`.

## Lenses and noise

| lens | questions | reader / metric | arms |
|---|---|---|---|
| **A24** | idea_test 24 | GLM-5.3-flash reader, strict judge verdict | HYBRID, HR @6,000; HYBRID+KW, DOC-3, LEX6, LEX7.1, LEX5B @6,000; F2 (LEX10b) and HYBRID @15,500; pilot 12: LEX5 ×2, HYBRID-intact |
| **R54** | 54 answerable dev | offline decisive-span pack retention | HYBRID, HR, HYBRID-intact, a0 sparse, a1, F1, F2 @6,000 / 8,500 / 15,500, C1, LEX9, LEX7 O1/O2, LEX5b S-a..S-d, LEX5 replay, LEX4 A/B |
| **D60** | 60 dev | local gemma4:e4b reader, strict (+ gold-unit coverage from `evidence_units`) | B2 (BM25), B3 (dense only), HYBRID, HR, LEX2, LEX3(+), AGENTIC3/5(+) |

Every matrix total was re-derived and matches the registry: A24 HYBRID 12, HR 13, LEX5B 17, LEX7.1 13,
F2@15.5k 15, HYBRID@15.5k 19; R54 F2@15.5k .882, S-c .854; D60 HR 28, LEX3 28, AGENTIC3 25.
`budget_matrix` (HR, LEX7.1 and LEX5B @15,500) was still running and unjudged (18 of 72 run dirs), so it is **not included**.

**Noise.** On the 24-question screen a gap of about ±4 is noise. Feature cells hold 4–20 questions, so any per-cell
difference of 2 or less is noise. Below I call something a *pattern* only when two lenses agree or when a single lens
shows at least 5:1. The D60 lens is bounded by its reader: only 58–75% of full-coverage cells come out correct
(A24 GLM: 64–94%).

## Summary table

| family | shines when | evidence (A:B = family wins : opponent wins, paired) | example ids |
|---|---|---|---|
| **LEX: lexical signal as a *channel*** (m3 sparse inside MaxMean F2; BM25) | (1) **Persian report questions whose gold is a KPI or number fragment**: BM25 *and* dense both rank the gold low, but m3 sparse ranks it 1–4. (2) **Single-document T1 lookups that share a rare term** (BM25) | R54 F2 vs HYBRID @6k **7:1** (fa 5:1, calculation 4:0, gold scattered 6:0); @15.5k 7:2. a0 sparse alone matches 6 of F2's 7 wins. D60 B2 vs B3 answers T1 **4:0**, rare term shared 4:1 | RPT-fa-T2-001 (ranks bm25 13 / dense 18 / sparse 1), BOOK-fa-T2-002 (41/10/4), RPT-fa-T2-002 (12/1/1), RPT-fa-T3-003 (11/1/3); T1: RPT-fa-T1-001, RPT-fa-T1-002, BOOK-fa-T1-001, BOOK-en-T1-001 |
| **HYBRID / meaning search** | **Cross-document (T3) and multi-part questions, calculation and comparison questions, paraphrased questions.** Also at a large budget: HYBRID@15.5k converts evidence into answers best | D60 coverage B3 (dense) vs B2 (BM25) **15:5**: T3 11:1, 2+ docs 11:1, calculation 6:0, comparison 5:0, paraphrase gap 8:0. D60 answers B3 vs B2 T3 **5:0**. R54 HYBRID vs a0 sparse: 3+ requirements 9:1, English 8:2. A24 @15.5k HYBRID vs F2 **4:0** | T3: RPT-fa-T3-002, BOOK-en-T3-003, LAW-fa-T3-001/003, LAW-en-T3-001 |
| **SELECTOR (LLM picks the bundle)** | (1) **Gold scattered over ranks or documents** (T3, second document): a retention gain. (2) **Multi-part T2 explanations**: a *precision* gain, because the same spans reach the reader with fewer distractors | A24 LEX5B vs HR **5:1** (T2 4:0, explanation 4:0, English 4:0); in 4 of those 5 wins HR also held the full decisive span. Full-span → correct: LEX5B 88% vs HR 65%. R54 S-c vs HR **9:4** (T3 7:1, 2+ docs 7:1); S-c vs F2 on the same candidates **11:2** (scattered gold **10:0**) | A24: BOOK-en-T1-001, RPT-fa-T2-002, RPT-en-T2-001, RPT-en-T2-002, LAW-en-T2-001. R54: BOOK-en-T3-002, LAW-en-T3-002/003, RPT-en-T3-003 |
| **AGENTIC** | **Arithmetic over two verified operands; entity × attribute comparison across documents; assembling a multi-part legal rule.** English only | 3 unique D60 wins that every B2/B3/H/HR/LEX arm missed; 5 questions right where H and HR were both wrong. It **loses** 7 where H or HR was right, 5 of them Persian (AGENTIC3 vs HR: fa 1:5, en 4:3) | RPT-en-T2-003, RPT-en-T3-003, LAW-en-T2-001 (+ RPT-en-T2-001, LAW-fa-T2-003) |

## (a) Where lexical signals beat meaning search

- **Exact numbers do not favour lexical search on this set.** Only 10 of 54 answerable questions share a number with
  their span, and those numbers are years or constraints, not the answer. On that feature: F2 vs HYBRID 1:0; B2 vs B3
  coverage 1:3. Named source or article: no consistent sign. Year constraint: no consistent sign.
- **The real lexical win is the m3 *learned sparse* channel on Persian RPT KPI and number text.** All 4 fa wins listed
  in the summary table have the gold at sparse rank 1–4, while BM25 ranks it 11–41. Dense ranks it 1 in two of them
  and 10–18 in the other two. Under plain RRF(BM25, dense) that single strong sparse hit is diluted. MaxMean F2 lets it
  through. Table/KPI evidence: F2 2:0 and a0 sparse 2:0 (n=4, small).
- **BM25 wins single-document T1 questions that contain a rare term** (D60 answers 4:0 on T1; 3 of the 4 are Persian).
- **Rare names and codes:** the "rare term shared" feature (document frequency ≤1% of the pool) favours B2 in answers
  (4:1) but not in coverage (3:4). It is weak.
- **A keyword *boost* or query rewrite hurts:** HYBRID+KW vs HYBRID 1:3 (English 0:3); LEX6 vs HYBRID 2:3; DOC-3 11 vs
  HYBRID 12.

## (b) Where meaning search beats lexical signals

- **T3, 2+ documents, calculation, comparison.** Dense-only B3 beats BM25-only B2 on gold coverage 11:1 on T3,
  6:0 on calculation and 5:0 on comparison, and wins every paraphrase-gap question on coverage (8:0).
- **Paraphrase gap** (folded overlap < .5, dense rank ≤ 10 and better than BM25; 10 questions): dense beats BM25 on
  coverage 8:0. Note that F2 also wins these against HYBRID (4:0 at 6k, 5:0 at 15.5k). So "paraphrase" does not mean
  "no lexical channel": it means the fusion must not let a weak BM25 rank drag the dense hit down.
- **Explanation questions with many parts:** HYBRID beats sparse alone 9:1 on 3+ requirements, and 9:1 when all the
  gold sits within the top 6 of the HYBRID ranking.
- **Lexical-only arms lose by never finding the gold.** Of B2's wrong answers, 27 of 34 lack part or all of the gold
  (LEX2: 33 of 38). Of HYBRID's wrong answers, 19 of 32 do.

## (c) Where LLM selection wins

- **Scattered gold.** S-c vs F2 on the *same* top-24 candidates: 11:2 overall, 10:0 where HYBRID's worst gold rank is
  above 6, 6:0 on T3, and 7:2 on 3+ requirements.
- **Answer gains come from precision, not recall.** In LEX5B's A24 wins over HR (5:1), HR already held the full
  decisive span in 4 of 5 (BOOK-en-T1-001, RPT-fa-T2-002, RPT-en-T2-001, RPT-en-T2-002). LEX5B turns a full-span pack
  into a correct answer 88% of the time; HR manages 65%. This matches the diagnosis finding that distractors cost about
  3 of 21 questions.
- **The order fixer is not a selector.** LEX7.1 O2 vs HR retention is 5:9 (Persian 0:5, calculation 1:5), and its
  answers tie 3:3.
- **Where selection does not help:** T1 (no retention wins). Its 4 retention losses to HR are T2 explanations and
  Persian items: BOOK-fa-T2-002, BOOK-en-T2-001, LAW-en-T2-002, LAW-fa-T3-002.

## (d) What AGENTIC won that the others missed (D60, trace mechanisms)

| id | type | winning arm | mechanism visible in the trace | others |
|---|---|---|---|---|
| RPT-en-T2-003 | calculation (share of balances) | AGENTIC5 | The plan asks for a share. The **verified-value table** holds two rows (13.3b customer balances; 5.4bn business deposits), and the answer computes the ratio | HR had **full** coverage but answered NOT_FOUND; LEX3 NOT_FOUND |
| RPT-en-T3-003 | cross-doc comparison (auditors) | AGENTIC3 | `route=timeline`, plan kind `comparison`, one facet per entity. The answer is laid out entity × attribute ("Zalando – KPMG / Delivery Hero – KPMG") and then compared. This is **entity reconciliation** | HYBRID R4 wrong ("new for both"); LEX3 R4 missing |
| LAW-en-T2-001 | multi-part legal rule | AGENTIC5+doc_card | The requirement table collects s.1, s.2, s.6 and the close-connection test (5 accepted rows of 22). This is **rule assembly** | HR abstained (0 coverage); HYBRID 2 of 4 parts. Fragile: plain AGENTIC5 got it wrong |
| RPT-en-T2-001 | 2-part explanation | AGENTIC3/5 | Verified rows supply the Spain → France/Austria detail (R4) | HR and LEX3: R2 wrong, R4 missing |
| LAW-fa-T2-003 | rule + exception | AGENTIC5+doc_card | Gets the housing-cooperative exemption right, but cites Art. 113 instead of Art. 110 | LEX3 also right; HR inverted the exception |

The losses are all Persian except two (BOOK-en-T2-002, BOOK-en-AB-001): LAW-fa-T1-001, LAW-fa-T2-001, LAW-fa-T3-002,
LAW-fa-T3-003 and RPT-fa-T3-002. AGENTIC coverage beats HR on T3 (5:2, one facet per entity) but loses on T2 (3:8)
and calculation (3:6). Its full-coverage → correct rate is no better than the other arms (59–61% vs 58–75% for the others), so the
verification step helps only when operands have to be *combined*.

## (e) Reader-side vs retrieval-side failures (wrong answerable cells)

| arm (lens) | wrong | reader side: full gold present | partial gold | no gold |
|---|---:|---:|---:|---:|
| HR (D60) | 31 | **16** | 11 | 4 |
| HYBRID (D60) | 32 | 13 | 11 | 8 |
| B3 dense (D60) | 33 | 14 | 12 | 7 |
| B2 BM25 (D60) | 34 | 7 | **18** | 9 |
| LEX2 (D60) | 38 | 5 | 15 | **18** |
| LEX3 (D60) | 31 | 11 | 14 | 6 |
| AGENTIC3 (D60) | 33 | 14 | **18** | 1 |
| AGENTIC5 (D60) | 35 | 11 | 17 | 7 |
| HR @6k (A24) | 11 | **6** | 3 | 2 |
| HYBRID @6k (A24) | 12 | 3 | 6 | 3 |
| LEX5B (A24) | 7 | 2 | 5 | 0 |
| LEX7.1 (A24) | 11 | 3 | 6 | 2 |
| F2 @15.5k (A24) | 9 | 4 | 5 | 0 |
| HYBRID @15.5k (A24) | 5 | 1 | 3 | 1 |

The lexical-only arms fail at **retrieval**. HYBRID-family arms fail about half the time at the **reader**.
The selector's remaining failures are partial retrieval. AGENTIC almost always finds *something* (1 no-gold cell)
but rarely *all* the gold (18 partial).
**F2 vs HYBRID @15.5k (0:4 in answers) is not a retrieval result.** All 4 questions have identical decisive-span
retention in both arms. In 2 of the 4, the F2 answer has every unit correct and fails strict only on one extra
wrong or unsupported claim. Full-span → correct is 94% for HYBRID@15.5k and 79% for F2@15.5k.

## Design clues (each tied to its evidence)

1. **Never ship a lexical-only channel for T3 or multi-part questions; keep dense in every arm.** Coverage B3 vs B2
   is 11:1 on T3 and 8:0 on paraphrase gap; B2 and LEX2 wrong cells are 79–87% retrieval failures.
2. **Lexical evidence belongs in a fusion channel, not in the query.** Add m3 sparse and fuse with max-style
   aggregation (MaxMean) so that one strong channel is not diluted by RRF (F2 vs HYBRID retention 7:1; fa 5:1).
   Drop query keyword boosts and rewrites (HYBRID+KW 1:3, LEX6 2:3).
3. **Route Persian KPI/number questions through the sparse-inclusive path.** The F2 and a0 wins are Persian RPT
   questions where sparse ranks the gold 1–4 and BM25 ranks it 11–41. AGENTIC (fa 1:5) and the LEX7 fixer (fa 0:5) are
   the arms that lose on Persian.
4. **Use the LLM selector only when the question has ≥2 parts, ≥2 subjects or a cross-document cue.** The selector
   recovers scattered gold (10:0 vs F2 on the same candidates; T3 7:1 vs HR) and has no T1 wins, so gate it on the
   question's decomposition (a query-time proxy for "gold scattered").
5. **Optimise pack *precision*, not only retention.** 4 of 5 LEX5B answer wins had the full span in HR too;
   conversion is 88% vs 65%. A cheap experiment: prune the HYBRID pack down to the minimal supporting bundle, then fill.
6. **At 15.5k, keep HYBRID's order and let F2 decide only which chunks go in** (one declared test: "F2 membership,
   HYBRID order"). F2's retention gain (+7:2) did not convert: 0:4 on answers with identical spans, 79% vs 94%
   conversion.
7. **Add agentic-style *operand verification* only for calculation and entity-comparison questions.** The wins are
   RPT-en-T2-003 (verified two-row table → ratio) and RPT-en-T3-003 (entity × attribute table). A cheaper form: a
   reader "operand table" step (part → value → quote → compute) triggered by derived/comparison cues. AGENTIC as a
   whole does not beat HR (5:8).
8. **Tighten the abstention rule.** HR answered NOT_FOUND with full coverage (RPT-en-T2-003) and HYBRID-family errors
   are ~50% reader-side, so NOT_FOUND should require that *no* part is supported (it agrees with diagnosis clue 4).
9. **Do not route on "number in question", "year" or "named article".** None of them separates the families
   (every cell ≤ 2 net). The features that do separate them are tier/#documents, #requirements, gold scatter,
   language and paraphrase gap. The first three can be estimated at query time from a decomposition.
10. **Budget beats method at this scale.** HYBRID@15.5k scored 19/24 against 12 at 6k. Every future LEX claim should
    stay matched-budget, and the pending budget_matrix cells (HR/LEX7.1/LEX5B @15.5k) should be rerun through this
    matrix once they are judged (`python matrix.py && python analyze.py`, after adding the columns).

## Feature rules (deterministic; `features.py`)

- **tier / language / domain / pool / screen24**: from the question file. screen24 is the per_q keys of
  `idea_test/results_b.json`.
- **answer type** (first rule that matches): unanswerable → abstain; any required requirement of kind `derived` →
  calculation; answer_format `comparison` → comparison; format `list` or value_type list / kind item → list; all
  value_types number/interval → number (date if the unit mentions year/date); all boolean → yes-no;
  entity (+number) → name-entity; else explanation. The dev set has no date or yes-no primaries.
- **# required units** = required requirements; **# gold docs** = distinct evidence doc_ids; **# gold chunks** =
  HYBRID-index chunks of the same doc whose folded text contains a decisive span (`lex4_replay/decisive_spans.jsonl`)
  or that list the span's unit_id.
- **Rank per channel**: position of a gold chunk in the saved top-40 of BM25 / bge-m3 dense / m3 sparse
  (`lex8/rows8_fusion.jsonl`) and in HYBRID RRF / F2 order40 (`lex10/b/rows10b.jsonl`); 41 = absent. Per span take the
  best gold chunk. The question gets best = min over spans, worst = max over spans, spread = worst − best.
  **gold scattered** = HYBRID worst > 6. **channel advantage** = min(BM25, sparse) best vs dense best.
- **Overlap**: content words (length ≥ 2, fa+en stopword list) of the question vs the union of decisive spans.
  *exact* = lower-cased raw tokens; *folded* = `normalize.fold` + diacritics/ZWNJ removed + 5-character prefix.
  **numbers shared** = numerals (Persian digits mapped) in both; **years** = 13xx/14xx/18xx–20xx.
  **rare term** = folded question token with pool chunk document frequency ≤ 1%. **entities (en)** = capitalised
  non-initial tokens.
- **paraphrase gap** = folded overlap < 0.5 AND dense best ≤ 10 AND BM25 best worse than dense best. The author's
  `wording` tag (paraphrase/ordinary) is kept separately.
- **table/KPI evidence** = any gold unit of kind ≠ paragraph (fact/table) or a `table` unit id.
- **constraints**: year (a year in the question or a requirement time constraint); named source (regex for article /
  section / law / report names, fa ماده/تبصره/بند/قانون..., or a version constraint); numeric condition (comparator
  words or a non-year number); **negation/exception** (en not/except/unless/without/exempt...; fa
  مگر/جز/استثنا/معاف/بدون/نیست/نمی...).
- **Win rule**: answer lenses: A correct and B wrong. Retention/coverage lenses: A ≥ B + 0.05.
