# RESIZED 2026-09-25 by the user's review budget (overrides the counts and timeline below)

The user reviews every question in review_game at ~20 s each, max 1 hour total. Sizes: 18 cells = 3 tiers (T1 fact, T2 within-document, T3 cross-document) x 3 domains (RPT reports, BOOK non-fiction, LAW statutes) x 2 languages (fa, en).
- DEV: 3 per cell = 54 + 6 unanswerable = 60.
- TEST: 5 per cell = 90 + 6 unanswerable = 96.
- Total 156 reviews (~52 min) + ~8 min re-checking fixes. The 71 old dev questions: regression suite only (not re-reviewed).
- Timeline in agent time: sourcing + extraction ~half a day; authoring + self-check a few hours; user review ~1 h; then methods.
- Defaults taken (user may override): English law = EU regulations + UK Acts; Persian books fallback = Persian non-fiction articles/reports; the Red Book = private dev stress source only.
- Every tier searches the whole domain-language pool (Astra); a given-document condition is diagnostic only.
- Primary systems include HYBRID+rerank (baseline only; decided 2026-09-25). Stop/continue gates: work/analysis/STRATEGY_ASTRA_OPINION.md.

---

# Datasets v2: the agreed design (three tiers, three domains, two languages)

Agreed 2026-09-25. This is Claude's draft (`DESIGN_DRAFT.md`) revised with Astra's critique (gpt-6-astra, high effort,
read-only). The brief is `ASTRA_BRIEF.md` and the full critique is in `ASTRA_session.log`: its final answer comes after
the last line that is exactly `codex`, and the file is UTF-16. The question schema is `SCHEMA.json`. Section 13 lists
where Claude and Astra still disagree. **Nothing here is authored yet.** Decisions that are the user's are in
section 14.

## 1. Goals
- **T1, local fact.** One coherent local passage, list or table region answers the question, including the labels
  that tie the answer to its subject and constraints.
- **T2, within-document integration.** One document is enough, but a complete answer has to combine at least two
  distinct evidence regions of that document.
- **T3, cross-document integration.** Every sufficient body of evidence needs at least two distinct documents.
- **General RAG, not only company reports.** Questions and schemas use domain-neutral slots: subject; aspect
  (property or relation); constraints (time, place, version, population, unit, condition). The set has three domains
  in two languages.
- **Two kinds of set.**
  - Dev is for fast, informative iteration while LEX and AGENTIC are improved. LEX's goal is the best balance of
    speed and accuracy.
  - Test is for a credible one-shot evaluation.
- **The user is the final gate.** The user checks every question in the review game (`review_game/`). The game shows
  the question, the expected answer, the answer requirements, and the evidence highlighted in the source text.

## 2. Tier rules (Astra: evidence dependence, not character distance)
A question's tier is set by the **simplest sufficient evidence anywhere in the frozen pool**. That search includes
restatements and answers that can be derived. Character distance, section count, evidence length and document count
are **recorded as descriptive metadata and reviewer flags. They never decide a tier.** The draft's rules of 1,000 and
1,500 characters are dropped: all six legacy within-report lists fall inside spans of 172 to 885 characters.

| Tier | Admission rule | Reviewer's removal test |
|---|---|---|
| T1 | One coherent local region (a passage, list or table region, possibly several extraction units) directly gives the answer and its binding labels | - |
| T2 | One document is sufficient; no single local region gives the complete answer; the question requires a **named integration operation** | Removing any one required region makes a specific answer obligation impossible or changes its meaning. "The answer gets shorter" does not count, except for an explicitly bounded list |
| T3 | Every sufficient evidence bundle spans at least 2 documents; no single document supplies or supports deriving the complete answer | The same test, per document. A document that another document can substitute for is not "indispensable"; alternative bundles are recorded |

- **Document identity follows publication structure.** Splitting one PDF into chapters must not create T3
  questions. Editions, revisions and separately published volumes are distinct documents. They still stay in one
  **source family** for splitting and analysis.
- **Unanswerable questions have no evidence tier.** Their `tier` is null and `intended_tier` records the attempted
  task: T1 or T3. They are scored separately.

**Subtype quotas per 96 test questions:**

| T2 subtype | Count | T3 subtype | Count |
|---|---:|---|---:|
| Definition or reference bridge (resolve and apply) | 24 | Cross-time or cross-version comparison | 24 |
| Rule plus condition or exception | 24 | Cross-subject comparison | 24 |
| Comparison or derivation within the document | 24 | Cross-document bridge (doc A tells what to look for in doc B) | 24 |
| Bounded synthesis or list | 24 | Bounded aggregation or reconciliation | 24 |

- These quotas are global. A domain is not forced into operations that do not fit it: statutes carry most of the
  rule-and-exception questions, reports most of the derivations.
- T3 must have at least **12 non-numeric comparisons or reconciliations**.
- T3 targets 72 two-document questions and 24 three-document questions.
- Dev follows the same proportions, as closely as its small cells allow.

"Give the founding date and the employee count" is two lookups, not T2. "Which activities fall under the definition,
and which does the exception exclude?" is T2.

## 3. Domains and sources
| Code | Persian | English |
|---|---|---|
| **RPT** (company reports) | Corpus C, 97 natural reports from 42 companies (frozen) | Corpus D, 15 reports from 11 companies, **plus new reports acquired to form independent test blocks** |
| **BOOK** (historical or expository non-fiction prose, modern register) | Candidates: Hassan Pirnia, «ایران باستان» (multi-volume); Ahmad Kasravi, «تاریخ مشروطهٔ ایران» and «تاریخ هیجده‌سالهٔ آذربایجان»; other public-domain modern prose from verified transcriptions. **No classical literature as a silent substitute**: it changes the register | Project Gutenberg non-fiction spread over many authors, for example Darwin (*Origin* 1st and 6th editions, *Descent*), Mill, Smith, Gibbon, Macaulay, Tocqueville (English), Parkman. Test questions must not concentrate on Darwin and Smith |
| **LAW** (statutes, regulations and their amendments; the same genre in both languages) | Iranian statutes from the Majlis Research Center (rc.majlis.ir): Civil Code, Labour Law, Commercial Code, Direct Taxation Act (versions before and after the 1394 amendment), Constitution (1358 and 1368), Electronic Commerce Law, and others | EU regulations from EUR-Lex (GDPR, Data Governance Act, AI Act, DSA, and others) and UK legislation (legislation.gov.uk, point-in-time versions, Open Government Licence) |

- **NIST and other technical standards are left out of the core.** Astra's reason: mixing Persian statutes with
  English standards confounds language with genre. They may be added later as a separate diagnostic domain.
- **The Red Book (Jung) is only a private dev stress source, outside the balanced core.** Its extraction rebuilt word
  spaces from glyph gaps, so every quote from it needs a fidelity check. It is also a copyrighted edition.

Source rules:
1. **Accepted text sources.** Born-digital text or verified human transcriptions. No OCR of scans. Check
   completeness: the Wikisource index of one *Tarikh-e Mashruteh* file records missing pages.
2. **Provenance record for every source**, in `sources_v2.json`: edition, publication or effective date, URL, sha256
   of the downloaded file and of the extracted text, completeness, reuse basis (licence or permission, and its
   jurisdiction), and extraction verdict.
3. **Pipeline.** Each source goes through `tools/extract/`, gets a quality check, and is frozen with
   `freeze_extend.py` BEFORE any authoring.
4. **Pool size.** Each domain-language cell needs at least 7 disjoint blocks (4 test, 2 open dev, 1 check). Its pool
   therefore needs **at least 14 documents**, plus distractors. The draft's 6-10 is withdrawn.
5. **English reports.** Acquire by block need, not by count. Adjacent annual reports often restate each other's
   figures, so check that a cross-year question still needs both reports.

## 4. Counts
Counts are unique information needs. Paraphrases of one need are not extra questions.

### Answerable, new (balanced: 16 per tier x domain x language cell in test)
| Domain-language | Open T1 | Open T2 | Open T3 | Check T1 | Check T2 | Check T3 | Test T1 | Test T2 | Test T3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RPT-fa | 6 | 6 | 6 | 2 | 2 | 2 | 16 | 16 | 16 |
| RPT-en | 6 | 6 | 6 | 2 | 2 | 2 | 16 | 16 | 16 |
| BOOK-fa | 6 | 6 | 6 | 2 | 2 | 2 | 16 | 16 | 16 |
| BOOK-en | 6 | 6 | 6 | 2 | 2 | 2 | 16 | 16 | 16 |
| LAW-fa | 6 | 6 | 6 | 2 | 2 | 2 | 16 | 16 | 16 |
| LAW-en | 6 | 6 | 6 | 2 | 2 | 2 | 16 | 16 | 16 |
| **Total** | **36** | **36** | **36** | **12** | **12** | **12** | **96** | **96** | **96** |

### Supplementary
| Domain-language | Open abstention | Test abstention | Legacy (dev-legacy suite) |
|---|---:|---:|---:|
| each RPT cell | 2 | 4 | fa 47, en 24 |
| each BOOK / LAW cell | 2 | 4 | 0 |
| **Total** | **12** | **24** | **71** |

- Each dev cell's abstention questions are one intended T1 and one intended T3. Each test cell's are two of each.
- **Totals.**
  - Dev: 108 open + 36 check + 12 abstention = **156 new**, plus up to 71 legacy questions.
  - Test: 288 answerable + 24 abstention = **312**.
  - Authors over-produce about 15%. Agents review the reserve before the user sees it, and it is ordered in advance
    so a rejected question is replaced from the same stratum.
- **Dev-fast: 36 questions.** Two per answerable cell, taken from dev-open, for the inner loop. Legacy regressions are
  reported separately.
- **No run time is promised for dev-fast** until the revised systems have been measured.
- **Source blocks.** Every block lies wholly inside one split.

  | Split | Blocks per cell | Questions per block |
  |---|---|---|
  | Test | at least 4 (aim for 6) | about 12 |
  | Open dev | 2 | about 9 |
  | Check | 1 | 6 |

  That gives 24 to 36 test blocks, and the analysis must say so: 288 questions are not 288 independent contexts.

### Report-source allocation (a proposal, checked against the source inventory before freezing)
- **RPT-fa open dev.** The legacy blocks: jabama, alibaba, flytoday, okala, tipax, khanoumi, nobitex, bitestan, podro,
  karnameh, dataak. Plus divar, cafebazaar and tapsell, which were already dev material in `pilot*`.
- **RPT-fa test.** Blocks drawn from snapp, digikala, irancell, irvc, sheypoor, yektanet, mci, triboon, najva,
  derakcloud, dezhino, trendix and techrasa.
- **RPT-en.**

  | Split | Companies |
  |---|---|
  | Dev | zalando, deliveryhero, vodafone, wise, turkcell |
  | Test | mowi, arcelormittal, leroy, maersk, riotinto, tatasteel, plus newly acquired companies and years |

- **Exposure.** Several test report sources were exposed earlier: `test_fresh` and `test.jsonl` were run and their
  answers analysed. The report part of the test is therefore called **fresh-question evaluation**, and every source
  carries an exposure class. No claim of generalisation to unseen companies.

## 5. Search settings (Astra: the pool is primary for every tier)
| Tier | Primary | Diagnostics |
|---|---|---|
| T1 | frozen domain-language pool | source document supplied |
| T2 | frozen domain-language pool | source document supplied (the "chat with this document" setting) |
| T3 | frozen domain-language pool | sufficient document set supplied |

- **Pools.** Six frozen manifests: `RPT-fa`, `RPT-en`, `BOOK-fa`, `BOOK-en`, `LAW-fa`, `LAW-en`.
  - Indexes are built once over each full pool. The document-supplied conditions use **scope masks**, so BM25
    statistics stay fixed.
  - A diagnostic oracle scope is always kept apart from the independently built pool. A pool is never derived from
    gold references (P0 rule).
- **Dev and test share the searchable pool**, so dev documents act as distractors. Describe the test accordingly: held-out questions and source blocks
  inside a known collection.
- **Every question must have a unique answer in its pool.** Its subject and constraints must bind the answer, so every
  condition is valid.
- **What the diagnostics cover.**
  - On dev, every diagnostic runs for every question.
  - On test, the document-supplied condition covers **72 preset questions**: 24 per tier, 4 per domain-language cell.
    It runs for HYBRID, LEX and AGENTIC.
  - A **gold-evidence reader control** covers 36 packets: 2 per tier x domain x language cell. It separates
    retrieval and packing failures from synthesis failures.
  - **B0 (no retrieval) runs on all 288 test questions.** It is reported by domain and tier as an estimate of how much
    each question depends on retrieval. The test keeps questions that B0 answers correctly: nothing is filtered after
    results are seen.

## 6. Authoring, review and sealing
1. **Source curator** (Claude subagent, one per block). Never sees any method output. Picks information needs, writes
   `requirements` and `evidence` with exact quotes and offsets, and writes the necessity audit (T2/T3) or the absence
   audit (abstention).
2. **Question writer** (a separate subagent). Sees only the requirement spec and the evidence. Writes the natural
   question and the expected answer. About half use ordinary terminology and half natural paraphrase. Persian follows
   the `persian-report-prose` rules. No obscure synonyms invented to make a method fail. No question built to reverse
   the earlier LEX result.
3. **Mechanical verifier** (`verify_v2`, extending `verify_multidoc.py` on branch `upgrades`).
   - Checks: schema; quotes equal the frozen text at their offsets; unit ids exist; every requirement is supported by
     at least one evidence bundle; derived requirements have their inputs.
   - Flags: copied spans, answer leakage, near-duplicates, lexical-overlap band, and the exposure check against every
     earlier question file.
   - Records the descriptive metadata: distance, regions, documents.
   - It checks that the audits are **complete in structure**; it cannot certify their conclusions.
4. **Independent source reviewer** (a third subagent). Opens the documents and checks answers, tier (by the removal
   test), necessity against the **final expanded pool**, and absence. Verdict: accept, fix or reject, with the exact
   fix.
5. **The user in the review game.** Rates, edits and tags every question. Accepting binds to a **content hash** of the
   question, answer, requirements and evidence, so a later substantive edit voids the acceptance.
6. **Claude** applies the edits, reruns the verifier, and freezes each set with sha256 in a manifest.

**Sealing the test.** Separate authoring agents are not enough on their own.
- Test files live under `work/datasets_v2/test/`. Method-building agents never read them. No shared summary or
  inherited agent context may reveal test content.
- The game gets an explicit split-access policy: it does not auto-discover sealed files.
- The user reviews test questions only **after** these are frozen:
  - method code and configurations, and baseline tuning allowances;
  - primary contrasts and analysis weights;
  - inclusion and rejection rules and the replacement order;
  - the scoring rubric and the retry policy;
  - the diagnostic subsets and runtime budgets.
- During test review the user sees no method answers. Fixes to test questions trigger no method tuning.

**Dev-check.** The user helps design the methods, so check questions cannot be read during development and still
count as hidden.
- Dev-check is reviewed only after a provisional method freeze. Then comes **one** check evaluation.
- If that result drives more tuning, the check set becomes open dev. A new check set is needed before any later claim
  of an independent gate.

**Review-game changes needed** (the game is being built in parallel):
- render `requirements` as the answer-point list;
- read the top-level `necessity_audit`, `absence_audit` and `integration_operation`;
- bind reviews to `content_hash`;
- apply the split-access policy;
- use code-point offsets correctly in JavaScript (Python code points are not JavaScript UTF-16 indexes).

## 7. Reusing the 71 legacy dev questions
- **Where they go.** All 71 go into a separate **dev-legacy** suite (open, scored separately). None goes into
  dev-check or test.
- **Their old acceptance and family labels are not inherited.** They are converted mechanically to the v2 schema, then
  re-reviewed semantically for tier.
  - **Expected re-tiering.** Routing (16) becomes T1. Cross-year, cross-company and aggregation (43) mostly become T3,
    but each is checked again against the expanded pool. The within-report lists (6) are T1 unless a real integration
    operation exists (C-046 is two lookups). The unanswerables (6) get tier null.
  - **Mapping.**

    | Old field | v2 field |
    |---|---|
    | company | `subject` |
    | metric | `aspect.label` |
    | period | `constraints.time`, keeping fiscal year, calendar year and report year apart |
    | definition | `constraints.definition` (not blindly `condition`) |
    | source spans | `evidence` with ids |
- **Repairs before use.**
  - **C-021:** the comparison is unsupported ("528" against "more than 150" supports neither "larger" nor about
    3.5x). Ask for the stated figures only.
  - **Derived answers:** no legacy record has a `derived` unit. Every requested difference, total or winner (C-003,
    D-013 and the others) becomes an explicit derived requirement.
  - **Atomicity:** D-024's eight percentages become eight requirements.
- **Two uses.** The legacy suite is the regression check against the old dev run
  (`work/multidoc/devrun/`). It also goes through the game, marked "legacy, Claude-reviewed", when the user has time.
  It is not on the critical path.

## 8. Schema (full definition in `SCHEMA.json`)
Three layers:
- **question:** id, revision, split, tier, intended_tier, subtype, language, domain, pool, doc scope,
  information-need id, source block and family, question text, expected answer, answer format;
- **answer requirements:** atomic propositions (subject, aspect, constraints, typed expected value, comparator,
  precision, required or optional, kind, derivation);
- **evidence:** stable ids, doc and unit ids, exact quote, code-point offsets against a frozen text hash, role, and
  the requirement ids each piece supports.

Around these layers:
- `evidence_bundles`: each bundle is a complete, sufficient set of evidence ids, and any one bundle is enough;
- `integration_operation` and `necessity_audit` for T2/T3;
- `absence_audit` for abstention;
- `exposure` and `provenance`;
- `review`, bound to `content_hash`;
- `notes`.

**Visibility.**
- Methods see only `id`, the question text, `language`, `pool` and the run condition's scope.
- Requirements, evidence, tier, audits and the expected answer are **evaluator-only**, as in today's `Query`
  separation.
- The judge's tuple becomes `(subject, aspect, constraints, expected_value)`, replacing `(company, period, metric,
  definition, value, unit)`.

## 9. Scoring
- **Primary outcome:** strict answer success on the 288 answerable test questions. The headline is the
  **equal-cell macro average** over the 18 tier x domain x language cells.
- **Secondary:** requirement recall, substantive-claim precision and F1, averaged per question first.
  - T2/T3 score the **requested relationship or conclusion**, not only whether the operands appear.
  - Derived requirements are scored separately from operands.
- **Diagnostics:** wrong bindings (right value, wrong subject, time or version), wrong bounds or units, omissions,
  unsupported additions, false abstention, citation support; also gold-document recall, all-required-documents hit,
  requirement-evidence coverage after packing, and routing coverage.
- **Abstention supplement:** correct abstention and fabricated completion. Never pooled into the headline.
- **Judge.** New, versioned, domain-neutral rubrics:
  - `judge-v2-single` for T1;
  - `judge-v2-multi` for T2/T3, with a reference packet of requirements, evidence excerpts, alternatives,
    definitions and derivations.
  - `judge-final-v3` and `judge-multi-v1` stay frozen with the old study.
  - Primary judge: gpt-oss-120b. Second judge for sensitivity: DeepSeek V4 Flash.
- **Deterministic inner-loop matcher.** Free and instant, for dev.
  - It handles Persian and Arabic digits and letters, scale words, unit conversion, subject/time/version binding,
    bounds and tolerances, list items with aliases, and structured arithmetic.
  - It returns **unknown** when it cannot decide, and its coverage is reported. No method is chosen on the easy subset
    alone.
- **Calibration.** 120 blinded dev answers, adjudicated by the user in a second mode of the game.
  - 96 representative answers: 16 per tier x language, balanced across domains.
  - 24 targeted cases: binding, bound, negation, omission and unsupported addition.
  - Rubric-development cases are kept apart from the final check. Errors are reported by class and by language.
  - The rubric is frozen before test review.
- **LEX speed and accuracy.** Report the **accuracy-latency frontier**, not one weighted score:
  - strict success and requirement coverage;
  - paired latency differences, p50/p95 and timeouts;
  - planner and verifier calls and tokens;
  - retrieval, packing and reader time separately;
  - preprocessing time, index size, memory and amortised cost.

  Every method gets the same reader, evidence budget, output allowance, hardware and scheduling. Cross-question
  writeback and answer caching are off. Indexes are prebuilt, and their cost is reported.

## 10. Systems, contrasts, power
- **Primary systems:** BM25, dense (bge-m3), HYBRID, LEX (new), AGENTIC (new), plus B0 as a control.
- **Primary contrasts:** LEX vs HYBRID and AGENTIC vs HYBRID, Holm-adjusted.
- **Inference:** a paired bootstrap over source blocks, stratified by cell, keeping every question of each sampled
  block.
- **Test answer generations: about 2,100.**

  | Part | Answers |
  |---|---:|
  | 5 systems x 312 questions | 1,560 |
  | B0 | 288 |
  | Document-supplied diagnostics | 216 |
  | Gold-evidence controls | 36 |
- **Power.** 80% power, paired disagreement 0.20:

  | Scope | Detectable difference |
  |---|---:|
  | 288 questions, one contrast | about 7.4 points |
  | Two contrasts | about 8.1 points |
  | One tier (96), two contrasts | about 14.1 points |
  | Clustered in blocks of 12 (design effect 1.55-2.10), overall | about **10-12 points** |

  Simulate the real clustered design from dev results before freezing. No promise to detect 5-point gains, and a
  non-significant result is not treated as equivalence.

## 11. Budget
- **Ledger:** Astra read `reader_budget.json` at **$5.03 spent under the $10 cap**, so **about $4.97 is left**. The
  task statement says $5.4. The plan uses the lower figure.
- **Judging cost:** gpt-oss-120b has cost about $0.00005 per verdict, but v2 reference packets are longer. Measure
  multi-part judging cost on dev first.

| Reservation | USD |
|---|---:|
| Dev judging (milestones only; the free matcher runs in between) | 0.75 |
| Calibration and retries | 0.50 |
| Primary test judging | 1.25 |
| Second-judge sensitivity | 0.80 |
| Contingency | 1.50 |
| **Ceiling** | **4.80** |

- **No OpenRouter cost for authoring:** it is done by Claude subagents. No stronger remote reader is budgeted.
- **GPU:** the local reader is free, but the time is real. 2,100 test answers at 20-45 s each is **about 12-26 serial
  hours**, one job at a time. Measure it on dev.

## 12. Timeline (working days; the gates matter more than the dates)
| Days | Deliverable | Earliest calendar estimate |
|---|---|---|
| 1-2 | Agree design and schema, tier exemplars (2 per tier x domain), game acceptance behaviour | 09-25 - 09-26 |
| 3-6 | Acquire and validate BOOK/LAW fa+en and English report sources; allocate blocks; freeze pools and the exposure ledger (`sources_v2.json`, `source_blocks_v2.json`, `exposure_v2.json`) | 09-27 - 10-01 |
| 7-9 | Author 156 new dev questions plus reserve; convert and repair the legacy set; independent source review | 10-02 - 10-04 |
| 10-13 | The user reviews open dev (108 + 12) in the game; corrections; freeze open dev | 10-05 - 10-08 |
| 14-23 | LEX/AGENTIC and baseline development on dev; judge calibration; sealed test authoring (312 + reserve) in parallel | 10-09 - 10-20 |
| 24 | Provisional freeze; the user reviews dev-check; one check evaluation | 10-21 |
| 25 | Final freeze of methods and protocol (pre-registration) | 10-22 |
| 26-30 | The user reviews the test (312); corrections; reserve replacements | 10-23 - 10-29 |
| 31-32 | Final dataset freeze; one-shot run (about 12-26 GPU-h); judging | 10-30 - 10-31 |
| 33 | Blinded adjudication and analysis | 11-01 |

- **Human review:** about **20-35 hours** of the user's question review, plus answer calibration. Re-estimate after
  the first 30 questions.
- **Source acquisition is the first gate.** If sources or review run late, the schedule moves: weak integration
  questions are never used to fill quotas.

## 13. Where the draft changed, and remaining disagreements
**Adopted from Astra (the draft was wrong or weaker):**
- tier by semantic evidence dependence, not character thresholds;
- pool search primary for T2 as well;
- a balanced 16-per-cell test instead of reports-heavy;
- a separate legacy suite instead of 71 reused questions inside dev;
- no reading of dev-check before a freeze;
- LAW made coherent across languages, with NIST out;
- pools of at least 14 documents per cell;
- derived requirements explicit;
- evidence bundles instead of document sets;
- new domain-neutral judge rubrics;
- B0 on the whole test;
- exposure classes;
- a realistic GPU and human-time budget.

**Claude's remaining reservations (noted, not blocking):**
1. **BOOK-fa feasibility.** Seven disjoint blocks of modern, public-domain Persian non-fiction as clean digital text
   may not exist. Pre-declared fallback, in this order:
   - (a) accept verified transcriptions from Persian Wikisource or Ganjoor prose of the modern period;
   - (b) keep BOOK-fa with 3 test blocks, report it as descriptive, and drop it from the macro-average (declared
     before authoring);
   - (c) never substitute classical texts silently.
2. **Size of dev-open.** 108 new questions (6 per cell) is thin for per-cell diagnosis during a week of heavy
   iteration. Astra's answer is to rely on the legacy 71 for reports and to expand dev-open later if needed.
   - Claude would add 36 more open-dev questions (8 per cell instead of 6) if the user's review speed allows. It is
     not adopted now, to protect review time.
3. **Timeline.** The Claude-side steps (acquisition, authoring) can probably run faster than Astra's working days.
   The user's review sessions are the real pacing item.
4. **Budget figure.** The task says $5.4 and Astra's ledger reading says $4.97. The plan fits under both, with a
   ceiling of $4.80.

## 14. Decisions for the user
1. **Approve the balanced allocation** (test: 288 + 24; dev: 156 new + legacy 71) and the estimated 20-35 review
   hours.
2. **Choose the LAW-en source mix:** EU regulations plus UK Acts (recommended), or EU only.
3. **Fallback order for BOOK-fa** (section 13, item 1).
4. **Red Book:** keep it as a private dev stress source, or drop it.
