# Why T2 and T3 fail on the dev run: retrieval or reader?

Dev set: 18 within-document (T2) and 18 cross-document (T3) questions. The dev run's local reader is gemma4:e4b. The judge is openai/gpt-oss-120b with the v2 rubric. "Strict" means every required requirement is correct.
Scripts and data are in this folder: `step1_retrieval.py` (writes `step1_rows.jsonl` and `step1_summary.json`), `step2_analyze.py` (`step2_summary.json`), `step3_patterns.py` (`step3_patterns.json`), `arms/diag_arms.py`, `run_step2.sh`, `runs/`, `judge/`.

## Short answer

**Both retrieval and the reader fail, and they fail about equally often. Retrieval matters more for T3; the reader matters more for T2.** The local gemma reader is the weakest link in the chain.

| setup (T2+T3 dev, same judge) | T2 /18 | T3 /18 | abstentions T2/T3 | mean recall T2/T3 |
|---|---:|---:|---|---|
| local gemma4:e4b + HYBRID+rerank evidence (dev run) | 2 | 6 | 4 / 7 | 0.37 / 0.41 |
| gpt-oss-20b + the **same** HYBRID+rerank evidence (replayed) | 5 | 6 (1 unjudged T2) | 2 / 3 | 0.56 / 0.50 |
| gpt-oss-20b + **gold evidence only** (B1) | **10** | **11** | 1 / 0 | 0.82 / 0.83 |
| z-ai/glm-5.3-flash + **gold evidence only** (B1) | **12** (1 unjudged) | **17** | 0 / 0 | 0.85 / 1.00 |
| local reader, other dev arms: HYBRID / LEX3 / AGENTIC5 | 2 / 4 / 3 | 5 / 5 / 4 | | |

32 of the 36 questions are answered strictly correctly by at least one reader given gold evidence, and 18 by both. The questions can therefore be answered from their gold units. The dev scores are low because of the pipeline, not because the questions are broken; possible question defects are listed at the end.

Decomposition on HYBRID+rerank (T2+T3, 36 questions; one oss-20b cell is still unjudged):
- Changing only the reader (gemma → gpt-oss-20b, identical evidence) moves 8 → 11 correct, **+3**. On T3 alone it gains nothing (6 → 6).
- Changing only the evidence (HR → gold, same gpt-oss-20b reader) moves 11 → 21, **+10**.
- A stronger reader on gold (glm) reaches 29 of 35 judged cells. The remaining gap between gold-with-oss and gold-with-glm (+8) is reader strength again.

By the HR arm's own gold coverage:

| HR coverage class | questions | local+HR | oss-20b+HR | oss-20b gold | glm gold |
|---|---:|---:|---:|---:|---:|
| all required gold units in the evidence | 21 | 8 | 10 | 13 | 17 |
| some of them | 9 | 0 | 0 | 4 | 7 |
| none of them | 6 | 0 | 1 | 4 | 5 |

- **With full coverage**, gemma still fails 13 of 21 questions. This is a reader failure, and a better reader on clean gold evidence fixes most of it.
- **With partial or no coverage**, no reader succeeds on the retrieved evidence (1 of 15). This is a retrieval failure, and gold evidence fixes 8–12 of the 15.

## 1. Retrieval diagnosis (no API calls; 4 arms × 36 questions = 144 cells, local-reader verdicts)

A requirement counts as covered when every gold unit of some evidence bundle that supports it lies inside the record's final `evidence_units`. That is the cut-aware set of units the reader actually saw.

| arm | tier | correct | wrong, full coverage (reader) | wrong, partial | wrong, no coverage (retrieval) | requirement coverage | oracle docs with gold in evidence | mean items / chars |
|---|---|---:|---:|---:|---:|---:|---:|---|
| HYBRID | T2 | 2 | 9 | 1 | 6 | 0.76 | 0.78 | 6.4 / ≈6000 |
| HYBRID+rerank | T2 | 2 | 10 | 3 | 3 | 0.84 | 0.94 | 6.6 |
| LEX3 | T2 | 4 | 4 | 5 | 5 | 0.64 | 0.78 | 6.0 |
| AGENTIC5 | T2 | 3 | 5 | 4 | 6 | 0.58 | 0.83 | 5.7 |
| HYBRID | T3 | 5 | 4 | 5 | 4 | 0.66 | 0.68 | 6.8 |
| HYBRID+rerank | T3 | 6 | 3 | 6 | 3 | 0.62 | 0.65 | 6.7 |
| LEX3 | T3 | 5 | 4 | 7 | 2 | 0.66 | 0.68 | 6.2 |
| AGENTIC5 | T3 | 4 | 2 | 8 | 4 | 0.57 | 0.62 | 7.2 |
| **all 4 arms** | **T2** | 11/72 | **28** | 13 | 20 | 0.71 | 0.83 | |
| **all 4 arms** | **T3** | 20/72 | 13 | **26** | 13 | 0.63 | 0.66 | |

- **T2:** 28 of 61 wrong cells (46%) had every required gold unit in front of the reader. 33 of 61 lacked some or all of it.
- **T3:** only 13 of 52 wrong cells (25%) had full coverage. In 26 of the 52 multi-document T3 cells that were wrong, **one of the two oracle documents contributed no gold unit**; in 11 cells neither did.
- **Where the gold sits:** when gold is present it is usually the first item (T2: 53 of 144 gold positions at rank 1; T3: 58 of 124). So ordering inside the pack is not the problem.
- **Missing requirements (T2 53, T3 77 across the 4 arms):** about half were in the arm's candidate list but ranked below the 6000-character pack (T2 26, T3 41). The rest never appeared at all (T2 25, T3 35).
- **Arm differences:** LEX3 packs its whole candidate list, so it has no "retrieved but not packed" misses; its misses are all never-found. AGENTIC5 has the lowest coverage of the four arms (0.58 / 0.57).

## 2. The gold-evidence reader check

Setup:
- **B1 GoldEvidence** is the question's gold units in author order. They are rendered by the shared `pack_evidence` into the same evidence block and answered with the same v2 answer prompt, format and token budget. The readers were gpt-oss-20b and glm-5.3-flash via OpenRouter.
- **ReplayHR** takes the dev HYBRID+rerank record's offered items, re-packs them and answers with gpt-oss-20b. The re-packed evidence units are identical to the dev evidence in 36 of 36 cases. It makes no retrieval, embedding or reranker calls and uses no GPU.
- **Judging:** judge_multi with the same judge, questions file and rubric. RPT-fa-T2-003 stayed unjudged for glm-gold and oss-replay after three retries, because the judge's reply omitted units R4–R7. It is counted as not correct.

Accuracy with gold evidence:
- gpt-oss-20b: T2 10/18, T3 11/18.
- glm-5.3-flash: T2 12/18, T3 17/18.
- The local reader's dev accuracy on retrieved evidence was T2 2–4/18 and T3 4–6/18, depending on the arm.

## 3. Question family × failure cause

The cells columns count 4 arms × questions with the local reader. The last four columns count questions strictly correct.

| family | questions | cells | correct | wrong, full coverage (reader) | wrong, partial | wrong, no coverage (retrieval) | local+HR | oss-20b+HR | oss-20b gold | glm gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T2 bounded synthesis | 5 | 20 | 3 | 9 | 4 | 4 | 1 | 2 | 2 | 4 |
| T2 comparison / derivation | 5 | 20 | 1 | 9 | 0 | 10 | 0 | 1 | 3 | 3 (1 unjudged) |
| T2 definition bridge | 2 | 8 | 0 | 4 | 2 | 2 | 0 | 2 | 1 | 2 |
| T2 rule + exception | 6 | 24 | 7 | 6 | 7 | 4 | 1 | 0 | 4 | 3 |
| T3 bounded aggregation | 2 | 8 | 0 | 4 | 2 | 2 | 0 | 0 | 1 | 1 |
| T3 cross-doc bridge | 3 | 12 | 0 | 0 | **10** | 2 | 0 | 0 | 1 | 3 |
| T3 cross-subject | 8 | 32 | 10 | 6 | 7 | 9 | 2 | 2 | 4 | 8 |
| T3 cross-time / version | 5 | 20 | 10 | 3 | 7 | 0 | 4 | 4 | 5 | 5 |

- **T3 cross-doc bridge** is purely a retrieval problem. The second hop, the bridged document, is never evidenced (10 of 12 cells partial), yet glm solves 3 of 3 given gold.
- **T2 comparison/derivation** and **T2 bounded synthesis** are mostly reader problems: the gold is present but the model does not compose it.

## 4. Failure patterns (example ids; /H, /HR, /L3, /A5 = arm)

### Retrieval side

1. **The second document or second hop is never retrieved (T3).** In 26 T3 cells one oracle document has no gold unit in the evidence, and in 11 cells none does. The query names the bridge entity only implicitly, so single-shot retrieval finds only the document the question is worded around.
   Examples: BOOK-fa-T3-002 (all 4 arms, R2/R3 never retrieved), BOOK-en-T3-001 (all arms miss R1, 3 of 4 abstain), RPT-fa-T3-003/H,L3, LAW-en-T3-002 (no gold in any arm), LAW-en-T3-003/H,HR.
2. **Relevant, but ranked below the 6000-character pack.** 26 of 53 missing T2 requirements and 41 of 77 missing T3 requirements were in the candidate list, just not packed. Long chunks (about 1000 characters each, 6–7 items) crowd out the second fact.
   Examples: RPT-fa-T2-003/H (all 3 requirements in the candidates, none packed; HR packs them at ranks 6–7), LAW-en-T2-001/HR (9 gold units in the candidates, none packed; abstains), BOOK-en-T3-002/HR,A5 (6 gold candidates cut, abstains), LAW-en-T3-003/H,HR, LAW-fa-T2-002 (R4's unit sits next to a packed one but is cut in H/HR).
3. **The evidence is fragmented.** The chunk holding the operand is present, but the qualifier or bridge sentence that binds it (another article, a footnote, a table caption) is not. The reader then attaches the value to the wrong year or subject.
   Examples: RPT-fa-T3-002/H,L3 (only 1 of 3 requirements covered; R2/R3 answered wrong or contradicted), LAW-fa-T3-002/H,A5, BOOK-en-T3-003/L3,A5.

### Reader side (the gold was in the evidence)

4. **Omits one required part of a multi-part answer.** 18 T2 and 11 T3 covered requirements were judged "missing". A multi-part question gets a one-part answer, and gpt-oss-20b does the same even on gold evidence.
   Examples: RPT-en-T2-002 (R3, the €440–770m range, missing in every arm and in oss-gold), BOOK-fa-T2-001 (R2, the sword, missing for all readers including glm-gold), LAW-en-T2-002/H (R2, the 30-day transition), BOOK-en-T3-002/H,L3 (3 of 4 parts), RPT-en-T3-002/HR (the end-2022 operand).
5. **The derivation step is not done or is wrong.** 16 T2 and 6 T3 covered derived or conclusion requirements were missing or wrong: the model restates the operands but does not compute, or computes from the wrong pair.
   Examples: RPT-fa-T2-001 (glm-gold gives 21.8M and 86% but not about 18.7M), RPT-fa-T2-002/HR,A5 (total installs = 470M instead of 700+470), BOOK-fa-T2-003/HR (20 years instead of 15), RPT-en-T3-002/HR (fall computed from 2023 instead of 2022), BOOK-fa-T3-001/oss-gold (7 years instead of 9).
6. **Wrong binding or reversed rule/exception.** The value is right but the condition is inverted.
   Examples: LAW-fa-T3-003/H,L3 (says a 60-year-old civil servant cannot retire), LAW-fa-T2-003 (housing co-op exemption inverted in H, A5 and both gold readers), BOOK-fa-T2-003 (Tughril "did not tolerate" instead of "kept" the rulers), RPT-en-T2-001 (R2, the points-based four levels, "wrong" in 3 arms).
7. **False abstention despite full gold coverage.** 6 T2 and 4 T3 cells abstain with every required unit present, plus 10 T3 cells with partial coverage. The small reader answers NOT_FOUND when the answer has to be assembled rather than quoted.
   Examples: BOOK-fa-T3-001 (all 4 arms; glm-gold answers "9 years" correctly), RPT-en-T2-003/H,HR,L3 (oss-20b on the same HR evidence answers correctly), RPT-fa-T2-003/HR, BOOK-en-T2-001/H.

Judge note: 4 gold-evidence cells score 0 with every required unit correct. The cause is one extra claim judged unsupported or wrong-binding (e.g. glm on LAW-en-T2-003, where citing Art. 19(4) instead of 19(5) was labelled wrong binding). The strict metric punishes extra detail.

Possible question defects to review, failing for every reader including on gold evidence:
- **LAW-fa-T2-003:** "تعاونی مسکن کارمندان". All readers decide it is not the "مسکن" co-op of Art. 110, so the wording may be ambiguous.
- **BOOK-fa-T2-001 R2:** the sword is a "qualifier" unit that no reader treats as a feature of the lion.
- **RPT-fa-T2-001:** the gold is a KPI tile plus a sentence, and the product is never stated in the text.

## 5. What this implies for LEX / AGENTIC design

1. **Evaluate the methods with a competent reader.** With gemma4:e4b, any retrieval gain is hidden because the reader converts full-coverage evidence into a correct answer only 8 times out of 21 (HR). The AGENTIC gate "+4 net T2+T3" is unreachable when the reader, not retrieval, is the bottleneck for T2. Screen LEX/AGENTIC on dev with an OpenRouter reader (glm-5.3-flash on gold: 29/35), or report gold-evidence (B1) upper bounds next to every arm.
2. **Make retrieval requirement-aware for T3.** Decompose the question into its parts (subject × aspect × constraint, domain-neutral) and retrieve per part. Guarantee at least one evidence slot per part or document before filling by score: 26 of 52 wrong T3 cells lack one oracle document. For bridge questions, add a second hop that queries with the entity or term found in hop 1. This is where LEX/AGENTIC should beat HYBRID; so far it does not (AGENTIC5's T3 coverage of 0.57 is below HYBRID's 0.66).
3. **Pack more, smaller units.** Half the missing requirements were already among the candidates. Packing unit or sentence windows around the hit, instead of about 1000-character chunks, plus a coverage check ("is every part evidenced? if not, retrieve again"), would recover them without new search. LEX3's all-or-nothing candidate list shows the opposite failure: 0 cut but more never-found.
4. **Add a verification or synthesis step for the reader.** Use a checklist answer: list the question's parts, fill each from the evidence with a citation, then compute the derived value explicitly. That addresses omissions (pattern 4), missed arithmetic (5) and false NOT_FOUND (7). The abstain rule should require that no part is supported, not merely that the answer is not stated verbatim. This belongs in the shared answer contract, or as an AGENTIC verification call, so every arm keeps a fair comparison.
5. **Keep the evidence clean.** oss-20b scores 10/21 on full-coverage HR evidence but 13/21 on the same questions' gold units, so distractor chunks cost about 3 questions. Precision-oriented packing (verified units first) helps the reader too.

## Cost

- Readers: gpt-oss-20b gold $0.0019, glm-5.3-flash gold $0.0050, gpt-oss-20b replay $0.0028.
- Judge (gpt-oss-120b, 3 × 36 plus retries): the rest.
- `reader_budget.json` went from $5.1256 before to $5.1584 after, so this task cost **$0.033**. The study total stays under the $20 cap.
