# S17 The seven gold-evidence misses (2026-11-18)

No new runs, no API calls. Everything is read from saved files.

Sources:
- reader answers: `S2a_packs/answers.jsonl` (arm GOLD); the exact pack the reader saw: `S2a_packs/packs/GOLD/<qid>.json`
- first judge (gpt-oss-120b): `S2a_packs/judge/GOLD/verdicts_openai__gpt-oss-120b.jsonl` (83 of 90 strictly correct)
- second judge (Mistral Large): `S8_factorial/judge2/GOLD/verdicts_mistralai__mistral-large-2512.jsonl` (88 resolved, 82 strictly correct, 2 unresolved: S8_factorial/tables.md "GOLD 82..84")
- questions, requirements, gold bundles: `datasets_v2/test_freeze/questions/test_v2run_all.jsonl`

The class is my reading of the saved question, pack, answer and both judge records. It is not an independent human review.
The GOLD pack is the first annotated bundle, one item per evidence quote in bundle order, with its heading path.

## 1. Table

| Question id | Tier | Lang | Expected answer (gist) | Given answer (gist) | Judge 1 | Judge 2 | Class | Why |
|---|---|---|---|---|---|---|---|---|
| RPT-en-T3-002 | T3 | en | ArcelorMittal 57.9 Mt (2024), Tata Steel Group 29.94 Mt (FY2023-24), ratio about 1.9 | Tata 29.94 MT correct; ArcelorMittal "NOT_FOUND"; no ratio | wrong (R1 wrong, R3 missing) | unresolved (reply not parseable) | Evidence bundle not sufficient | The pack holds the ArcelorMittal figure only as the bare row "Total 43.5 14.4 57.9" and, as a separate item, a caption "Crude steel production by process in 2024 (in million tonnes)". Nothing in the pack says that 57.9 is the company total. The reader abstained on that value, as its prompt tells it to. |
| RPT-en-T3-001 | T3 | en | Mowi 501,530 t GWT salmon vs Lerøy 147,701 GWT salmon (171,228 incl. trout), ratio about 3.4 | Mowi 501 530 t correct; Lerøy "NOT_FOUND", "comparison cannot be made" | wrong (R2, R3, optional R4 missing) | wrong (same units) | Evidence bundle not sufficient | Lerøy's figure appears as the table row "Salmon Trout \| 147 701" and a separate header item "Total harvest volume in GWT ... \| 2024 \| 2023". The row label joins salmon and trout, and the bundle omits the total row (171 228) that would resolve it. The figure is not tied to salmon, so the reader declined. |
| BOOK-en-T1-005 | T1 | en | Malthus's Island: starts at about 7 million; after fifty years population 28 million, subsistence for 21 million | Starts at 7 millions; gives 28 / 21 millions for "after twenty-five years"; "after fifty years: NOT_FOUND" | wrong (R2, R3 missing) | wrong (R2 wrong, R3 missing) | Evidence bundle not sufficient | The second bundle item reads "In the next twenty-five years the population would be twenty-eight millions ...". The pack lacks the preceding passage that makes this the second period, so nothing in it places 28 and 21 at fifty years. The reader reported the period that the text states. |
| BOOK-fa-T3-002 | T3 | fa | Vosough al-Dowleh was first vice-president of the first Majlis; he concluded the 1919 agreement with Britain; the British claim says 130,000 lira was split among him, Sarem al-Dowleh and Nosrat al-Dowleh | Same three facts, with a bracketed source note per bullet | wrong (R3 graded wrong_value, "money unit mismatches GBP"; the answer states 130,000 lira as the text does) | wrong (strict 0 although all three requirements graded correct; an `unsupported` mark on the restated source sentence) | Judge strictness | All three required requirements are present and match the reference. Judge 1 faulted a unit that the source text itself gives as lira. Judge 2 graded every requirement correct and failed the answer on a claim that only restates the cited sentence. A human reviewer would likely accept it. |
| LAW-fa-T1-002 | T1 | fa | Five religious-minority seats: Zoroastrians 1, Jews 1, Assyrian and Chaldean Christians together 1, Armenian Christians south and north 1 each (Art. 64) | "In total 5": lists Zoroastrians 1, Jews 1, Assyrian and Chaldean 1, Armenian south 1, Armenian north 1 | wrong (R2, Jews, graded missing; the Jews line is in the answer and was marked `extra` with no requirement id) | correct | Judge strictness | The answer states every seat and the total of five. Judge 1 did not link the Jews line to requirement R2. Judge 2 accepts the answer. A human reviewer would accept it. |
| LAW-en-T1-004 | T1 | en | Consumer Rights Act 2015, s.28(3): "without undue delay" and in any event not more than 30 days after the contract date | "Not more than 30 days after the day on which the contract is entered into" | wrong (R1 "without undue delay" missing) | unresolved (reply not parseable) | Reader error | The pack holds both limbs, s.28(3)(a) and (b). The answer gives only the 30-day limb and leaves out the required first limb. What it says is true and the omission is a defect against the reference. Whether a human reviewer would accept a one-limb answer is not shown by the saved data. |
| LAW-en-T3-004 | T3 | en | Police force: "higher maximum amount" (DPA 2018 s.157(2)(a)), GBP 17,500,000 for a body that is not an undertaking (s.157(5)(b)); EU company: up to EUR 20,000,000 or 4% of worldwide turnover, whichever is higher (GDPR Art. 83(5)(a)) | Police: "higher maximum amount ... the specific figure is not stated"; EU company figure correct | wrong (R2 missing; R1, R3 correct) | wrong (R2 missing) | Evidence bundle not sufficient | The 17.5 million figure is in the pack only as the fragment "(b) in any other case, £17,500,000." placed before the line "(5) The higher maximum amount is-". The pack has no text for the undertaking limb or for the body that is not an undertaking, so the figure is not tied to the higher maximum amount. The reader said so and gave the correct EU figure. |

## 2. Counts

| Class | Questions | Ids |
|---|---|---|
| Evidence bundle not sufficient | 4 | RPT-en-T3-002, RPT-en-T3-001, BOOK-en-T1-005, LAW-en-T3-004 |
| Judge strictness (an answer a human would likely accept) | 2 | BOOK-fa-T3-002, LAW-fa-T1-002 |
| Reader error | 1 | LAW-en-T1-004 |
| Question or annotation problem | 0 as a separate class | the four bundle cases are annotation choices (quotes cut out of tables or clauses) and are counted under the bundle class |

Second judge on the seven: correct on 1 (LAW-fa-T1-002), wrong on 4 (RPT-en-T3-001, BOOK-en-T1-005, BOOK-fa-T3-002, LAW-en-T3-004), unresolved on 2 (RPT-en-T3-002, LAW-en-T1-004).
The second judge also fails two answers that the first accepted (RPT-en-T2-005 and BOOK-fa-T2-004), which gives 82 correct of 88 resolved (S8_factorial/tables.md, "GOLD 82..84").

Tiers of the seven: T1 3, T2 0, T3 4. Languages: English 5, Persian 2 (matches S2a_packs/RESULTS.md sec. 2.4: T1 27/30, T2 30/30, T3 26/30; en 40/45, fa 43/45).

## 3. What this shows and what it does not

- Of the seven, two answers would very likely pass with a different judge or a human reader, and four packs do not contain enough text to tie the key value to its subject.
  These four packs are the annotated bundles as quoted, in bundle order.
- Only one miss is a clean reader error.
- If the two judge-strictness cases were counted correct, the gold reference would be 85 of 90. This is a count from my reading and is not a new score.
- The classification covers the primary reader only, uses one reader run, and rests on one person's (my) reading of the saved records. No second annotator checked it.
