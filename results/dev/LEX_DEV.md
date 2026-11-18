# LEX on the dev files, 2026-09-22

LEX is the user's design: report cards, then LLM keywords, then the reports' own phrases (the model picks them),
then code-compiled regex over the whole reports, then a window around each hit, then the shared answer. It makes
3 model calls. Code: `ragbench/approaches/lex/`. Research: `REGEX_LLM_2026.md`.

- Runs: `work/runs/lex_dev` (pilot.jsonl, 64 records per arm) and `work/runs/lex_dev_tl` (pilot_timeline.jsonl, 8
  records).
- Scoring: key facts, and text scoring for timelines.
- Analysis script: `work/runs/lex_dev/lex_analysis.py`.
- The test set was not touched.
- The stage was stopped by the system for low memory at 12:30. Every LEX arm, B2 and B3 were complete. OURS L0
  reached 33 of 72 records and is left out of the conclusions below.

## Wide dev (64 records per arm)

| arm | Acc % | lookup | locate | multi-sec | cross-year | Ev. recall % | calls | vs LEX better / worse |
|---|---|---|---|---|---|---|---|---|
| **LEX** | **75.0** | 78.6 | 54.5 | 80.0 | 60.0 | 63.1 | 2.98 | – |
| LEX + llm_regex (the model writes the regexes) | 23.4 | 25.0 | 9.1 | 13.3 | 0.0 | 11.6 | 3.00 | 1 / 34 |
| LEX + free_keywords (no call 2) | 56.2 | 67.9 | 54.5 | 33.3 | 20.0 | 46.1 | 2.00 | 3 / 15 |
| LEX + fixed_window (±250 raw characters) | 65.6 | 75.0 | 63.6 | 46.7 | 40.0 | 66.0 | 2.98 | 3 / 9 |
| LEX + window=250 | 76.6 | 85.7 | 63.6 | 80.0 | 20.0 | 64.6 | 2.98 | 3 / 2 |
| LEX + window=1000 | 75.0 | 82.1 | 63.6 | 66.7 | 60.0 | 59.6 | 2.98 | 2 / 2 |
| LEX + open_corpus (the cards route) | 65.6 | 71.4 | 63.6 | 46.7 | 60.0 | 51.2 | 3.00 | 2 / 8 |
| LEX + open_corpus + title_routing | 57.8 | 67.9 | 36.4 | 40.0 | 60.0 | 45.2 | 3.00 | 1 / 12 |
| B2 (BM25) | 78.1 | 92.9 | 63.6 | 66.7 | 40.0 | 74.2 | 1.00 | 8 / 6 |
| B3 (bge-m3) | 71.9 | 85.7 | 72.7 | 46.7 | 40.0 | 68.2 | 1.00 | 8 / 10 |
| AGENTIC3 | 84.4 | 92.9 | 72.7 | 73.3 | 80.0 | 78.5 | 3.14 | 11 / 5 |

Median latency: LEX (the first arm, so its calls were fresh) 4.1 s; AGENTIC3 12.1 s. The later LEX arms replayed their
keyword call from the shared cache, so their latencies are not comparable.

## Dev timelines (8 questions)

| arm | years covered | precision | fully correct | flag | false trends |
|---|---|---|---|---|---|
| AGENTIC3 | 0.708 | 0.802 | 0.375 | 0.75 | 0.00 |
| **LEX** | **0.583** | 0.833 | 0.125 | 0.50 | 0.25 |
| LEX window=250 | 0.510 | 0.929 | 0.250 | 0.50 | 0.00 |
| LEX fixed window | 0.510 | 0.625 | 0.125 | 0.25 | 0.25 |
| LEX free_keywords | 0.375 | 0.708 | 0 | 0.25 | 0 |
| LEX llm_regex | 0.250 | 0.722 | 0.125 | 0 | 0 |
| B3 | 0.562 | 0.662 | 0 | 0.25 | 0.50 |
| B2 | 0.177 | 0.500 | 0 | 0 | 0.50 |

## Findings

1. **A 4B model cannot write the regexes; code must.**
   - All 274 of its regexes compiled, so syntax was never the problem.
   - In 53 of 72 records none of its regexes matched anything.
   - Accuracy was 23.4%, against 75.0% when code compiles the model's chosen words. It lost 34 questions and won 1.
   - This matches the literature (QueStER: an untrained 4B model's queries score about half of BM25). It is also the
     free-form vs code-compiled comparison no paper has published.
2. **Grounding in the reports' own words is the biggest single gain: +18.8 points** (56.2 → 75.0), from call 2
   picking phrases that actually occur in the reports. It matters most where wording differs: cross-year 20 → 60,
   multi-section 33 → 80.
3. **Structure-aware windows beat fixed character windows by 9.4 points** (75.0 vs 65.6).
   - Size matters little: 250, 500 and 1000 characters give 76.6, 75.0 and 75.0.
   - A 250-character structure window has the best timeline precision (0.929) and no false trends.
   - Fixed windows cut tables and sentences.
4. **Card routing finds the reports.**
   - With the report list withheld, cards put the question's reports among the 3 routed in 72 of 72 records, from
     pools of 3 to 16 reports.
   - Titles alone did so in 64 of 72. The keywords on the card are what make routing reliable.
   - Accuracy still fell 9.4 points. The cause is selection, not routing: every single-report question was routed to
     3 reports, and the round robin then gave the 2 wrong reports as many windows as the right one (8 of 55 got
     worse).
   - Fix, not yet run: share windows across reports only when the question spans several years; otherwise rank
     every window by score.
5. **Against the others:**
   - Accuracy: LEX 75.0 is level with BM25 (78.1) and vectors (71.9); the paired differences are inside the noise.
     It is below AGENTIC3 (84.4) at a third of AGENTIC3's latency.
   - LEX is ahead on multi-section (80.0 vs 66.7 and 46.7) and cross-year (60 vs 40). It is behind on single lookups
     (78.6 vs 92.9 for BM25).
   - On timelines it covers more years than both baselines (0.583 vs 0.177 and 0.562) but fewer than AGENTIC3 (0.708).
     It also makes false trend claims (0.25), which AGENTIC3's verification prevents.

## lex-v4: the defects found by the second opinion, fixed (2026-09-22; NOT yet run on dev)

Source: `LEX_ASTRA_OPINION.md`. Every claim was checked against the code before fixing.

1. **Table windows.** Matching rows now come first: phrase matches before relaxed ones, then by weight. Each keeps
   its neighbours, the header is always included, and rows print in document order. Before, rows were taken in
   document order up to the cap, so a late matching row could be displaced. A matching row longer than the room left
   (text cells run to 680 characters) is shown as a slice around the match instead of being dropped; the old code
   dropped it.
2. **Credit from the emitted text.** A window is scored on the terms that match the text it SHOWS (tables row by row
   with the header). Before, it was credited with every term its units matched, including rows it left out. A hit
   whose term falls outside its own window is dropped.
3. **Measure-aware relaxed match.** The AND match drops count words (تعداد, total), but keeps measure-kind words
   (share, rate, growth, average): «سهم واردات» no longer matches «واردات» alone. A term made only of measure words gets
   no relaxed match. A relaxed match weighs `w_relaxed` = 0.6 of a phrase match.
4. **Word boundaries.** Every compiled pattern must start a word. A last word shorter than 4 letters must also end
   one, allowing ها/های/ی/ان/ات (and s/es in English). «کار» no longer matches inside «کاربر» or «کسب‌وکار».
   Model-written regexes are left as written.
5. **Selection by characters.** Windows are chosen against the exact characters they cost in the shared block (the
   header rendered as `contracts.item_header` renders it), so the packer never cuts or drops one. A window that more
   than half repeats a chosen one is skipped (`overlap_max`). The old `covered` set only skipped anchors.
6. **Reports.** One window per report first, only when the question needs several: a given multi-report scope, or
   routed reports for a question naming at least 2 years or at least 2 names. Then everything goes by score, with the
   routed card rank as a modest additive prior (`w_card_prior`), not a multiplier. This replaces the unconditional
   round robin that gave wrong routed reports equal windows in open-corpus mode.
7. **Provenance.** A table window records the rows it shows (`meta.table_rows`, `table_rows_total`).

**Metric caveat:** the shared evidence metric counts unit ids, so a table id counts as "found" whether or not its
answer row survived. `ev_recall` therefore flatters every method's table items, and LEX's table windows most. Row
provenance is recorded so this can be audited.

**Not changed, by decision:**
- BM25 fusion, per-report phrase allocation and the missing-year pass are additions for the next dev round.
- Window size by answer kind is deferred: window 250 took cross-year from 60 to 20.
- The definition note was dropped: a different name does not prove a definition changed.
- There is no 4th call.

## LEX v5 + LEX2 against every earlier method, dev files, 2026-09-22

Runs: `work/runs/lex_dev5(_tl)` (LEX v5 arms, LEX2) and `work/runs/lex_dev(_tl)` (the earlier methods, LEX v1).
Table and script: `work/runs/lex_dev5/compare/COMPARE.md`, `work/runs/lex_dev5/compare_all.py`. Key-fact scoring; text
timeline scoring. Latency is the wall clock with cached replays charged at their original call time. The stage was
stopped for low memory one step from the end, so B4 has no dev timeline records. The test set was not touched.

| Method | Wide dev acc % | Timeline years | fully right | false trends | calls | median s |
|---|---|---|---|---|---|---|
| B1 gold evidence (ceiling) | 98.4 | 0.84 | 0.25 | 0.50 | 1.0 | 1.3 |
| **LEX2 (all upgrades)** | **84.4** | 0.61 | 0.38 | 0.25 | 3.0 | **4.9** |
| **LEX2, open corpus** | **84.4** | 0.58 | 0.25 | 0.50 | 3.0 | 6.5 |
| AGENTIC3 | 84.4 | 0.71 | 0.38 | 0.00 | 3.1 | 12.1 |
| AGENTIC4 | 84.4 | 0.68 | 0.38 | 0.00 | 3.1 | 12.1 |
| LEX v5 + bm25 | 82.8 | 0.57 | 0.38 | 0.50 | 3.0 | 4.5 |
| AGENTIC2 | 81.2 | 0.74 | 0.38 | 0.00 | 3.1 | 11.2 |
| B2 BM25 | 78.1 | 0.18 | 0.00 | 0.50 | 1.0 | 1.3 |
| LEX v5 + report_vocab | 75.0 | 0.65 | 0.12 | 0.50 | 3.0 | 4.4 |
| LEX v1 | 75.0 | 0.58 | 0.12 | 0.25 | 3.0 | 4.1 |
| LEX v5 | 73.4 | 0.65 | 0.12 | 0.50 | 3.0 | 5.2 |
| LEX v5 + year_pass | 73.4 | **0.73** | 0.12 | 0.50 | 3.0 | 4.4 |
| B3 vectors | 71.9 | 0.56 | 0.00 | 0.50 | 1.0 | 1.4 |
| LEX v5 lite (no labels) | 67.2 | 0.50 | 0.00 | 0.25 | 3.0 | 4.3 |
| OURS L0 | 64.1 | 0.31 | 0.00 | 0.00 | 2.0 | 3.2 |
| B5 grep agent | 57.8 | 0.11 | 0.00 | 0.00 | 6.2 | 15.6 |
| LEX v5, no phrase pick | 56.2 | 0.47 | 0.00 | 0.50 | 2.0 | 3.5 |
| OURS L1 | 54.7 | 0.23 | 0.00 | 0.00 | 2.0 | 4.6 |
| B4 tree navigation | 51.6 | – | – | – | 4.1 | 4.6 |
| AGENTIC v1 | 46.9 | 0.34 | 0.12 | 0.25 | 4.2 | 14.3 |
| LEX v5, model writes regex | 20.3 | 0.21 | 0.12 | 0.00 | 3.0 | 4.1 |
| B0 no retrieval | 7.8 | 0.00 | 0.00 | 0.00 | 1.0 | 0.3 |

Paired (64 records): LEX2 beats LEX v5 on 9, loses 2. LEX2 against AGENTIC3 is 5 / 5. LEX2 against BM25 is 6 / 2.
BM25 fusion alone accounts for almost all of it: 9 better / 3 worse against LEX v5.

Findings:
1. **LEX2 matches the best agent at less than half its time**: 84.4 with 3 calls and 4.9 s, against AGENTIC3's 84.4
   with 12.1 s. AGENTIC3 still claims no false trend (0.00 against 0.25) because it verifies each value against its
   quote, and it covers more years (0.71 against 0.61).
2. **The BM25 safety net is the upgrade that pays**: +9.4 points alone (73.4 -> 82.8), as the second opinion
   predicted; it fills what the regex misses on plain lookups.
3. **The missing-year pass does what it was built for and nothing else**: timeline years 0.65 -> 0.73, the best of any
   LEX arm, with wide accuracy unchanged. In LEX2 its gain is lost (0.61): the BM25 windows take the room the year
   pass needs, an interaction to fix before trusting LEX2 on series.
4. **Per-report phrase lists**: +1.6, inside the noise.
5. **Layer 1 works**: with the report list withheld, LEX2 still scores 84.4 (open corpus), so the cards find the
   reports without help.
6. **The label layer earns its 47 s**: the lite variant, which needs no facts or labels, drops to 67.2.
7. **v5 against v1** is 4 better / 5 worse on the wide set (the bug fixes are not visible there) but better on the
   timelines: years 0.58 -> 0.65.
8. The model writing its own regexes stays catastrophic at 20.3.
