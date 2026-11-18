# Human check: dev answers where methods disagree

`review_dev.html` is a blinded review of the 20 dev60 questions where, at 6,000 characters, LEX5B's strict verdict differs from HYBRID's or HR's. Each item shows the question (Persian has an English gloss), the reference answer and one candidate answer. It never shows which method wrote the answer. There are 46 items in all: each distinct answer from LEX5B and from the disagreeing control or controls, with identical texts merged. They are shuffled with seed 7.

## How to use

1. Open `review_dev.html` directly in a browser by double-clicking it. You do not need a server.
2. Judge each candidate: **1** Correct, **2** Wrong, **3** Unsure. The page then moves to the next item. You can add a short note (press Enter to move on). Use ← and → to go back and forward.
3. Your answers save in this browser as you go, so you can close the page and come back later.
4. When you finish, click **Export**, copy all the text in the box and paste it into the chat.

## Files

- `review_dev.html`: the page.
- `key_dev.json`: maps each item to its question id, the method(s) that produced it and the judge's strict verdict. Do not open it before you review.
- `build_dev.py` and `review_dev.template.html`: regenerate the page and the key from `dev60/dev60_results.json` and `questions/*_dev.jsonl`.

# Test review files (GLM reader, 6,000 characters only)

The test review covers the primary reader at 6,000 characters only, for the 30 test questions on which the machine
judge separated any compared pair. No other reader or budget was reviewed by a human.

- `key_test.json` + `human_test_export.json`: the first session (2026-10-05). It has 77 items covering 78 cells, at
  about 12 s per item. It reviewed only the pipelines of each discordant pair, including all 30 LEX5B cells.
- `key_test_sym.json` + `human_test_sym_export.json`: the symmetric completion (2026-10-06). It has 39 items covering
  40 cells, plus 2 cells inherited from byte-identical answers, at about 36 s per item. It holds control answers only,
  and was reviewed after the asymmetry was known.
- Both sessions together give the symmetric merge (`code/test_run/human_merge_sym.py` ->
  `results/test/TEST_RESULTS_HUMAN_SYM.md`). "Symmetric" describes coverage, not matched session conditions. The
  first session's own merge is superseded; see `results/test/superseded_asymmetric_review/`.
- The review pages (`review_*.html`) embed the answers with their questions and are available on request.
