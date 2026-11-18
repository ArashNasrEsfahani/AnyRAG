# S1 human-review packets: how to load them and merge the results

Two blinded packets of GLM-5.3-flash answers at 6,000 characters. The pages never show which pipeline wrote an answer. Byte-identical answers to the same question appear once, and the key maps each item back to every cell it stands for. Rebuild both packets with `python build_packets.py`. It uses fixed seeds, makes no model calls and gives the same files every time.

| packet | what it covers | questions | cells | items shown |
|---|---|---:|---:|---:|
| `packets/A_concordant24/` | a seeded random sample (seed 20261110) of 24 of the 60 machine-concordant answerable test questions × HYBRID, HR, LEX5B, SETR. Questions are in random order and answers are shuffled within each question | 24 | 96 | 82 |
| `packets/B_second_reviewer/` | the 120 cells already reviewed by the first reviewer (30 questions × 4 pipelines), in one random order across questions, for an independent native Persian/English reviewer | 30 | 120 | 116 |

Each packet holds:
- `review_<packet>.html`: the earlier stand-alone page (the same template as `work/datasets_v2/human_check/review_*.html`). Each item now also has a **Requirements** card (optional requirements are marked) and a **Gold evidence quotes** card.
- `game/<A|B>_review.jsonl` and `game/<A|B>_review.answers.jsonl`: the question-v2 rows and the blinded answers (`qid`, `answer_id`, `answer`, no method field) for the answer-check mode of `review_game/server.py`.
- `key_<packet>.json`: maps each item and answer_id to its question id and pipeline(s). **The reviewer must not open it.** Give the reviewer only the `.html` file, or only the `game/` folder.

## Reviewing

**Option 1: stand-alone page (same workflow as before).** Double-click `review_A_concordant24.html` (or `review_B_second_reviewer.html`). Press 1 for Correct, 2 for Wrong, 3 for Unsure. You can add a note, and ←/→ move between items. Progress saves in the browser. When you finish, click **Export** and save the text as `A_export.json` (or `B_export.json`). The format is unchanged: `{"set", "seconds", "results": [{"item", "verdict", "note"}]}`. Packet B runs in its own browser storage (`s1_second_reviewer_v1`), so reviewing it on the same machine does not overwrite packet A or the earlier reviews. Still, give the second reviewer a clean browser profile, or a different machine.

**Option 2: review game.** Run this from the repository root:

    .venv\Scripts\python.exe review_game\server.py --port 8770 --dir work\posthoc_2026-11\S1_evaluator\packets\A_concordant24\game

(use `...\B_second_reviewer\game` for packet B). Open http://localhost:8770/, pick `A_review.jsonl`, and switch to **Answer check**. Each question shows its requirements as answer points and its evidence in context, with one tab per blinded answer. Verdicts go to `game/A_review.answercheck.jsonl`. When an answer is saved more than once, the latest save wins.

Strict rule, the same for both options: an answer is correct only if every **required** requirement matches. Half correct counts as wrong, and so does an extra unsupported claim that changes the answer. A note containing "half correct" makes the cell wrong whatever button was pressed. In the game, partial and unsupported both count as wrong. Unsure or blank cells are dropped and counted.

## Merging and estimating

    .venv\Scripts\python.exe work\posthoc_2026-11\S1_evaluator\human_estimate.py --a A_export.json [--b B_export.json] [--judge 1|2]

Either file can be a page export or a game `answercheck.jsonl`. The script writes `human_results/human_estimate_judge<k>.{json,md}` and reports:
- per pipeline, the number of human-correct answers out of the 90 answerable questions. This is a census on the 30 reviewed questions (first reviewer, symmetric merge) plus an estimate on the 60 concordant questions, computed three ways:
  - **naive**: machine verdicts, the way the paper reports them now;
  - **PPI**: the prediction-powered difference estimator `sum_C M + (N/n) sum_S (H - M)`, with a finite-population variance;
  - **RG**: the Rogan-Gladen sensitivity/specificity correction, in the spirit of Lee et al. 2025, arXiv 2511.21140, computed per pipeline and pooled over the four pipelines.
  Each comes with a 95% interval, and the output also counts how often the judge was stricter or laxer than the human on the sample;
- PPI estimates of the paired nets (LEX5B vs HYBRID, LEX5B vs HR, SETR vs the others);
- with `--b`: the second reviewer against the first, per pipeline and overall (agreement, Cohen's kappa, and who marks more answers correct), plus each judge's kappa against the second reviewer and how often each judge is stricter or laxer.

`--judge 2` uses this stream's second judge (`judge2/`) as the predictor in place of the pinned judge.

Tests: `python test_human_estimate.py` builds synthetic exports in `synthetic/`, one in each format, and checks:
- the parsers and the half-correct rule;
- the PPI identity (human = judge gives the naive count with SE 0);
- PPI 95% coverage, by Monte Carlo over 4,000 synthetic populations;
- hand-computed kappa values.
The current run prints ALL PASS, with PPI coverage 0.928. Synthetic data only: no real human verdicts were made up.
