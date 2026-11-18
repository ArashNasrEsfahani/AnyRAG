# Checking the recommendation on your own data

The study's recommendations rest on 90 test questions, one primary reader and one judge, and most are
post hoc. Before you commit, compare the recommended setup (A) against one alternative (B) on your
own questions. This recipe is the same paired design the paper used, scaled down to what a team can do
in a day or two.

## 1. Pick exactly one comparison

Change one thing between A and B, so a difference has one cause. Typical pairs:

- A = hybrid + cross-encoder reranker, B = plain hybrid (is the reranker worth its latency?)
- A = hybrid + LLM selection call, B = plain hybrid at a larger budget (the no-GPU choice)
- A = reranker, B = iterative search agent (is the agent worth ~4x the tokens?)
- A = 1,000-character chunks, B = your current chunk size

Keep everything else fixed: documents, chunking (unless that is the comparison), reader model and its
settings, answer prompt, evidence budget.

## 2. Build a question set

- **Size.** As many as you can afford, at least 50. Be realistic about what you can detect: with 90
  questions the paper could only detect a difference of about 11 answers; a 5-answer (~5.6 point)
  difference needs about 400 to 460 questions. Small sets can show a large win or rule out a disaster,
  not settle a close call.
- **Mix.** Use real user questions if you have logs, in the proportion users ask them. Label each as
  T1 (one passage), T2 (several parts of one document) or T3 (several documents); the setups differ most
  on T2 and T3.
- **Reference answers.** Write a short reference answer and the points a correct answer must contain.
  If you can, also note which passages contain the evidence (this lets you do step 6).
- Add a few questions your documents cannot answer, and check that both setups abstain.

## 3. Remove questions the model already knows

Run the reader **closed-book** (no retrieved evidence) on every question. Mark the ones it answers
correctly and set them aside (`exclude = 1` below), or report results with and without them. In the
study the reader answered 26 of 90 from memory, mostly well-known English texts, and an agent's apparent
lead came from those questions. Private company reports were never answered from memory (0 of 30).

## 4. Run both setups and score blind

- Run A and B on the same questions, once each, logging tokens and latency.
- Score every answer correct / incorrect with the same rules for both: an LLM judge with a strict rubric,
  or a person who does not know which setup produced the answer.
- If people review only some answers, review **the same questions for both setups**. In the study a
  review that covered one pipeline more thoroughly produced a significant lead that vanished once all
  pipelines were reviewed equally.

## 5. Paired comparison with a sign test

Write one row per question:

```
question_id,a,b,exclude
q001,1,0,0
q002,1,1,0
q003,0,1,1
```

`a` and `b` are 1 if that setup answered correctly. Then run:

```
python scripts/sign_test.py results.csv
```

The script counts the discordant pairs (b = questions only A got right, w = questions only B got
right), reports the net difference b - w, and the exact two-sided sign test p-value
(p = 2 x P(X <= min(b, w)) for X ~ Binomial(b + w, 1/2), capped at 1). Questions both got right or both got
wrong carry no information about which is better and are ignored.

Reading the result:

- p < .05 and the net favours A: keep A, on these questions.
- p >= .05: the two are **not distinguished** on your data. Choose on cost, latency and simplicity, and
  do not report the winner as better.
- Look at the T2/T3 subset separately; that is where the study saw differences. Treat subgroup results as
  descriptive.

## 6. Optional: check what reaches the reader

If you annotated evidence passages, count for each setup the questions whose final pack contains a
complete evidence bundle. In the study this explained most wrong answers: the reader answered 83 of 90
when given only the gold evidence, so most losses happened in retrieval and packing. A setup that puts
more complete bundles into the pack is usually the better one even when accuracy differences are noisy.

## 7. Optional: estimate the noise

Run the reader a second time on the identical packs of one setup. The change in its score is your
run-to-run noise (about ±2 answers of 90 in the study). A difference smaller than that is not a
difference.

## Checklist

- [ ] One change between A and B
- [ ] Same reader, prompt and budget
- [ ] Closed-book run; memory-answerable questions excluded or reported separately
- [ ] Blind, symmetric scoring
- [ ] Sign test on discordant pairs; tokens and median latency reported beside accuracy
