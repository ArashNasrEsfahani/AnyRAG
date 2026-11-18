# SETR: a zero-shot SetR baseline on the 60 dev questions (2026-10-01)

**What SETR is.** SETR follows SetR (Lee, Jo, Park and Lee, ACL 2025, arXiv 2507.06838, "Shifting from Ranking to Set
Selection for Retrieval Augmented Generation"; code at github.com/LGAI-Research/SetR). It is a literature baseline. The
model makes one chain-of-thought call. In that call it lists the question's information requirements, matches
passages to each requirement, and chooses the **set** of passages that covers them.

**Where things are.**
- Code: `RAG-Paper-upgrades`, branch `upgrades`, commit `1cdb056`.
  - `ragbench/approaches/setr/approach.py`, run with `--approach setr`.
  - Tests: `ragbench/tests/test_setr.py` (5/5 pass; `test_hybrid` 87/87).
- Harness: `run_setr.py` (`run`, `judge`, `report`) and `report_setr.py`.
- Results: `results_setr.json` (summary and per-question rows), `runs/`, `judge/`.
- One row was added to the registry, `RESULTS_REGISTRY.jsonl`.

**Caveat.** This is the much-inspected dev set. The p-values are descriptive only.

## 1. Fairness and design (declared in `PROGRESS.md` before any SETR call)

| item | SETR | matches |
|---|---|---|
| candidates | HYBRID's own fused list (BM25 + bge-m3 dense, RRF k=60, depth 40, the same `hx.search`), top 40 | HYBRID / HR first stage |
| shown to selector | the top **20**, full chunks; a fixed 1,500-char cap per passage, which **never bound** (0 passages cut) | SetR's top-20 default |
| selector model | glm-5.3-flash, the same Together pin as the LEX5B selector and the reader; 1 attempt; reasoning effort low | LEX5B selector |
| deadline | **8 s** total, 2,000-token completion cap (CoT needs more than LEX5B's 3 s and 600 tokens) | declared up front |
| fallback | deadline, call error, no marker or no valid id: plain HYBRID order | - |
| pack | selected chunks first (in the model's order), then the rest of the top 40 in HYBRID order, through HYBRID's own offer and chunk packer: intact chunks, 6,000 chars | HYBRID packer |
| answer | the shared GLM answer call | all arms |
| judge | gpt-oss-120b strict v2, **pinned to deepinfra/bf16** as in `PROTOCOL.md` (3 identical attempts; malformed replies go to adjudication) | the test protocol |

**How retention was measured (step B).**
- It was computed exactly as HYBRID's 0.720 and HR's 0.806 were: lex4 decisive-span retention over the texts the reader
  actually saw, rebuilt from the run records.
- So step B and step C come from the same single SETR run. No second selector pass was made; this was declared
  beforehand.

## 2. Faithfulness to the paper

**Faithful:**
- The system message and the "selection IRI" user prompt from `generate_data.py`, verbatim. This includes "Number of
  passages is unlimited" and the `### Final Selection: [] []` format.
- Passages rendered `[i] text`, with newlines collapsed.
- The top 20 candidates.
- A single call combining requirement identification and set selection.

**Not faithful:**
- **Zero-shot.** The paper fine-tunes Llama-3.1-8B-Instruct (5 epochs, about 40K examples distilled from GPT-4o). We
  prompt glm-5.3-flash directly. This makes SETR the zero-shot, prompted form of SetR's teacher prompt, not the released
  SetR model.
- **Packing.** The paper passes only the selected set to the generator. We pack the selected set first and then fill
  up to 6,000 chars with the remaining HYBRID candidates, the same budget as every other arm.
  - The selected set is small (mean 2.9 chunks; 24 questions with 2 and 16 with 3), so the fill matters.
  - A selected-set-only arm was not run.
- **Parser.** It reads the last `Final Selection` marker and tolerates bold markup.
- **Passage text.** We show chunk text only, with no title, as their code does. Our chunks come from multi-document
  pools, not Wikipedia.

## 3. Step A: smoke test

- Three dev questions outside the idea_test 24: BOOK-fa-T1-003, LAW-en-T2-002 and BOOK-en-T3-002.
- All 3 parsed, with finish `stop` and served by Together. Selection took 2.2–4.0 s.
- A first attempt failed because Ollama was down. It made 0 calls, cost $0, and its directory was cleared. Those cells
  were later reused in step C.

## 4. Step B: selection retention (54 answerable questions)

| arm | span retention | complete | b/w vs SETR (per question span) |
|---|---:|---:|---|
| HYBRID | 0.720 | 32 | SETR better 15 / worse 0 |
| HR | 0.806 | 35 | SETR 10 / 2 |
| LEX5B (offline S-c 0.854 / 38; dev60 shown 0.842 / 36) | 0.842 | 36 | SETR 7 / 3 |
| **SETR** | **0.868** | **42** (fa 23 / en 19) | - |

## 5. Step C: answers (60 dev questions at 6,000 chars; controls reused from dev60)

| arm | /60 | /54 answerable | AB /6 | fa /27 | en /27 | tokens/q | p50 s | p95 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID | 35 | 30 | 5 | 15 | 15 | 2,169 | 1.69 | 3.89 |
| HR | 38 | 32 | 6 | 17 | 15 | 2,192 | 2.66 | 7.37 |
| LEX5B | 43 | 37 | 6 | 17 | 20 | 4,397 | 4.04 | 8.75 |
| **SETR** | **46 (46–48)** | **40 (40–42)** | 6 | 20 | 20 | **7,748** | 4.17 | 6.69 |

**Judging.** Two SETR cells are `needs_adjudication`: BOOK-en-T3-002 and RPT-fa-T2-003. They are counted 0 above, and
the bounds are shown in brackets.

| SETR vs (54 answerable) | net | better / worse | sign p |
|---|---:|---:|---:|
| HYBRID (30) | **+10** | 12 / 2 | 0.013 |
| HR (32) | **+8** | 9 / 1 | 0.021 |
| LEX5B (37) | **+3** | 5 / 2 | 0.45 |

- **Against LEX5B:**
  - SETR is better on BOOK-en-T2-001, BOOK-fa-T1-001, BOOK-fa-T2-001, LAW-fa-T3-002 and RPT-fa-T3-002.
  - It is worse on BOOK-en-T1-001 and RPT-fa-T2-003. RPT-fa-T2-003 is one of the two adjudication cells.
- **The selector:**
  - 0 fallbacks in 60 calls.
  - Select time p50 2.47 s, p95 3.90 s, max 5.6 s.
  - 5,565 selector tokens per question.
  - By language: fa 9,757 and en 5,740 total tokens per question.
- **Spend:**
  - $0.077 for the reader and selector, from the SETR ledgers.
  - $0.0075 for the judge.
  - The shared reader file went 10.4957 → 10.6233 over the window, which includes the parallel agent.

## 6. Honest reading

**1. SETR is the strongest 6,000-char arm on dev.**
- It leads on both retention (0.868 vs LEX5B's 0.842 and 0.854) and answers (40 vs 37 out of 54).
- The +3 over LEX5B is not significant (5/2).

**2. It pays for that in tokens.**
- At 7,748 tokens per question it uses **3.57x** HYBRID. That is above the 2.5x resource gate LEX5B was held to.
- It is also 1.76x LEX5B.
- It is not a size-matched comparison. SETR's selector sees the full top 20 (about 18k chars); LEX5B sees budgeted
  excerpts of 2,600 tokens.
- Latency is comparable to LEX5B: p50 4.2 s against 4.0 s, and a lower p95.
- **This is the price of showing whole passages.** The fair contrast is not "set selection beats LEX5B". It is "an LLM
  selector that reads full candidates beats one that reads excerpts, at about 1.8x the tokens".

**3. The judges differ.**
- The control verdicts are dev60's unpinned judge; SETR's are pinned (deepinfra/bf16).
- PROTOCOL.md exception 2 shows that the unpinned judge can flip a verdict.

**4. LEX5B's 3 s deadline would have failed SETR often.**
- SETR's p50 select time alone is 2.5 s.
- The 8 s deadline was declared, so SETR had no fallbacks at all.

## 7. Should SETR join the frozen test?

**Recommendation: yes, as a declared literature-baseline arm (SETR_6000), but only through a formal re-freeze before
the first test answer.**

The test has not started: there is no `test_run/` directory. Leaving out a published baseline that beats the candidate
on dev, after seeing that it does, would be selective reporting.

What the re-freeze requires:
- The code commit moves from `b31cd4c` to `1cdb056` or later.
  - The `--test-protocol` gate now refuses the current HEAD until `MANIFEST.json` is regenerated.
  - The untracked `ragbench/approaches/lex5b/fast.py`, which another agent created and I did not touch, also counts as
    an uncommitted change under `ragbench/`.
- A dated `PROTOCOL.md` section with the reason for the change.
- SETR's index pointer directory prebuilt and hashed.
- A seventh arm in the rotation, with k = i mod 7.
- About $0.20 more spend on the test.

How SETR should be framed:
- As a baseline, **not** as a gate control. LEX5B's +5/90 gates stay against HYBRID and HR, as frozen.
- LEX5B vs SETR is reported as a secondary paired comparison, together with SETR's token ratio. SETR at 3.57x would
  fail the 2.5x resource gate that LEX5B must pass.
