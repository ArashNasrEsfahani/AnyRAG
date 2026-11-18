# LEX11-M: F2 membership, HYBRID presentation order, one unchanged answer call (2026-09-26)

**Spec.** Astra R1 (`work/analysis/LEX11_ASTRA_R1.log`, from the last `tokens used`): the LEX11-M pipeline (steps 1-6)
and the declared development plan. Every arm, check and gate interpretation was declared in `PROGRESS.md` before any
LEX11 number was computed.

**Code.** `ragbench/approaches/lex11/` on `upgrades`, commit 738842d:
- `order.py`: H(c), reorder, re-render, invariance check.
- `approach.py`: cells `Lex11M6000`, `Lex11M15500`, `F2At6000`, run with `--approach-class`.
- `ragbench/tests/test_lex11.py`: 5 checks, all pass.

**Harness** (this folder): `audit11.py`, `eval11.py`, `answers11.py`, `report11.py`, `reg11.py`, `viz11.py`.

**Spend.** $0.066 in total: $0.052 for the answers (reader file 10.1925 → 10.2448) and $0.013 for the judge. The
budget was $0.15.

**Caveat.** This is the much-inspected dev set. On 24 questions, one answer moves the count by 1. On 54, one question
moves retention by 0.019.

## 0. Implementation check made before any result: which dense channel is "HYBRID"?

Astra warned that sharing a model name is not proof that two dense channels match. The warning was correct.

- **The lex8 HF bge-m3 dense list does not reproduce HYBRID's order.** RRF(BM25, HF dense) equals the saved HYBRID
  order40 on only **10/54** questions, although the top-40 overlap averages 39.8.
- **HYBRID's own dense channel is the Ollama bge-m3 embedding** used in `HybridIndex.search`.
  RRF(BM25, `dense_ollama`) equals the saved order on **54/54**.

H therefore uses HYBRID's own ranks, from `hx.search(k=80, depth=40)`. That costs one query embedding, about 43 ms at
p50. A diagnostic H built from the HF dense ranks would change the order on 4/54 packs at 6,000 and 15/54 at 15,500.

**Online parity: 72/72 cell records exact.**
- F2 channels, F2 order40, packs and chars all match.
- The HYBRID BM25 and dense lists equal rows71's.
- LEX11-M's pre-order equals the F2 pack.

## 1. Audit of saved artifacts (no reruns, $0; `audit/`)

I read all 93 saved strict-correct verdicts on the 13 calculation/comparison questions that have any. Sources: the
D60 devrun arms, the A24 idea_test screen, lex5_pilot, lex5b, lex6, lex7.1, lex10b and the lex10b control. Each
verdict was checked against the question's requirements. The original scores are kept; `audit/adjudication.jsonl`
adds an adjudication column.

| flag | cells | adjudicated |
|---|---:|---|
| **inverse_ratio_false_positive**: AGENTIC5, D60, RPT-en-T2-003 (Wise) | 1 | **0** |
| bounds_dropped_minor: RPT-fa-T3-002 gives "over 40 million" as 40 (HR D60, HR A24, HR lex5_pilot, LEX10b); HR D60 also states the gap as exactly 13 million | 4 | 1 (kept) |
| magnitude_omitted_minor: BOOK-en-T3-003, direction right but the 4–5 year gap not stated | 3 | 1 (kept) |
| ok | 85 | 1 |

**The Wise case.**
- AGENTIC5 reports 13.3 / 5.4 = 2.463 as the share.
- The correct answer is ≥ 5.4 / 13.3 ≈ 41%, and it is a lower bound.
- The judge accepted the inverted ratio.

Consequences:
- AGENTIC5's D60 score falls from 23/60 to **22/60** after adjudication.
- WIN_LOSS (d)'s "arithmetic over verified operands" example is **void**.

No other direction or operand-role error was found, and no unsupported extra claim changes an answer.

## 2. Offline, 54 answerable dev questions (`TABLES11.md`, `results11.json`, `rows11.jsonl`)

| arm | retention (complete) @6,000 | b/w vs HYBRID | retention (complete) @15,500 | b/w vs HYBRID |
|---|---:|---:|---:|---:|
| HYBRID | 0.7200 (32) | – | 0.8471 (39) | – |
| F2 | 0.7792 (33) | 7/1 | 0.8824 (42) | 7/2 |
| **LEX11-M** | **0.7792 (33)** | 7/1 | **0.8824 (42)** | 7/2 |
| F2-2ch = MaxMean(BM25, dense) | 0.7206 (32) | 3/1 | 0.8361 (39) | 4/3 |
| F2-intact (lex4 packages) | 0.7693 (32) | 7/2 | 0.8731 (40) | 5/2 |

**Invariance holds by construction and by assertion.**
- On every question and at both budgets, membership, displayed text (including the clipped item) and chars are
  identical to F2's.
- Retention equals F2's 0.7792 / 0.8824 exactly.
- Four sanity values reproduce the saved figures exactly: F2 at both budgets, S-c 0.8540 and F2-intact@6,000 0.7693.

**Where the rescued evidence moves.**
- No pack item outside both HYBRID channel top-40s holds a decisive span. Sparse-only items average 0.04 per pack at
  6,000 and 0.26 at 15,500. The F2 gain is therefore re-ranking inside HYBRID's own candidate lists.
- I measured rescued evidence as decisive items in the F2 pack that hold a span missing from HYBRID's same-budget
  pack.
  - @6,000: 8 items, **5 moved later**, mean position 4.1 → 5.4.
  - @15,500: 10 items, **7 moved later**, mean position 9.8 → 14.4. Examples: RPT-fa-T2-001 moves 4 → 16 of 17;
    BOOK-fa-T2-002 moves 3 → 13.
- The first decisive item moves later in 10 questions (earlier in 5) at 6,000, and later in 13 (earlier in 5) at
  15,500.
- Astra's central risk is real: the passages F2 rescued are exactly the ones H pushes down.

**Attribution ablation: sparse earns its place.** MaxMean over BM25 + dense only lands at HYBRID's level (0.7206 and
0.8361). The three-channel F2 gain (+0.059 and +0.046) comes entirely from the learned-sparse channel, not from escaping
RRF's consensus preference.

**Selector pruning replay** (saved S-c replies; 47 parsed, 7 deadline fallbacks; no new call):

| variant | retention (complete) | chars (parsed q) | q losing spans vs S-c | complete lost | calc/comparison q losing | negation/exception q losing |
|---|---:|---:|---:|---:|---:|---:|
| S-c saved (core + full fill) | 0.8540 (38) | 5,766 | – | – | – | – |
| core only | 0.7778 (33) | 2,283 | 9 | 5 | 5 | 3 |
| core + 20% fill | 0.8148 (36) | 3,230 | 6 | 2 | 3 | 3 |

- Pruning cuts 44–60% of the characters but loses operands and exceptions.
- Core + 20% loses spans on BOOK-en-T1-001 and RPT-en-T2-002, two of LEX5B's four precision wins over HR.
- The filler currently rescues what the selector omits. As Astra required, **no new selector screen is justified.**

**Representation audit.** Raw chunks beat intact packages at the same F2 order: 0.7792 against 0.7693, and 0.8824
against 0.8731, with fewer items. Part of S-c's apparent gain over F2 is therefore not selection, and the package
change must not be credited to the selector.

## 3. Answer screen: idea_test 24 (`eval/ANSWER_TABLES11.md`, `eval/answer_results11.json`)

**Setup.**
- New cells, interleaved: F2@6,000, LEX11-M@6,000 and LEX11-M@15,500.
- All controls are reused records and verdicts, not re-run and not interleaved.
- Reader: GLM glm-5.3-flash with the providers.json pins. Judge: gpt-oss-120b, strict (v2).

| cell | correct /24 | fa / en | shown ret. (100%) | p50 s | p95 s | tokens/q |
|---|---:|---:|---|---:|---:|---:|
| HYBRID@6,000 | 12 | 5 / 7 | 0.728 (14) | 1.62 | 3.69 | 2,169 |
| HR@6,000 | 13 | 6 / 7 | 0.828 (17) | 2.74 | 5.59 | 2,215 |
| LEX7.1@6,000 | 13 | 6 / 7 | 0.781 (15) | 2.30 | 4.38 | 3,152 |
| LEX5B@6,000 | 17 | 6 / 11 | 0.852 (17) | 4.38 | 7.74 | 4,315 |
| **F2@6,000 (new)** | **12** | 6 / 6 | 0.811 (16) | 2.16 | 5.08 | 2,221 |
| **LEX11-M@6,000 (new)** | **12** | 6 / 6 | 0.811 (16) | 2.34 | 5.99 | 2,214 |
| HYBRID@15,500 | 19 | 9 / 10 | 0.878 (18) | 2.68 | 11.26* | 5,152 |
| HR@15,500 | 16 | 7 / 9 | 0.863 (18) | 3.61 | 5.73 | 5,229 |
| LEX7.1@15,500 | 14 | 7 / 7 | 0.877 (19) | 2.90 | 6.39 | 6,097 |
| LEX5B@15,500 | 15 | 8 / 7 | 0.898 (19) | 4.99 | 8.72 | 6,731 |
| F2@15,500 (LEX10b) | 15 | 7 / 8 | 0.898 (19) | 1.41 | 2.80 | 5,150 |
| **LEX11-M@15,500 (new)** | **15** | 7 / 8 | 0.898 (19) | 2.34 | 4.38 | 5,148 |

\* HYBRID@15,500's p95 is contaminated by Ollama warm-up (budget_matrix note).

**Record retention for LEX11-M.** `common.shown_texts` assumes the pack is a prefix of the offered list. LEX11-M's pack
is reordered, and its clipped item can sit mid-pack, so the record-based figure was wrong (0.877). LEX11-M's shown
retention is therefore taken from the offline rows. Those packs are identical, with 72/72 parity.

**Gates, same budget only.**
- **Tokens pass under both interpretations.**
  - LEX11-M@15,500 uses 5,148 tokens/q: ≤ the historical absolute cap of 5,422, and ≤ 12,880 (2.5x the same-budget
    HYBRID).
  - Both 6,000 cells use about 2,214–2,221 tokens/q.
- **Accuracy fails everywhere** (+3 needed).

  | cell | vs HYBRID | vs HR | vs LEX7.1 | vs LEX5B | vs F2 |
  |---|---:|---:|---:|---:|---:|
  | LEX11-M@6,000 | 0 | −1 | −1 | −5 | 0 |
  | LEX11-M@15,500 | **−4** | −1 | +1 | 0 | 0 |

- **Latency** is measured across sessions against the controls.
  - LEX11-M@6,000 fails p95 against HYBRID (1.62x).
  - LEX11-M@15,500 fails against LEX10b (p50 1.66x, p95 1.56x). This is mostly answer-call variance: the retrieval stage
    is 70 ms at p50, including 43 ms for HYBRID's ranks.

**Order is not the missing piece.**

| comparison | same evidence? | result |
|---|---|---|
| LEX11-M vs F2 @6,000 | identical, interleaved | 1:1 |
| LEX11-M vs F2 @15,500 | identical | 1:1 |
| LEX11-M vs HYBRID @15,500 | same decisive-span retention on all four differences | 0:4 |

- Swapping order alone flips answers both ways, 1:1 at each budget. That is the reader-variance floor.
- At 15,500, HYBRID still wins four questions (LAW-fa-T2-002, RPT-en-T2-001, BOOK-fa-T2-001, BOOK-en-T1-001), all with
  the same decisive-span retention. Three of these four are the questions F2 lost in LEX10b. HYBRID's own presentation
  order did not recover them.
- **What does differ between LEX11-M and HYBRID@15,500 is membership:** the packs share most chunks (12–16 of 15–16 in LEX10b's comparison); the rest is non-decisive
  context. HYBRID converts full packs at 94% (17/18). LEX11-M converts 74% (14/19) and F2 79% (15/19).
- The ordering hypothesis from WIN_LOSS clue 6 is **not supported**.

## 4. Verdicts and admission

1. **LEX11-M is not admitted to the final test.**
   - It adds nothing over F2 at either budget: 12 = 12 and 15 = 15.
   - At 15,500 it trails the same-budget HYBRID by 4. No +3 gate is met against any control.
   - It passes the token gate under both interpretations.
2. **F2@6,000 (new cell, 12/24)** confirms that F2's offline retention gain (+0.059 over HYBRID) does not convert
   into answers at 6,000: 12 against HYBRID's 12, net 0.
3. **What survives as a finding:**
   - learned sparse is the entire F2 gain (ablation);
   - raw chunks beat intact packages;
   - selector pruning loses operands and exceptions, so the filler is load-bearing;
   - the flagship AGENTIC arithmetic win was a judge false positive;
   - at 15,500 the F2-versus-HYBRID gap is not about order.
4. **Nothing from LEX11 goes to the untouched test.** The only positive lever remains the budget itself (HYBRID
   12 → 19).

## Artifacts

| file | contents |
|---|---|
| `PROGRESS.md` | the declarations and the log |
| `audit/` | the audit: `cells.md`, `decisions.json`, `adjudication.jsonl` |
| `rows11.jsonl`, `TABLES11.md`, `results11.json` | offline traces and tables: packs, pre/post order, H, decisive items, F2-2ch order, selector replay |
| `eval/runs`, `eval/judge`, `eval/parity.json`, `eval/ANSWER_TABLES11.md`, `eval/answer_results11.json` | the answer screen |
| `viz_lex11.json` | per question: offline arms, LEX11-M order, answers and verdicts for all 12 cells, audit flags |

**Registry.** 11 rows with `budget_chars`, all tagged `gate_tokens: 2.5x (user amendment 2026-09-26)`:
- 8 offline rows. The ablation, representation and replay rows are marked DIAGNOSTIC.
- 3 answer cells, which carry both token-cap interpretations.
