# LEX6 evaluation (2026-09-26)

**Verdict: STOP.** LEX6 is not on course for either gate. The idea is sound, and the machinery works as designed. But
retention barely moves (0.733 against HYBRID-intact's 0.729), which leaves it far below HYBRID+rerank (0.806). On the
24-question answer screen it gets 11 right, against 12 for HYBRID and 13 for HYBRID+rerank. None of the extension arms
helps:
- CARDS and SECTIONS lower retention.
- The router never returns within 650 ms.
- Without the deadline, the router mostly runs out of tokens.

Per the stop rule, this closes the LEX6 cycle; the 60-question dev run is not recommended.

**Where things are**
- Spec: `work/analysis/LEX6_DESIGN_FINAL.md`, which is Astra R2, sections 4–7.
- Code: `RAG-Paper-upgrades`, branch `upgrades`, in `ragbench/approaches/lex6/`:
  - `core.py`: document ranking, local retrieval, slots, packing, and the shared `plan()`.
  - `router.py`, `cardsrc.py`, `approach.py`.
  - Tests: `ragbench/tests/test_lex6.py`, 59/59. `test_lex5` still passes 69/69.
- `--approach lex6` is registered, with flags `rank=s2|s5`, `absent_only`, `global_only`, `no_anchor`, `cards`,
  `sections`, `router` and `select` (LEX6+S).
- `lex4 Packager.item()` now exports the provenance of every displayed interval (`meta.provenance`).
- `devrun/analyze_v2.py` lists the LEX6 arms under the unchanged LEX gates. `make_plan.py lex6` writes `plan_lex6.json`
  (not run).
- Harness in this folder:
  - `eval_offline.py` (repairs, rows, tables) and `tables_offline.py`.
  - `answers.py` and `report_answers.py`.
  - `viz_lex6.py`.
  - Step log: `PROGRESS.md`.
- Tables: `OFFLINE_TABLES.md` / `offline_results.json` (step 3) and `ANSWER_TABLES.md` / `answer_results.json`
  (step 4). Per-question rows are in `rows/*.jsonl`.
- Spend: **$0.021** of this task's own ledgers, covering router calls, 24 answers and 21 judge calls. The shared
  budget file stood at 6.456 when checked; that includes the parallel card run's $0.99. The hard stop is 7.60.

## Step 1 – measurement repairs

- **Source identity for every arm.** The LEX5 replay was recomputed with (document, text) retention and with
  **interval retention**: a span counts only if its source interval lies inside a displayed provenance interval.
  - All six configs are unchanged. glm-full-24 stays at 0.858 with 42 questions fully retained; oss20b-full-24 stays
    at 0.838 with 40.
  - The earlier text-only numbers were therefore not inflated.
  - The saved dev evidence under source identity: HYBRID 0.720, HYBRID+rerank 0.806.
  - Every LEX6 arm reports both measures, and they agree everywhere.
- **Bundle-aware document sufficiency.** A question now counts as a hit when the documents of *one* sufficient bundle
  are all in the top k, not the union of all bundles' documents. Only RPT-fa-T3-002 differs between the two
  definitions.
- **docrank fix.** A document with no evidence now gets no rank credit: V is ranked only where V > 0, the centroid only
  where a centroid exists, and ties go by document id. One gold document is now left unranked by V.
  - The shortlist check was repeated after the fix. The 8-document shortlist contains all union documents on 54/54
    questions.
  - The top-3 F0 nominations hold a sufficient bundle's documents on 53/54.
- **Provenance.** Every displayed interval is serialized: in the `lex6` trace, in `rows/*.jsonl` and in the items'
  `meta.provenance`.
- **Feasible oracle pack.** This uses exact package costs and alternative bundles, computed by branch and bound over
  intact packages with exact rendered cost. It is reported with and without the 3,000-character anchor.

## Step 3 – offline matrix (54 answerable dev questions)

| arm | retention | 100% | spans gained / lost vs B0 | exact no-change | verdict |
|---|---:|---:|---|---:|---|
| HYBRID-intact (B0) | 0.729 | 32 | – | 54 | control packer |
| **LEX6 main** (S2+S5, local, package-aware, anchor) | 0.733 | 32 | +6 / −10 | 2 | no real gain; 99 B0 packages displaced |
| S2-only nominations | 0.746 | 33 | +6 / −7 | 21 | best arm; fewer displacements, since the V top documents are already in B0 |
| S5-only nominations | 0.733 | 32 | +6 / −10 | 2 | same as main |
| ABSENT-ONLY | 0.727 | 32 | +5 / −10 | 6 | does not prevent displacement (Astra was right) |
| global candidates only | 0.727 | 32 | +5 / −8 | 19 | fresh local search adds only 1 question of candidate availability |
| no anchor | 0.733 | 32 | +6 / −10 | 2 | the anchor almost never binds under the 3,000 cap |
| CARDS | 0.724 | 32 | +5 / −10 | 2 | generated cards do not earn their $0.99 of indexing |
| SECTIONS | 0.709 | 30 | +6 / −13 | 0 | extra optional slots displace more |
| CARDS+SECTIONS | 0.691 | 29 | +4 / −13 | 0 | worst |
| ROUTER (650 ms) | 0.733 | 32 | = main | 2 | drop rule fires: 54/54 calls hit the deadline |

References (other packers): HYBRID+rerank 0.806 (35 fully retained); LEX5 replay 0.838–0.858. The routing-offline
gold-document oracle reaches 0.805; it is a conditional diagnostic, not a ceiling.

**Chain** (questions out of 54):

| arm | bundle documents reachable | candidates available | feasible (no anchor) | feasible (with anchor) | actual |
|---|---:|---:|---:|---:|---:|
| B0 | 53 | 46 | 46 | 44 | 32 |
| LEX6 main | 53 | 47 | 47 | 45 | 32 |
| gold-document oracle | 54 | 54 | 54 | 47 | – |

- The bottleneck is not documents and not candidates. On **14 questions** B0 could already hold a complete bundle, yet
  packs none.
- Document-aware reservation does not close that gap. It adds one available question, and displacement costs about as
  many spans as it gains.
- Anchor opportunity cost: under the gold-document oracle, **7 questions** are feasible without the anchor but not
  with it.

**By slice.**
- LEX6 main gains on T3 (0.583 → 0.681) and on long documents (0.725 → 0.775).
- It loses on T1 (0.981 → 0.926) and T2 (0.623 → 0.594), where displaced B0 evidence mattered.
- RPT-fa is unchanged at 0.556.
- By language: fa 0.722 → 0.735, en 0.736 → 0.732.

**Router.**
- The trigger fired on 54/54 questions.
- At the declared 650 ms deadline, **54/54 calls timed out**, so the router result was used 0 times.
- A diagnostic with a 5 s deadline (not an arm):
  - 48/54 replies were **truncated** at the 160-token cap, because reasoning uses it up. 6 were valid.
  - Call time was 739 ms at p50 and 1,021 ms at p95.
  - Each call used about 785 tokens (627 prompt + 158 completion).
- With the card order permuted, 50/54 were truncated and 4 were valid.
- There were 0 other invalid outputs. The prompt fits: 683 tokens estimated, 750 by the provider's count, against a
  limit of 800.
- The router is infeasible on latency and on its completion budget, whatever the cards contain.

**LEX6+S.** Implemented (`select` flag) and unit-tested through the approach path, but not run. By the LEX5 pilot, its
selector reads about 7k tokens against about 1.1k of room under the token gate, and the offline matrix allowed router
calls only.

## Step 4 – 24-question answer screen

- Reader: GLM `z-ai/glm-5.3-flash`, pinned to together.
- Judge: v2 strict (`gpt-oss-120b`).
- HYBRID and HYBRID+rerank records and verdicts are **reused** from idea_test and lex5_pilot (same reader, endpoint,
  settings and judge); they were not re-run.
- Extension-arm choice rule, declared before any card result: take the best 1–2 extension arms by offline retention
  that fit the token gate **and** beat LEX6 main. None qualified, so only LEX6 main was answered.

| arm | correct /24 | fa | en | T1 | T2 | T3 | shown retention | p50 s | p95 s | LLM tokens/q | $/q |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID (reused) | 12 | 5 | 7 | 6 | 2 | 4 | 0.728 | 1.62 | 3.69 | 2,169 | 0.00035 |
| HYBRID+rerank (reused) | **13** | 6 | 7 | 6 | 1 | 6 | 0.828 | 2.74 | 5.59 | 2,215 | 0.00037 |
| LEX6 | 11 | 5 | 6 | 5 | 1 | 5 | 0.754 | 1.46 | 2.52 | 2,141 | 0.00036 |

**Gate ratios on the 24.** These use the unchanged thresholds. The latencies are not interleaved: the controls were
measured earlier the same day.

| vs | net correct | tokens | p50 | p95 |
|---|---:|---:|---:|---:|
| HYBRID | −1 | 0.99 | 0.90 | 0.68 |
| HYBRID+rerank | −2 | 0.97 | 0.53 | 0.45 |

**Accuracy path: resources pass, accuracy fails.**
- The resource ratios are all at most 1.5×.
- The net answer count is negative against both controls. The gate needs +3/54 against each.

**Efficiency path: infeasible against HYBRID.**
- LEX6 does all of HYBRID's work and then adds to it, so its p50 cannot be 30% lower than HYBRID's. The measured ratio
  is 0.90.
- Against HR, the efficiency ratios pass, but only because the reranker is slow. The answer loss (−2 on 24) is also
  already at the limit of the ≤ 1/54 allowance.

**Stage latencies (p50).**

| stage | ms |
|---|---:|
| embed | 77 |
| hybrid | 4 |
| LEX6 plan, including packaging | 45 |
| answer | 1,322 |

The deterministic mechanism is cheap. It simply does not improve the evidence.

**Answer changes against HYBRID:**
- **Gains:** RPT-fa-T3-002 (a sufficient bundle's second document reserved) and RPT-en-T2-001.
- **Losses:**
  - RPT-fa-T1-002: a displaced B0 package held the only span, so retention went 1 → 0.
  - RPT-en-T2-002: same evidence as HYBRID; the reader varied.
  - LAW-en-T2-001: a span was lost to displacement.

## Verdicts per mechanism

| mechanism | verdict |
|---|---|
| S2+S5 document ranking | Good for retrieval (F0 R@3 = 53/54 bundle; the shortlist holds a bundle 54/54), but good ranking is not the bottleneck. |
| Nomination + reservation (package-aware) | Neutral to harmful: +6 / −10 spans. Replacing B0's tail costs about as much as it recovers. |
| S5 (centroid) channel | Adds nothing over S2+S5; S2 alone is the best variant (0.746). |
| Fresh local retrieval vs global-only | +1 question of availability, +0.006 retention: a marginal retrieval-recovery effect. |
| ABSENT-ONLY | No safer: −10 spans. Absence of the document is not the right guard. |
| Anchor (3,000) | Almost never binds for LEX6. Costs 7 feasible questions in the gold-document oracle. |
| Generated cards (CARDS) | Does not earn its indexing cost: $0.99 and 12.2M input tokens, for 0.724 against 0.733. |
| SECTIONS | Harmful: 0.709. More optional slots means more displacement. |
| Router | Infeasible: 100% deadline fallbacks at 650 ms; 89% truncated even at 5 s. |
| LEX6+S | Not run: token gate infeasible by construction (LEX5 pilot). |

## Is LEX6 on course for the gates?

**No. STOP.**
- Offline, the best arm (S2-only, 0.746) stays 0.06 below HYBRID+rerank's 0.806.
- The gold-document oracle shows that the headroom lies in within-document selection and packing, not in document
  choice. It separates candidate availability (46 → 54), feasible packing (46 → 54) and the actual pack (32).
- The answer screen is −1 against HYBRID and −2 against HR.
- The +3/54 gate against both controls is out of reach, and the efficiency gate is structurally impossible against
  HYBRID.
- As Astra R2 allowed, failure under the unchanged gates is an acceptable final result. Report LEX6 as a negative
  result: document-aware reservation does not fix T2/T3 packing.

## Visualizer data

`viz_lex6.json` (1.65 MB, no BOM) mirrors `work/viz/lex_viz_data.json`. It has the same top-level keys and the same
per-question keys:
- `docs`, `spans`, `bm25`, `dense`, `cards`: the candidate packages.
- `cmp`: HYBRID and HYBRID+rerank, each with pack, verdict, answer, ms and tokens.

It adds, per question:
- **`lex6.LEX6`**:
  - `docrank`: every pool document with V, C, card BM25, card dense and F0, each with its rank.
  - `shortlist` and `nominated`.
  - `slots`: two alternatives each, with local rank, characters, provenance and decisive spans, plus the top-5 local
    candidates.
  - `b0`, `anchor`, `reservations` and `displaced`.
  - `final`: the pack, with the reason, characters, running total, spans and provenance of each package.
  - `answer`, `verdict`, timings and tokens per stage.
  - `match`: the recomputed pack equals the recorded one on 24/24.
- **`router_offline`**: the router's input prompt, trigger, raw output, status, tokens and ms from the 650 ms arm and
  the 5 s diagnostics.
