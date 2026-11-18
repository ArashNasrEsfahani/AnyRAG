# LEX6 offline matrix (54 answerable dev questions; no answer calls)

## 1. Measurement repairs

### 1a. LEX5 replay recomputed with source identity (step1_select.py passed bare strings)

| config | text-only (as reported) | 100% | source identity (doc, text) | 100% | interval (span inside a displayed provenance interval) | 100% |
|---|---:|---:|---:|---:|---:|---:|
| glm-full-24 | 0.858 | 42 | 0.858 | 42 | 0.858 | 42 |
| glm-full-40 | 0.855 | 40 | 0.855 | 40 | 0.855 | 40 |
| glm-head-24 | 0.782 | 33 | 0.782 | 33 | 0.782 | 33 |
| glm-snippet-24 | 0.837 | 39 | 0.837 | 39 | 0.837 | 39 |
| oss20b-full-24 | 0.838 | 40 | 0.838 | 40 | 0.838 | 40 |
| oss20b-snippet-24 | 0.804 | 36 | 0.804 | 36 | 0.804 | 36 |

No LEX5 replay question changed: text-only matches never came from the wrong document or from outside a displayed interval. Saved dev evidence (answer-time packer, source identity): HYBRID 0.720 (32 full), HYBRID+rerank 0.806 (35 full).

### 1b. Document ranking with zero credit for missing evidence (doc-id ties); union vs bundle-aware

Hit = every gold document of the union (resp. of at least one sufficient evidence bundle) is in the top k. Only V2-dev-RPT-fa-T3-002 has alternative bundles whose documents differ from the union.

| signal | R@1 union / bundle | R@2 | R@3 | R@5 | R@8 | gold docs left unranked |
|---|---|---|---|---|---|---:|
| S2 chunk-max V | 32 / 32 | 50 / 50 | 50 / 50 | 51 / 52 | 52 / 53 | 1 |
| S5 centroid | 31 / 31 | 49 / 49 | 51 / 52 | 52 / 53 | 53 / 54 | 0 |
| F0 = RRF(V, C) | 33 / 33 | 50 / 50 | 52 / 53 | 52 / 53 | 54 / 54 | 0 |

- Shortlist (<= 8: top-2 V, top-2 C, rest by F0): all union docs 54/54, a bundle's docs 54/54 (the check repeated after the zero-credit fix, as Astra required).
- Nominations (top 3 by F0): all union docs 52/54, a bundle's docs 53/54.
- Old S2 (pool-order ties, every document ranked) R@2 union: 50/54 (unchanged by the fix at R@2).

## 2. Mechanism matrix (Astra R2 section 7)

Retention = decisive-span retention of the 6,000-char pack with source identity; interval = the span's source interval lies inside a displayed provenance interval. Gained/lost = spans present vs absent relative to B0 (HYBRID-intact), summed over questions. Displaced = B0 packages no longer represented.

| arm | what | retention | interval | 100% /54 | idea 24 | spans gained / lost | questions better / worse | displaced pkgs (chars) | exact no-change /54 | plan ms p50 / p95 |
|---|---|---:|---:|---:|---:|---|---|---|---:|---|
| HYBRID-intact | B0 (control packer) | 0.729 | 0.729 | 32 | 0.748 | +0 / -0 | 0 / 0 | 0 (0) | 54 | 34 / 64 |
| LEX6 | main: S2+S5, local, package-aware, anchor | 0.733 | 0.733 | 32 | 0.754 | +6 / -10 | 5 / 3 | 99 (81599) | 2 | 11 / 22 |
| LEX6+rank=s2 | S2-only nominations | 0.746 | 0.746 | 33 | 0.803 | +6 / -7 | 4 / 2 | 59 (47712) | 21 | 35 / 82 |
| LEX6+rank=s5 | S5-only nominations | 0.733 | 0.733 | 32 | 0.754 | +6 / -10 | 5 / 3 | 104 (85296) | 2 | 16 / 29 |
| LEX6+absent_only | ABSENT-ONLY | 0.727 | 0.727 | 32 | 0.754 | +5 / -10 | 4 / 3 | 89 (73874) | 6 | 10 / 23 |
| LEX6+global_only | global candidates only (no fresh local search) | 0.727 | 0.727 | 32 | 0.754 | +5 / -8 | 4 / 3 | 64 (52305) | 19 | 5 / 8 |
| LEX6+no_anchor | packing ablation: no anchor | 0.733 | 0.733 | 32 | 0.754 | +6 / -10 | 5 / 3 | 101 (83699) | 2 | 11 / 28 |
| LEX6+cards | CARDS (generated-card fusion) | 0.724 | 0.724 | 32 | 0.754 | +5 / -10 | 4 / 3 | 99 (80517) | 2 | 33 / 67 |
| LEX6+sections | SECTIONS | 0.709 | 0.709 | 30 | 0.721 | +6 / -13 | 5 / 6 | 130 (106257) | 0 | 41 / 107 |
| LEX6+cards,sections | CARDS + SECTIONS | 0.691 | 0.691 | 29 | 0.700 | +4 / -13 | 3 / 6 | 131 (106924) | 0 | 39 / 113 |
| LEX6+router | ROUTER | 0.733 | 0.733 | 32 | 0.754 | +6 / -10 | 5 / 3 | 99 (81599) | 2 | 746 / 800 |
| LEX6+router,diag5s | DIAGNOSTIC: router with a 5 s deadline | 0.733 | 0.733 | 32 | 0.754 | +6 / -10 | 5 / 3 | 90 (73859) | 7 | 842 / 1163 |
| LEX6+router,diag5s,permute | DIAGNOSTIC: 5 s router, card order permuted | 0.733 | 0.733 | 32 | 0.754 | +6 / -9 | 5 / 3 | 91 (74885) | 5 | 827 / 1157 |

References (other packers, not re-run): saved HYBRID 0.720; saved HYBRID+rerank **0.806** (35 full); LEX5 replay glm-full-24 0.858 / oss20b-full-24 0.838 (LLM selector, fails the token gate); routing_offline ORACLE gold docs + per-doc reservation 0.805 (a conditional diagnostic, not a ceiling).

## 3. Chain per arm (questions of 54)

bundle-doc sufficiency: some sufficient bundle's documents are all reachable (global candidates' documents or nominated); nominated: all inside the nominations; candidate availability: every span of some bundle is inside some candidate package the arm can place (global + slot alternatives); feasible: those candidates can hold a complete bundle within 6,000 exact rendered chars (feasible oracle; with the 3,000-char anchor fixed in the second column); actual: the arm's pack retains a complete bundle.

| arm | bundle docs reachable | bundle docs nominated | candidates available | feasible (no anchor) | feasible (with anchor) | actual |
|---|---:|---:|---:|---:|---:|---:|
| HYBRID-intact | 53 | 53 | 46 | 46 | 44 | 32 |
| LEX6 | 53 | 53 | 47 | 47 | 45 | 32 |
| LEX6+rank=s2 | 53 | 50 | 47 | 47 | 45 | 33 |
| LEX6+rank=s5 | 53 | 52 | 47 | 47 | 45 | 32 |
| LEX6+absent_only | 53 | 53 | 47 | 47 | 45 | 32 |
| LEX6+global_only | 53 | 53 | 46 | 46 | 44 | 32 |
| LEX6+no_anchor | 53 | 53 | 47 | 47 | 45 | 32 |
| LEX6+cards | 53 | 51 | 47 | 47 | 45 | 32 |
| LEX6+sections | 53 | 53 | 47 | 47 | 45 | 30 |
| LEX6+cards,sections | 53 | 51 | 47 | 47 | 45 | 29 |
| LEX6+router | 53 | 53 | 47 | 47 | 45 | 32 |
| LEX6+router,diag5s | 53 | 53 | 47 | 47 | 45 | 32 |
| LEX6+router,diag5s,permute | 53 | 53 | 47 | 47 | 45 | 32 |
| gold-document oracle (every chunk of the gold docs as a package) | 54 | - | 54 | 54 | 47 | - |

Anchor opportunity cost (gold-document oracle): 7 questions feasible without the anchor but not with it. Truncated searches: 0.

## 4. Retention by slice (largest gold document's chunk count for length; tertiles)

| arm | fa (27) | en (27) | T1 (18) | T2 (18) | T3 (18) | RPT-fa (9) | short docs (<=108 chunks) (21) | mid docs (16) | long docs (>460 chunks) (17) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HYBRID-intact | 0.722 | 0.736 | 0.981 | 0.623 | 0.583 | 0.556 | 0.751 | 0.704 | 0.725 |
| LEX6 | 0.735 | 0.732 | 0.926 | 0.594 | 0.681 | 0.556 | 0.746 | 0.673 | 0.775 |
| LEX6+rank=s2 | 0.759 | 0.732 | 0.981 | 0.631 | 0.625 | 0.556 | 0.778 | 0.673 | 0.775 |
| LEX6+rank=s5 | 0.735 | 0.732 | 0.926 | 0.594 | 0.681 | 0.556 | 0.746 | 0.673 | 0.775 |
| LEX6+absent_only | 0.722 | 0.732 | 0.926 | 0.575 | 0.681 | 0.556 | 0.746 | 0.673 | 0.755 |
| LEX6+global_only | 0.722 | 0.732 | 0.926 | 0.575 | 0.681 | 0.556 | 0.746 | 0.673 | 0.755 |
| LEX6+no_anchor | 0.735 | 0.732 | 0.926 | 0.594 | 0.681 | 0.556 | 0.746 | 0.673 | 0.775 |
| LEX6+cards | 0.716 | 0.732 | 0.926 | 0.594 | 0.653 | 0.500 | 0.722 | 0.673 | 0.775 |
| LEX6+sections | 0.685 | 0.733 | 0.926 | 0.558 | 0.644 | 0.444 | 0.683 | 0.685 | 0.765 |
| LEX6+cards,sections | 0.667 | 0.715 | 0.926 | 0.558 | 0.588 | 0.389 | 0.659 | 0.685 | 0.735 |
| LEX6+router | 0.735 | 0.732 | 0.926 | 0.594 | 0.681 | 0.556 | 0.746 | 0.673 | 0.775 |
| LEX6+router,diag5s | 0.735 | 0.732 | 0.926 | 0.594 | 0.681 | 0.556 | 0.746 | 0.673 | 0.775 |
| LEX6+router,diag5s,permute | 0.735 | 0.732 | 0.926 | 0.594 | 0.681 | 0.556 | 0.746 | 0.673 | 0.775 |

## 5. LEX6+router

- Trigger fired on 54/54; calls attempted 54; router result used 0; deadline fallbacks 54 (100% of attempted; drop rule > 50%); invalid outputs 0.
- Statuses: {'fail:deadline': 54}
- Call wall ms p50 / p95: 676 / 684; tokens per attempted call with known usage: n/a; per question over all 54: 0; $ 0.00000; prompt tokens max est/provider: 683 / 0.

## 5. LEX6+router,diag5s

- Trigger fired on 54/54; calls attempted 54; router result used 6; deadline fallbacks 0 (0% of attempted; drop rule > 50%); invalid outputs 48.
- Statuses: {'fail:truncated': 48, 'ok': 6}
- Call wall ms p50 / p95: 739 / 1021; tokens per attempted call with known usage: 785; per question over all 54: 785; $ 0.00365; prompt tokens max est/provider: 683 / 750.

## 5. LEX6+router,diag5s,permute

- Trigger fired on 54/54; calls attempted 54; router result used 4; deadline fallbacks 0 (0% of attempted; drop rule > 50%); invalid outputs 50.
- Statuses: {'fail:truncated': 50, 'ok': 4}
- Call wall ms p50 / p95: 689 / 886; tokens per attempted call with known usage: 784; per question over all 54: 784; $ 0.00364; prompt tokens max est/provider: 683 / 750.

