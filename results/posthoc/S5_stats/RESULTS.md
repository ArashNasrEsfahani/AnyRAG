# S5 statistics — post-hoc analyses (2026-11-10)

All of this is **post hoc and exploratory**, done after the test split was unsealed. No API calls, no GPU
(`CUDA_VISIBLE_DEVICES=""` in every process). Inputs are the frozen test verdicts, read through
`work/datasets_v2/test_run/report_test.py` via `paper/analysis/layout.py`, both read-only. Every number below was
copied from `tables.md`, `analyses.log`, `glmm_final.log` or `glmm_pack.log`, which the scripts in this folder generate
from `cells.json`, `stats_results.json`, `glmm_final_results.json` and `glmm_pack_results.json`.

## 0. Reproduction check (passed)

`s5_data.py` loads every reader × arm × question cell (4,608 cells: 4,573 judged, 35 `needs_adjudication`, all 35 on
answerable questions) and writes `cells.json`. It then runs 200 checks against `paper/analysis/revision_stats.json`
and `test_results.json`. They cover: 20 components; answerable bounds for all 48 reader × arm cells; net bounds, sign-p
ranges, unresolved-cell counts and resolved b/w for all 20 GLM primary/descriptive comparisons (machine and human_sym);
component-bootstrap CIs for the two primary comparisons and human LEX5B–HR; per-reader nets and interaction bounds.
**Result: 200/200 OK** (`repro_check.txt`). Examples: LEX5B–HYBRID @6000 net +5..+7, sign p .118–.332; LEX5B–HR @6000
net −3..0, sign p .607–1. The revision_stats helpers are copied into `s5_lib.py` (same RNG, seed 0, 10,000 resamples,
same percentile indices). They are not imported, because importing `revision_stats.py` would rewrite the paper's JSON.

## 1. What was run

| analysis | script | output |
|---|---|---|
| data + reproduction | `s5_data.py`, `s5_lib.py` | `cells.json`, `repro_check.txt` |
| GLMM, question intercept (AGQ) and Laplace (question; component+question) | `glmm_agq.py`, `glmm_final.py` | `glmm_final_results.json`, `glmm_final.log` |
| GLMM, question + pack intercepts (**primary GLMM**) | `glmm_pack.py` | `glmm_pack_results.json`, `glmm_pack.log` |
| TOST / non-inferiority, MDE, reader multiplicity | `analyses.py` | `stats_results.json`, `analyses.log` |
| markdown tables of all the above | `tables.py` | `tables.md` |

**GLMM specification.** Response: machine strict verdict. Fixed effects: arm, i.e. pipeline × budget as 8 cells with
treatment coding vs HYBRID_6000, plus reader in sum-to-zero coding, so arm contrasts are averages over the 6 readers on
the logit scale. This is form 0. Form 1 adds pipeline:reader (15 df). Form 2 also adds budget:reader +
pipeline:budget:reader (20 more df; the fixed part is then saturated). Unresolved cells are fitted twice, all 0 and all
1. Main scope: the 90 answerable questions (4,320 answers). Sensitivity scope: all 96 (4,608 answers).
Random-effect structures:
- (a) Question intercept, ML with **adaptive** Gauss–Hermite quadrature, 25 nodes (`AGQ`).
- (b) Question intercept, Laplace.
- (c) Component + question-in-component intercepts, Laplace (block-Newton joint mode, lme4 nAGQ=1 equivalent).
- (d) **Question + pack (question × arm) intercepts**, Laplace.

I added (d) because all six readers answered byte-identical packs, so their six answers to one pack are not
independent given the question. A question-only model overstates the information about arm contrasts. (d) is
strongly preferred over (a)/(b): pack-variance LRT χ² = 84.0, p ≈ 2e-20, boundary-halved, answerable90, unres=0.
SEs are Wald SEs from a finite-difference Hessian.

All three model structures converged on every form, scope and fill, with gradient ≤ 1e-5. The one exception: the
component variance in (c) goes to the boundary, σ_c ≈ 0.000–0.2 on answerable90, flagged "not converged" only because
the Hessian is singular at σ_c = 0. Statsmodels was not in the .venv. `pip install statsmodels` worked and installed
0.15.0, but I did not use it in the end: its mixed GLM is variational Bayes only, so I wrote the ML estimators myself
in numpy/scipy (gradients checked against finite differences).

## 2. Results

### 2.1 GLMM (all 6 readers pooled; answerable90; log-odds ratio (SE), Wald p)

Primary GLMM = (d) question + pack, form 0 (`tables.md`, "Contrasts: Laplace, question + pack intercepts, answerable90, form 0"):

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B vs HYBRID @6000 | +0.747 (0.273), OR 2.11 [1.24, 3.60], p = .006 | +0.701 (0.262), OR 2.02 [1.21, 3.37], p = .008 |
| LEX5B vs HR @6000 | +0.474 (0.271), OR 1.61 [0.94, 2.73], p = .080 | +0.342 (0.261), OR 1.41 [0.84, 2.35], p = .191 |
| HR vs HYBRID @6000 | +0.273 (0.268), p = .308 | +0.360 (0.258), p = .163 |
| SETR vs LEX5B @6000 | −0.284 (0.272), p = .297 | −0.264 (0.263), p = .315 |
| budget effect HYBRID (15500−6000) | +0.573 (0.271), OR 1.77, p = .034 | +0.622 (0.261), OR 1.86, p = .017 |
| budget effect HR | +0.474 (0.267), p = .076 | +0.360 (0.259), p = .164 |
| budget effect LEX5B | +0.125 (0.275), p = .649 | +0.180 (0.265), p = .498 |
| budget effect SETR | +0.076 (0.270), p = .779 | +0.050 (0.260), p = .848 |

σ_question = 3.35 / 3.12 and σ_pack = 0.97 / 0.92 (unres = 0 / 1). Population-averaged accuracy, mean over readers, %:
- unres=0: HYBRID_6000 62.0, HR_6000 64.6, LEX5B_6000 68.9, HYBRID_15500 67.4, HR_15500 68.9, LEX5B_15500 70.0.
- unres=1: 63.1, 66.7, 69.9, 69.2, 70.1, 71.6 in the same order.

These pooled p-values are not corrected for multiplicity.

The question-only models (a)/(b) give the same point estimates to two decimals on the question-only scale, but SEs are
about 30% smaller (LEX5B–HR +0.380 (0.186), p = .041 / +0.290 (0.185), p = .117). That is the pack correlation being
ignored, so those p-values are too small and should not be used. Laplace vs AGQ for (a): log-odds ratios agree to
0.001, σ_q 3.03 vs 3.09. The all-96 scope and form 1 (reader-averaged arm effects) change nothing material
(`tables.md`).

**Pipeline × reader interaction: not supported.** LRTs, model (d), answerable90:
- form 1 vs form 0 (pipeline:reader, 15 df): χ² = 12.40, p = .648 (unres=0); χ² = 13.01, p = .602 (unres=1).
- form 2 vs form 1 (budget:reader + pipeline:budget:reader, 20 df): p = .810 / .546.
- form 2 vs form 0 (all reader interactions, 35 df): p = .839 / .631.

Same conclusion under (a)/(b)/(c), with p between .60 and .96 throughout, and on all-96.

**Component (shared source family) random effect: not supported once the question is in the model.** Laplace σ_c → 0
on answerable90 (LRT χ² ≈ 0.00, p = .5 / χ² = 0.008, p = .47). A profile of σ_c ∈ {0.01, 0.25, 0.5, 1.0} raises the
nll by at most 0.84. On all-96, σ_c ≈ 0.85 with p = .14–.17.

Per-reader LEX5B–HR @6000 log-odds ratios implied by form 1 of model (d) (unres=0): GLM +0.02 (0.44), gpt-oss-20b
+0.28, Qwen3-235B +0.92 (0.43), Llama +0.53, Gemma +0.83 (0.40), DeepSeek +0.17. These are shown for description; the
interaction test above does not reject homogeneity.

### 2.2 Equivalence and non-inferiority, LEX5B vs HR @6000, GLM reader (margins 3/5/8 of 90 chosen post hoc)

TOST at α = .05 corresponds to a 90% interval. Bootstrap = paired cluster bootstrap over the 20 components (and over
questions), 10,000 resamples, seed 0. Machine intervals are the union over the two extreme assignments of unresolved
cells. TOST/NI p use the worst-case extreme on each side.

| judge | net /90 | 90% CI, component (pp; points of 90) | 90% CI, question (pp) | smallest margin for equivalence (component; question) |
|---|---|---|---|---|
| machine | −3..0 | [−9.23, +6.32]; [−8.31, +5.68] pts | [−10.00, +6.67] | 8.31; 9.00 points |
| symmetric human | 0 | [−6.12, +6.84]; [−5.51, +6.15] pts | [−5.56, +5.56] | 6.15; 5.00 points |

| margin (pts / pp) | machine TOST p (comp.; quest.) | machine NI p (comp.) | human TOST p (comp.; quest.) | human NI p (comp.; quest.) |
|---|---|---|---|---|
| 3 / 3.33 | .502; .547 — not equivalent | .502 — not NI | .209; .219 — not equivalent | .190; .205 — not NI |
| 5 / 5.56 | .272; .348 — not equivalent | .272 — not NI | .092; .078 — not equivalent | .068; .072 — not NI |
| 8 / 8.89 | .062; .120 — **not equivalent** | .062 — not NI | **.017; .008 — equivalent** | **.004; .007 — non-inferior** |

A cluster-robust t check (df = 19) agrees: machine TOST p at 8 points = .072, human = .018. On the machine judge,
equivalence cannot be claimed even at ±8 points (±8.9 pp). Under the symmetric human merge it can be claimed only at
±8 points. Non-inferiority of LEX5B to HR holds only at 8 points, and only for the human merge.

### 2.3 Minimum detectable effect (paired, exact two-sided sign test, α = .05, power .80, n = 90; GLM reader)

| comparison | discordance ψ (resolved; range over unresolved) | MDE, exact test | MDE, normal approx. | power to detect 5/90 at n = 90 | n to detect 5/90 (exact; normal) |
|---|---|---|---|---|---|
| LEX5B–HYBRID @6000 (b/w 11/4 of 88) | .170 (.167–.189) | **11.1 points of 90 = 12.4 pp** (11.0–11.7 pts) | 10.8 pts | .18 | **458** (448–506); 431 |
| LEX5B–HR @6000 (b/w 6/7 of 87) | .149 (.144–.178) | **10.4 points of 90 = 11.5 pp** (10.2–11.4 pts) | 10.1 pts | .19 | **401** (387–478); 378 |

These assume the observed discordance rate holds under the alternative. Component-clustering design effect, estimated
as the bootstrap variance ratio: 1.25 for LEX5B–HYBRID, which gives MDE 12.4 pts and n ≈ 537 (normal); 0.72 for
LEX5B–HR, under 1, so no inflation is warranted there.

### 2.4 Multiplicity across readers (tables/readers.tex: 6 readers × 8 columns = 48 tests)

Tests: an exact sign test for the 6 paired columns, and an exact sign-flip permutation test on the per-question
contrast (L6−C6)−(L15−C15) for the 2 interaction columns. Every p is the range over all assignments of unresolved
cells. Holm and BH adjusted p are monotone in each raw p, so adjusting the min-p and max-p vectors bounds the adjusted
p. Full table: `tables.md`, last section.

- Raw p < .05 under every assignment: 3 of 48 tests.
  - gpt-oss-20b LEX5B–HYBRID @6000: +10..+11, p .027–.041.
  - DeepSeek HR budget effect: +10..+12, p .0005–.0063.
  - DeepSeek interaction vs HR: +10..+13, p .0065–.041.
- Raw p < .05 under some assignment only: 2 more (Qwen3 LEX5B–HR @15500, p .049–.210; Qwen3 HYBRID budget effect,
  p .035–.119).
- Holm within each column (6 readers): only **DeepSeek HR budget effect** stays below .05 under every assignment
  (Holm p .003–.038; BH .003–.038). DeepSeek interaction vs HR: Holm .039–.243, not robust. gpt-oss LEX5B–HYBRID:
  Holm .160–.248.
- Holm over all 48: nothing survives under every assignment. DeepSeek HR budget Holm .023–.305, BH .023–.305.
  Everything else has Holm = 1 and BH ≥ .42.
- Over the 24 LEX5B-vs-control tests: no adjusted p below .05 (the Holm/BH-24 columns in `tables.md`).
- GLM row: no test below .118 raw.

## 3. Cost

$0.00. No API calls, no ledger. CPU only.

## 4. Failed, changed or skipped

- My first GLMM (`glmm.py`, non-adaptive Gauss–Hermite with K = 15/25) did not converge in K for the variance
  components. σ_q moved 2.28 → 2.41 → 3.06 between K = 15 → 25 → 100, and nll moved by about 1 unit. Fixed effects
  were stable to ±0.003. Its outputs (`glmm_results_K15/25.json`, `glmm_K15/25.log`) are **superseded** and kept only
  as a record. A K = 40 run was stopped.
- Nested adaptive quadrature (component + question, `glmm_agq.py` with `re=cq`) did not stabilise: the BFGS line
  search failed, and with a non-adaptive outer integral the nll was worse than the nested model's own σ_c → 0 limit.
  It is replaced by Laplace for all nested structures. `glmm_agq.py` is used only for the AGQ question-intercept
  model, which is stable from K = 15 to 25 (nll within 0.005).
- Laplace with 6 binary answers per pack can underestimate σ_pack, which would make the SEs of model (d) slightly too
  small. No AGQ cross-check exists for (d).
- A three-level component + question + pack model was not fitted, since σ_c ≈ 0 in the two-level model.
- The GLMM pools all six readers. The paper's primary analysis is the GLM reader only, so the GLMM p-values are a
  different, model-based, post-hoc estimand and are not Holm-corrected.

## 5. Facts for the paper

1. Pooling all six readers in a logistic GLMM with question and pack (question × arm) random intercepts, LEX5B beats
   HYBRID at 6,000 characters (OR 2.11 [1.24, 3.60], p = .006; 2.02, p = .008 with unresolved cells counted correct).
   Its advantage over HR is not significant (OR 1.61 [0.94, 2.73], p = .080; 1.41, p = .19). There is no evidence of a
   pipeline × reader interaction (LRT χ²(15) = 12.4, p = .65; 13.0, p = .60). Post hoc, uncorrected. Source:
   `S5_stats/glmm_pack_results.json`, `tables.md`.
2. The equivalence of LEX5B and HR at 6,000 is not established on the machine judge, even at ±8 of 90. The 90%
   component-bootstrap interval is [−8.3, +5.7] points (TOST p = .062 at 8 points). Under the symmetric human merge,
   equivalence holds only at ±8 points (90% CI [−5.5, +6.2] points, TOST p = .017), not at ±5 (p = .092). Margins were
   chosen post hoc. Source: `S5_stats/stats_results.json` → `tost`.
3. With n = 90 and the observed discordance (15–17%), the paired design has 80% power only for differences of about
   10–11 of 90 (11.5–12.4 pp). A 5-of-90 difference has about 18–19% power and would need about 400–460 questions.
   Source: `S5_stats/stats_results.json` → `mde`.
4. Across the 48 per-reader tests of the reader-dependence table, only DeepSeek-V4-flash's HR budget effect (+10..+12)
   survives a within-column Holm correction under every assignment of unresolved cells (Holm p .003–.038). Nothing
   survives Holm over all 48 tests. Source: `S5_stats/stats_results.json` → `readers_multiplicity`.
5. A shared-source-family component random effect adds nothing once questions are modelled (σ_c → 0 on the 90
   answerable questions, LRT p ≈ .5). The variation is at the question and pack level (pack-variance LRT χ² ≈ 84).
   Source: `S5_stats/glmm_final_results.json`, `glmm_pack_results.json`.
