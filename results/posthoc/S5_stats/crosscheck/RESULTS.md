# S5 GLMM cross-check against statsmodels (2026-11-18)

Post hoc, exploratory, like the rest of S5. No API calls, no GPU, $0.00. Nothing was installed: R is not on the
machine, and the .venv has statsmodels 0.15.0 and pandas but no pymer4, bambi, PyMC, NumPyro, rpy2 or glmmTMB. So
statsmodels is the only standard package used. It has no maximum-likelihood GLMM: its mixed logistic model is
`BinomialBayesMixedGLM` (variational Bayes or posterior mode).

**Independence.** `crosscheck.py` reads `../cells.json` directly and builds every design with patsy formulas. It does
not import `glmm*.py` or `s5_*.py`. The only thing it takes from the custom code is the custom estimates, read from
`../glmm_pack_results.json` for comparison. The case counts match the custom fits exactly: answerable90 has 4,320
answers, 2,942 correct with unres=0 and 2,977 with unres=1; all96 has 4,608 answers, 3,208 and 3,243.

## Files

| file | what |
|---|---|
| `crosscheck.py` | `freq` mode: GEE, GLM and the interaction tests. `bayes <vb\|map> <default\|diffuse> <unres> <ref>` mode: one BinomialBayesMixedGLM fit |
| `run_all.sh` | runs the 32 Bayes fits (18 in parallel, single-threaded BLAS) and the freq job |
| `combine.py` | merges the outputs into `crosscheck_results.json` and writes `tables.md` (every estimator, all contrasts, all four scope × unres cells) |
| `freq.log`, `bayes.log` | run logs |

## What was fitted

The fixed part is the same as the custom form-0 model in every fit: `y ~ C(arm, Treatment(ref)) + C(reader, Sum)`,
i.e. 8 arms treatment-coded plus 6 readers in sum-to-zero coding. The main scope is answerable90. Unresolved cells
are coded 0 and, separately, 1.

1. **statsmodels `BinomialBayesMixedGLM`** with variance components `question = 0 + C(q)` and
   `pack = 0 + C(q × arm)`. This is the same random structure as the custom model (d).
   - Fitted with `fit_vb()` (mean-field variational Bayes) and with `fit_map()`.
   - Two prior settings: the defaults (`fe_p = 2`, `vcp_p = 0.5`) and diffuse priors (`fe_p = 10`, `vcp_p = 5`). The
     default `vcp_p = 0.5` is an N(0, 0.5²) prior on log σ, which pulls σ_q ≈ 3.3 towards 1.
   - Mean-field VB gives only marginal posterior SDs, so each contrast was read as a single coefficient by refitting
     with its reference arm as the baseline (HYBRID_6000, HR_6000, LEX5B_6000, SETR_6000).
2. **statsmodels GEE**: logistic, exchangeable working correlation, grouped by question, robust sandwich SE. The
   bias-reduced (Mancl–DeRouen) SE was also computed; it is within 0.002 of the robust SE (see `tables.md`).
3. **statsmodels GLM**: logistic with cluster-robust SE by question. The naive iid SE is in `tables.md` for reference.
4. **Pipeline × reader interaction**: 15 columns built by hand (pipeline treatment-coded × reader sum-coded), tested
   three ways:
   - a GEE robust Wald χ²(15);
   - a GLM likelihood-ratio test with question fixed effects;
   - a GLM likelihood-ratio test with pack fixed effects. Readers vary within a pack, so the interaction is identified
     within packs.

   For both fixed-effect LR tests, questions or packs with a constant outcome were dropped, because they carry no
   within-group information and their fixed effects diverge.

## Main table (answerable90; log-odds ratio (SE); OR [95% interval]; p)

The VB interval is mean ± 1.96 posterior SD. No VB p is given: see the caveat below the tables.

**unresolved = wrong (0)**

| contrast | custom ML-Laplace (q + pack) | sm VB, diffuse | sm VB, default | sm GEE exch., robust | sm GLM, cluster-q |
|---|---|---|---|---|---|
| LEX5B vs HYBRID @6000 | +0.747 (0.273), OR 2.11 [1.24, 3.60], p .006 | +0.726 (0.142), OR 2.07 [1.56, 2.73] | +0.696 (0.142), OR 2.01 [1.52, 2.65] | +0.319 (0.138), OR 1.38 [1.05, 1.80], p .021 | +0.320 (0.139), OR 1.38 [1.05, 1.81], p .022 |
| LEX5B vs HR @6000 | +0.474 (0.271), OR 1.61 [0.94, 2.73], p .080 | +0.457 (0.142), OR 1.58 [1.19, 2.09] | +0.447 (0.142), OR 1.56 [1.18, 2.06] | +0.189 (0.159), OR 1.21 [0.89, 1.65], p .234 | +0.189 (0.160), OR 1.21 [0.88, 1.65], p .237 |
| HR vs HYBRID @6000 | +0.273 (0.268), OR 1.31 [0.78, 2.22], p .308 | +0.271 (0.137), OR 1.31 [1.00, 1.72] | +0.247 (0.137), OR 1.28 [0.98, 1.67] | +0.131 (0.133), OR 1.14 [0.88, 1.48], p .327 | +0.131 (0.134), OR 1.14 [0.88, 1.48], p .331 |
| budget HYBRID | +0.573 (0.271), OR 1.77 [1.04, 3.01], p .034 | +0.567 (0.141), OR 1.76 [1.34, 2.32] | +0.539 (0.140), OR 1.71 [1.30, 2.26] | +0.267 (0.128), OR 1.31 [1.02, 1.68], p .037 | +0.267 (0.129), OR 1.31 [1.01, 1.68], p .038 |
| budget HR | +0.474 (0.267), OR 1.61 [0.95, 2.71], p .076 | +0.477 (0.139), OR 1.61 [1.23, 2.12] | +0.467 (0.138), OR 1.60 [1.22, 2.09] | +0.234 (0.112), OR 1.26 [1.01, 1.57], p .036 | +0.234 (0.113), OR 1.26 [1.01, 1.58], p .038 |
| budget LEX5B | +0.125 (0.275), OR 1.13 [0.66, 1.94], p .649 | +0.133 (0.143), OR 1.14 [0.86, 1.51] | +0.157 (0.143), OR 1.17 [0.89, 1.55] | +0.072 (0.123), OR 1.07 [0.85, 1.37], p .556 | +0.072 (0.124), OR 1.07 [0.84, 1.37], p .560 |
| budget SETR | +0.076 (0.270), OR 1.08 [0.64, 1.83], p .779 | +0.082 (0.140), OR 1.09 [0.82, 1.43] | +0.088 (0.140), OR 1.09 [0.83, 1.44] | +0.043 (0.070), OR 1.04 [0.91, 1.20], p .540 | +0.043 (0.071), OR 1.04 [0.91, 1.20], p .543 |
| σ_question / σ_pack | 3.346 / 0.966 | 3.384 / 0.843 | 3.227 / 0.842 | working ρ = 0.493 | — |

**unresolved = right (1)**

| contrast | custom ML-Laplace | sm VB, diffuse | sm VB, default | sm GEE exch., robust | sm GLM, cluster-q |
|---|---|---|---|---|---|
| LEX5B vs HYBRID @6000 | +0.701 (0.262), OR 2.02 [1.21, 3.37], p .008 | +0.686 (0.140), OR 1.99 [1.51, 2.61] | +0.659 (0.140), OR 1.93 [1.47, 2.54] | +0.322 (0.143), OR 1.38 [1.04, 1.83], p .025 | +0.322 (0.145), OR 1.38 [1.04, 1.83], p .026 |
| LEX5B vs HR @6000 | +0.342 (0.261), OR 1.41 [0.84, 2.35], p .191 | +0.333 (0.140), OR 1.40 [1.06, 1.84] | +0.331 (0.140), OR 1.39 [1.06, 1.83] | +0.148 (0.162), OR 1.16 [0.84, 1.59], p .359 | +0.148 (0.163), OR 1.16 [0.84, 1.60], p .363 |
| HR vs HYBRID @6000 | +0.360 (0.258), OR 1.43 [0.86, 2.37], p .163 | +0.355 (0.135), OR 1.43 [1.09, 1.86] | +0.332 (0.135), OR 1.39 [1.07, 1.81] | +0.174 (0.134), OR 1.19 [0.92, 1.55], p .194 | +0.174 (0.135), OR 1.19 [0.91, 1.55], p .198 |
| budget HYBRID | +0.622 (0.261), OR 1.86 [1.12, 3.11], p .017 | +0.618 (0.140), OR 1.85 [1.41, 2.44] | +0.591 (0.139), OR 1.81 [1.38, 2.37] | +0.304 (0.127), OR 1.36 [1.06, 1.74], p .017 | +0.304 (0.128), OR 1.36 [1.05, 1.74], p .018 |
| budget HR | +0.360 (0.259), OR 1.43 [0.86, 2.38], p .164 | +0.371 (0.138), OR 1.45 [1.11, 1.90] | +0.368 (0.137), OR 1.45 [1.10, 1.89] | +0.203 (0.109), OR 1.22 [0.99, 1.52], p .064 | +0.203 (0.110), OR 1.23 [0.99, 1.52], p .066 |
| budget LEX5B | +0.180 (0.265), OR 1.20 [0.71, 2.01], p .498 | +0.184 (0.141), OR 1.20 [0.91, 1.58] | +0.203 (0.140), OR 1.23 [0.93, 1.61] | +0.091 (0.125), OR 1.10 [0.86, 1.40], p .465 | +0.092 (0.126), OR 1.10 [0.86, 1.40], p .468 |
| budget SETR | +0.050 (0.260), OR 1.05 [0.63, 1.75], p .848 | +0.057 (0.138), OR 1.06 [0.81, 1.39] | +0.062 (0.138), OR 1.06 [0.81, 1.39] | +0.035 (0.072), OR 1.04 [0.90, 1.19], p .631 | +0.035 (0.073), OR 1.04 [0.90, 1.19], p .634 |
| σ_question / σ_pack | 3.123 / 0.918 | 3.159 / 0.800 | 3.025 / 0.801 | working ρ = 0.478 | — |

The all96 scope (4,608 answers) has GEE and GLM only; the Bayes fits were run on answerable90 alone. It gives the same
picture. GEE: LEX5B–HYBRID +0.307 (0.133), p .021 / +0.310 (0.138), p .025; LEX5B–HR +0.174 (0.153), p .257 /
+0.135 (0.156), p .389 (`tables.md`).

**Pipeline × reader interaction (15 df)**

| test | answerable90 unres=0 | answerable90 unres=1 | all96 unres=0 | all96 unres=1 |
|---|---|---|---|---|
| custom Laplace LRT, model (d) | χ² 12.40, p .648 | 13.01, p .602 | 11.55, p .713 | 11.91, p .686 |
| sm GEE robust Wald | χ² 11.04, p .749 | 11.06, p .748 | 10.40, p .794 | 9.65, p .841 |
| sm GEE bias-reduced Wald | 10.80, p .767 | 10.82, p .765 | 10.18, p .808 | 9.45, p .853 |
| sm GLM LR, question FE (68–71 non-constant questions) | 11.39, p .724 | 12.24, p .661 | 10.58, p .782 | 11.20, p .738 |
| sm GLM LR, pack FE (317–338 non-constant packs) | 15.75, p .399 | 15.91, p .388 | 15.09, p .445 | 14.81, p .465 |

## Reading the comparison

- **VB point estimates reproduce the custom ML estimates.**
  - With diffuse priors, every contrast is within 0.02 log-odds of the custom estimate in both unres codings. The
    largest gap is 0.021, for LEX5B–HYBRID at unres=0.
  - With default priors, the shrinkage priors move the estimates by up to 0.05.
  - VB σ_question also agrees (3.38 vs 3.35). VB σ_pack is about 13% lower (0.84 vs 0.97).
- **VB posterior SDs are not usable as standard errors.**
  - They are about 0.52 times the custom Wald SEs, and smaller even than the naive iid GLM SE.
  - This is the known underestimation of mean-field VB. The factorisation ignores the posterior dependence between each
    arm coefficient and that arm's 90 pack effects. Pack is question × arm, so the mean of an arm's pack effects is
    almost confounded with the arm effect.
  - So VB checks the **estimates**, not the **uncertainty**. That is why no VB p is given.
- **The uncertainty check comes from GEE and GLM-cluster.**
  - These are marginal (population-averaged) estimators with a well-founded sandwich SE.
  - Their log-ORs are about 2.2–2.5 times smaller than the conditional ones. That is the expected attenuation: the
    factor √(1 + 0.346(σ_q² + σ_p²)) ≈ 2.28 (unres=0) and 2.16 (unres=1).
  - Their z-statistics are the comparable quantity:

    | contrast | custom z (unres 0 / 1) | GEE z (unres 0 / 1) |
    |---|---|---|
    | LEX5B–HYBRID | 2.74 / 2.67 | 2.31 / 2.25 |
    | LEX5B–HR | 1.75 / 1.31 | 1.19 / 0.92 |

  - The direction is the same everywhere. Significance agrees on every reported contrast except one (next bullet). The
    GEE is, if anything, more conservative on the LEX5B–HR question.
- **The one difference in significance is the HR budget effect at unres=0.**
  - Custom: p = .076. GEE: p = .036; GLM-cluster: p = .038.
  - At unres=1 the two agree (custom p = .164; GEE p = .064).
  - None of the headline claims depend on this contrast. If the paper mentions the HR budget effect at all, it should
    be described as borderline, with the estimator-dependent p noted.
- **`fit_map` failed and is excluded.** Its joint posterior mode is degenerate:
  - σ_pack collapses to about 1e-8 and σ_question runs to 12–45;
  - the arm estimates change sign depending on which arm is the reference, e.g. LEX5B–HYBRID +0.004 (diffuse) vs
    +1.631 (default).

  This is the known failure of joint modes over random effects and variance parameters, not evidence about the data.
  The values are kept in `crosscheck_results.json` / `tables.md` for the record.

## Verdict on the custom estimator's conclusions

1. **LEX5B vs HYBRID @6000, positive and significant pooled: supported.**
   - VB reproduces the conditional log-OR, 0.73 vs 0.75.
   - GEE and GLM-cluster give a positive, significant marginal effect: OR 1.38 [1.05, 1.80], p = .021 (unres=0);
     p = .025 (unres=1). Bias-reduced SE: p = .022 / .026.
2. **LEX5B vs HR @6000, not significant: supported.**
   - VB reproduces the point estimate: 0.46 vs 0.47 (unres=0); 0.33 vs 0.34 (unres=1).
   - GEE p = .234 / .359 and GLM-cluster p = .237 / .363, both further from significance than the custom p = .080 / .191.
3. **No pipeline × reader interaction: supported.**
   - GEE robust Wald χ²(15) p = .75 / .75.
   - GLM LR with question fixed effects p = .72 / .66.
   - GLM LR with pack fixed effects p = .40 / .39.
   - Custom LRT p = .65 / .60.
4. **Budget effects: supported.**
   - HYBRID: significant positive in all estimators.
   - LEX5B and SETR: null in all estimators.
   - HR: borderline. The custom estimator is non-significant; GEE and GLM-cluster are just under .05 at unres=0 only.

What this does **not** check:
- the custom Laplace **standard errors** for the conditional model. No standard package available here fits that model
  by ML. The SE check is indirect, through the marginal GEE and GLM-cluster z-statistics;
- the Laplace-vs-quadrature accuracy for σ_pack (already listed as a limitation in `../RESULTS.md` §4).

## Paragraph for the paper

> We cross-checked the custom Laplace-ML GLMM against statsmodels 0.15.0 (lme4/glmmTMB were unavailable): its
> variational-Bayes mixed logistic model (`BinomialBayesMixedGLM`, same fixed effects, question and pack random
> intercepts, 4,320 answers) reproduced the conditional log-odds ratios to within 0.02 (LEX5B vs HYBRID at 6,000:
> 0.73 vs 0.75; vs HR: 0.46 vs 0.47), although its mean-field posterior SDs are about half the Wald SEs and were not used
> for inference, and its posterior-mode fit was degenerate and discarded. A population-averaged logistic GEE
> (exchangeable, clustered by question, robust SEs) and a GLM with question-clustered SEs gave the same directions with
> the expected marginal attenuation (LEX5B vs HYBRID OR 1.38 [1.05, 1.80], p = .021; vs HR OR 1.21 [0.89, 1.65],
> p = .23), and neither a GEE Wald test (χ²(15) = 11.0, p = .75) nor likelihood-ratio tests in GLMs with question or
> pack fixed effects (p = .72, .40) found a pipeline × reader interaction. The only discrepancy is the HR budget effect,
> borderline under GEE (p = .036) but not under the GLMM (p = .076); none of our claims rest on it.
