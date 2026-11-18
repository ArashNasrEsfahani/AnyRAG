# Human-accuracy estimate (judge 1 as the predictor), GLM @6,000, 90 answerable

Census on the 30 reviewed questions + estimate on the 60 machine-concordant questions from a sample of 24. Unusable (unsure/blank) cells in packet A: 1. Unresolved machine cells on C: 0.

| pipeline | census R /30 | naive (machine on C) /90 | PPI /90 [95% CI] | RG per-pipeline /90 [95% CI] (sens, spec) | RG pooled /90 [95% CI] | sample: judge stricter / laxer / agree |
|---|---:|---:|---|---|---|---|
| HYBRID_6000 | 20 | 71 | 66.0 [60.8, 71.2] | 68.0 [63.2, 72.8] (1.00, 0.75) | 68.8 [66.7, 70.8] | 0 / 2 / 22 |
| HR_6000 | 25 | 76 | 73.4 [69.4, 77.4] | 74.5 [71.3, 77.7] (1.00, 0.86) | 73.8 [71.7, 75.8] | 0 / 1 / 22 |
| LEX5B_6000 | 25 | 76 | 71.0 [65.8, 76.2] | 73.0 [68.2, 77.8] (1.00, 0.75) | 73.8 [71.7, 75.8] | 0 / 2 / 22 |
| SETR_6000 | 27 | 78 | 75.5 [71.7, 79.3] | 76.5 [73.3, 79.7] (1.00, 0.86) | 75.8 [73.7, 77.8] | 0 / 1 / 23 |

| paired net /90 | census R | naive | PPI [95% CI] |
|---|---:|---:|---|
| LEX5B_6000 vs HYBRID_6000 | +5 | +5 | +5.0 [-2.8, 12.8] |
| LEX5B_6000 vs HR_6000 | +0 | +0 | +0.0 [-5.8, 5.8] |
| SETR_6000 vs LEX5B_6000 | +2 | +2 | +4.5 [-2.2, 11.2] |
| SETR_6000 vs HYBRID_6000 | +7 | +7 | +9.5 [2.8, 16.2] |
| SETR_6000 vs HR_6000 | +2 | +2 | +2.0 [-3.8, 7.8] |
| HR_6000 vs HYBRID_6000 | +5 | +5 | +7.6 [0.5, 14.7] |

Second reviewer (packet B) vs first reviewer (symmetric merge) and the judges; unusable cells: 1.

| group | cells | agree | kappa R1-R2 | R1 correct | R2 correct | kappa J1-R2 | J1 stricter/laxer vs R2 | kappa J2-R2 | J2 stricter/laxer vs R2 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---|
| all | 119 | 110 | 0.783 | 96 | 89 | 0.325 | 35 / 4 | 0.317 | 36 / 4 |
| HYBRID_6000 | 30 | 27 | 0.780 | 20 | 19 | 0.487 | 8 / 0 | 0.487 | 8 / 0 |
| HR_6000 | 30 | 27 | 0.710 | 25 | 22 | 0.317 | 6 / 2 | 0.312 | 8 / 2 |
| LEX5B_6000 | 30 | 29 | 0.889 | 25 | 24 | 0.386 | 7 / 1 | 0.260 | 9 / 1 |
| SETR_6000 | 29 | 27 | 0.713 | 26 | 24 | 0.107 | 14 / 1 | 0.143 | 11 / 1 |
