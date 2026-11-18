## GLMM

Estimators: AGQ = ML, adaptive Gauss-Hermite quadrature, 25 nodes, question random intercept; LAPLACE = ML, Laplace approximation (lme4 nAGQ=1 equivalent), block-Newton joint mode

### Fit summary

| fit | n | fixed | logLik | converged | max abs grad | min Hess eig | sigma_q | sigma_comp |
|---|---|---|---|---|---|---|---|---|
| AGQ|answerable90|unres=0|form=0|re=q | 4320 | 13 | -1591.813 | True | 2.8e-07 | 5.8 | 3.086 | - |
| LAPLACE|answerable90|unres=0|form=0|re=q | 4320 | 13 | -1593.652 | True | 8.8e-07 | 5.8 | 3.029 | - |
| LAPLACE|answerable90|unres=0|form=0|re=cq | 4320 | 13 | -1593.652 | False | 6.9e-06 | -1.6e-11 | 3.029 | 0.000 |
| AGQ|answerable90|unres=0|form=1|re=q | 4320 | 28 | -1586.227 | True | 1.9e-05 | 3.7 | 3.098 | - |
| LAPLACE|answerable90|unres=0|form=1|re=q | 4320 | 28 | -1588.072 | True | 1.5e-06 | 3.7 | 3.040 | - |
| LAPLACE|answerable90|unres=0|form=1|re=cq | 4320 | 28 | -1588.072 | False | 1.6e-05 | -5.7e-06 | 3.040 | 0.000 |
| AGQ|answerable90|unres=0|form=2|re=q | 4320 | 48 | -1579.568 | True | 7.6e-07 | 0.72 | 3.112 | - |
| LAPLACE|answerable90|unres=0|form=2|re=q | 4320 | 48 | -1581.421 | True | 7.8e-06 | 0.72 | 3.054 | - |
| LAPLACE|answerable90|unres=0|form=2|re=cq | 4320 | 48 | -1581.421 | False | 4.8e-06 | -3.2e-10 | 3.054 | 0.001 |
| AGQ|answerable90|unres=1|form=0|re=q | 4320 | 13 | -1616.691 | True | 2.3e-05 | 6.1 | 2.902 | - |
| LAPLACE|answerable90|unres=1|form=0|re=q | 4320 | 13 | -1618.362 | True | 4.0e-06 | 6.1 | 2.856 | - |
| LAPLACE|answerable90|unres=1|form=0|re=cq | 4320 | 13 | -1618.358 | True | 4.4e-06 | 0.029 | 2.848 | 0.201 |
| AGQ|answerable90|unres=1|form=1|re=q | 4320 | 28 | -1610.695 | True | 7.0e-05 | 3.8 | 2.914 | - |
| LAPLACE|answerable90|unres=1|form=1|re=q | 4320 | 28 | -1612.371 | True | 2.0e-06 | 3.8 | 2.867 | - |
| LAPLACE|answerable90|unres=1|form=1|re=cq | 4320 | 28 | -1612.367 | True | 9.1e-06 | 0.028 | 2.860 | 0.201 |
| AGQ|answerable90|unres=1|form=2|re=q | 4320 | 48 | -1602.177 | True | 1.7e-06 | 0.73 | 2.931 | - |
| LAPLACE|answerable90|unres=1|form=2|re=q | 4320 | 48 | -1603.862 | True | 7.9e-06 | 0.73 | 2.883 | - |
| LAPLACE|answerable90|unres=1|form=2|re=cq | 4320 | 48 | -1603.858 | True | 3.3e-06 | 0.027 | 2.876 | 0.200 |
| AGQ|all96|unres=0|form=0|re=q | 4608 | 13 | -1631.983 | True | 1.4e-04 | 5.7 | 3.263 | - |
| LAPLACE|all96|unres=0|form=0|re=q | 4608 | 13 | -1634.097 | True | 4.5e-06 | 5.8 | 3.193 | - |
| LAPLACE|all96|unres=0|form=0|re=cq | 4608 | 13 | -1633.626 | True | 2.8e-06 | 2.3 | 3.084 | 0.852 |
| AGQ|all96|unres=0|form=1|re=q | 4608 | 28 | -1626.788 | True | 4.6e-05 | 3.8 | 3.273 | - |
| LAPLACE|all96|unres=0|form=1|re=q | 4608 | 28 | -1628.909 | True | 5.6e-06 | 3.8 | 3.203 | - |
| LAPLACE|all96|unres=0|form=1|re=cq | 4608 | 28 | -1628.439 | True | 9.0e-06 | 2.3 | 3.095 | 0.854 |
| AGQ|all96|unres=0|form=2|re=q | 4608 | 48 | -1620.606 | True | 1.1e-05 | 0.73 | 3.286 | - |
| LAPLACE|all96|unres=0|form=2|re=q | 4608 | 48 | -1622.736 | True | 8.0e-06 | 0.73 | 3.215 | - |
| LAPLACE|all96|unres=0|form=2|re=cq | 4608 | 48 | -1622.268 | True | 7.9e-06 | 0.73 | 3.107 | 0.856 |
| AGQ|all96|unres=1|form=0|re=q | 4608 | 13 | -1657.085 | True | 2.0e-05 | 6.1 | 3.072 | - |
| LAPLACE|all96|unres=1|form=0|re=q | 4608 | 13 | -1659.003 | True | 8.9e-06 | 6.1 | 3.015 | - |
| LAPLACE|all96|unres=1|form=0|re=cq | 4608 | 13 | -1658.407 | True | 9.3e-06 | 2.7 | 2.904 | 0.844 |
| AGQ|all96|unres=1|form=1|re=q | 4608 | 28 | -1651.592 | True | 7.1e-08 | 3.8 | 3.083 | - |
| LAPLACE|all96|unres=1|form=1|re=q | 4608 | 28 | -1653.516 | True | 2.1e-06 | 3.8 | 3.026 | - |
| LAPLACE|all96|unres=1|form=1|re=cq | 4608 | 28 | -1652.921 | True | 5.9e-06 | 2.7 | 2.914 | 0.846 |
| AGQ|all96|unres=1|form=2|re=q | 4608 | 48 | -1643.569 | True | 3.9e-07 | 0.75 | 3.099 | - |
| LAPLACE|all96|unres=1|form=2|re=q | 4608 | 48 | -1645.503 | True | 7.5e-06 | 0.75 | 3.041 | - |
| LAPLACE|all96|unres=1|form=2|re=cq | 4608 | 48 | -1644.911 | True | 4.2e-06 | 0.75 | 2.929 | 0.849 |

### Contrasts: AGQ, question intercept, answerable90, form 0 (arm + reader)

log-odds ratio (SE); OR [95% Wald CI]; Wald p

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +0.641 (0.185); OR 1.90 [1.32, 2.73]; p=0.001 | +0.628 (0.183); OR 1.87 [1.31, 2.68]; p=0.001 |
| LEX5B_6000 vs HR_6000 | +0.380 (0.186); OR 1.46 [1.02, 2.11]; p=0.041 | +0.290 (0.185); OR 1.34 [0.93, 1.92]; p=0.117 |
| HR_6000 vs HYBRID_6000 | +0.261 (0.181); OR 1.30 [0.91, 1.85]; p=0.149 | +0.338 (0.180); OR 1.40 [0.99, 1.99]; p=0.060 |
| SETR_6000 vs LEX5B_6000 | -0.228 (0.188); OR 0.80 [0.55, 1.15]; p=0.224 | -0.223 (0.185); OR 0.80 [0.56, 1.15]; p=0.229 |
| SETR_6000 vs HYBRID_6000 | +0.413 (0.182); OR 1.51 [1.06, 2.16]; p=0.023 | +0.405 (0.180); OR 1.50 [1.05, 2.13]; p=0.025 |
| budget HYBRID (15500-6000) | +0.535 (0.184); OR 1.71 [1.19, 2.45]; p=0.004 | +0.593 (0.182); OR 1.81 [1.27, 2.59]; p=0.001 |
| budget HR (15500-6000) | +0.471 (0.188); OR 1.60 [1.11, 2.31]; p=0.012 | +0.397 (0.186); OR 1.49 [1.03, 2.14]; p=0.033 |
| budget LEX5B (15500-6000) | +0.146 (0.191); OR 1.16 [0.80, 1.68]; p=0.444 | +0.179 (0.190); OR 1.20 [0.83, 1.73]; p=0.344 |
| budget SETR (15500-6000) | +0.086 (0.186); OR 1.09 [0.76, 1.57]; p=0.642 | +0.068 (0.184); OR 1.07 [0.75, 1.53]; p=0.713 |

Population-averaged accuracy (%, mean over readers): unres=0: HYBRID_6000 61.6, HR_6000 64.3, LEX5B_6000 68.2, SETR_6000 65.9, HYBRID_15500 67.1, HR_15500 69.1, LEX5B_15500 69.7, SETR_15500 66.8. unres=1: HYBRID_6000 62.6, HR_6000 66.3, LEX5B_6000 69.4, SETR_6000 67.0, HYBRID_15500 69.0, HR_15500 70.4, LEX5B_15500 71.2, SETR_15500 67.7.

### Contrasts: AGQ, question intercept, answerable90, form 1 (arm + reader + pipeline:reader; arm effects averaged over readers (sum coding))

log-odds ratio (SE); OR [95% Wald CI]; Wald p

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +0.639 (0.185); OR 1.89 [1.32, 2.72]; p=0.001 | +0.627 (0.183); OR 1.87 [1.31, 2.68]; p=0.001 |
| LEX5B_6000 vs HR_6000 | +0.365 (0.187); OR 1.44 [1.00, 2.08]; p=0.051 | +0.272 (0.186); OR 1.31 [0.91, 1.89]; p=0.143 |
| HR_6000 vs HYBRID_6000 | +0.274 (0.182); OR 1.31 [0.92, 1.88]; p=0.132 | +0.355 (0.181); OR 1.43 [1.00, 2.03]; p=0.050 |
| SETR_6000 vs LEX5B_6000 | -0.224 (0.188); OR 0.80 [0.55, 1.16]; p=0.233 | -0.222 (0.186); OR 0.80 [0.56, 1.15]; p=0.232 |
| SETR_6000 vs HYBRID_6000 | +0.415 (0.183); OR 1.51 [1.06, 2.17]; p=0.023 | +0.405 (0.181); OR 1.50 [1.05, 2.14]; p=0.025 |
| budget HYBRID (15500-6000) | +0.536 (0.184); OR 1.71 [1.19, 2.45]; p=0.004 | +0.594 (0.183); OR 1.81 [1.27, 2.59]; p=0.001 |
| budget HR (15500-6000) | +0.477 (0.189); OR 1.61 [1.11, 2.33]; p=0.012 | +0.402 (0.187); OR 1.49 [1.04, 2.16]; p=0.032 |
| budget LEX5B (15500-6000) | +0.147 (0.191); OR 1.16 [0.80, 1.69]; p=0.444 | +0.180 (0.190); OR 1.20 [0.83, 1.74]; p=0.343 |
| budget SETR (15500-6000) | +0.087 (0.186); OR 1.09 [0.76, 1.57]; p=0.641 | +0.068 (0.184); OR 1.07 [0.75, 1.53]; p=0.713 |

Population-averaged accuracy (%, mean over readers): unres=0: HYBRID_6000 61.6, HR_6000 64.4, LEX5B_6000 68.2, SETR_6000 65.9, HYBRID_15500 67.1, HR_15500 69.2, LEX5B_15500 69.6, SETR_15500 66.8. unres=1: HYBRID_6000 62.6, HR_6000 66.4, LEX5B_6000 69.3, SETR_6000 67.0, HYBRID_15500 69.0, HR_15500 70.5, LEX5B_15500 71.1, SETR_15500 67.7.

### Contrasts: AGQ, question intercept, all96, form 0 (arm + reader)

log-odds ratio (SE); OR [95% Wald CI]; Wald p

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +0.627 (0.183); OR 1.87 [1.31, 2.68]; p=0.001 | +0.615 (0.181); OR 1.85 [1.30, 2.64]; p=0.001 |
| LEX5B_6000 vs HR_6000 | +0.356 (0.184); OR 1.43 [0.99, 2.05]; p=0.054 | +0.267 (0.183); OR 1.31 [0.91, 1.87]; p=0.144 |
| HR_6000 vs HYBRID_6000 | +0.272 (0.179); OR 1.31 [0.92, 1.86]; p=0.129 | +0.347 (0.178); OR 1.42 [1.00, 2.01]; p=0.051 |
| SETR_6000 vs LEX5B_6000 | -0.223 (0.185); OR 0.80 [0.56, 1.15]; p=0.229 | -0.218 (0.183); OR 0.80 [0.56, 1.15]; p=0.234 |
| SETR_6000 vs HYBRID_6000 | +0.404 (0.180); OR 1.50 [1.05, 2.13]; p=0.025 | +0.396 (0.179); OR 1.49 [1.05, 2.11]; p=0.026 |
| budget HYBRID (15500-6000) | +0.523 (0.182); OR 1.69 [1.18, 2.41]; p=0.004 | +0.580 (0.181); OR 1.79 [1.25, 2.55]; p=0.001 |
| budget HR (15500-6000) | +0.462 (0.186); OR 1.59 [1.10, 2.28]; p=0.013 | +0.389 (0.184); OR 1.48 [1.03, 2.12]; p=0.035 |
| budget LEX5B (15500-6000) | +0.143 (0.189); OR 1.15 [0.80, 1.67]; p=0.450 | +0.175 (0.187); OR 1.19 [0.83, 1.72]; p=0.349 |
| budget SETR (15500-6000) | +0.085 (0.184); OR 1.09 [0.76, 1.56]; p=0.646 | +0.066 (0.182); OR 1.07 [0.75, 1.53]; p=0.716 |

Population-averaged accuracy (%, mean over readers): unres=0: HYBRID_6000 63.8, HR_6000 66.5, LEX5B_6000 69.9, SETR_6000 67.8, HYBRID_15500 68.9, HR_15500 70.9, LEX5B_15500 71.2, SETR_15500 68.6. unres=1: HYBRID_6000 64.8, HR_6000 68.4, LEX5B_6000 71.0, SETR_6000 68.9, HYBRID_15500 70.6, HR_15500 72.1, LEX5B_15500 72.6, SETR_15500 69.5.

### Contrasts: AGQ, question intercept, all96, form 1 (arm + reader + pipeline:reader; arm effects averaged over readers (sum coding))

log-odds ratio (SE); OR [95% Wald CI]; Wald p

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +0.624 (0.183); OR 1.87 [1.30, 2.67]; p=0.001 | +0.613 (0.181); OR 1.85 [1.29, 2.63]; p=0.001 |
| LEX5B_6000 vs HR_6000 | +0.340 (0.185); OR 1.41 [0.98, 2.02]; p=0.067 | +0.249 (0.184); OR 1.28 [0.89, 1.84]; p=0.176 |
| HR_6000 vs HYBRID_6000 | +0.284 (0.180); OR 1.33 [0.93, 1.89]; p=0.115 | +0.364 (0.179); OR 1.44 [1.01, 2.04]; p=0.042 |
| SETR_6000 vs LEX5B_6000 | -0.219 (0.186); OR 0.80 [0.56, 1.16]; p=0.238 | -0.217 (0.184); OR 0.81 [0.56, 1.15]; p=0.238 |
| SETR_6000 vs HYBRID_6000 | +0.405 (0.181); OR 1.50 [1.05, 2.14]; p=0.025 | +0.396 (0.179); OR 1.49 [1.05, 2.11]; p=0.027 |
| budget HYBRID (15500-6000) | +0.525 (0.182); OR 1.69 [1.18, 2.41]; p=0.004 | +0.582 (0.181); OR 1.79 [1.26, 2.55]; p=0.001 |
| budget HR (15500-6000) | +0.467 (0.187); OR 1.60 [1.11, 2.30]; p=0.012 | +0.394 (0.185); OR 1.48 [1.03, 2.13]; p=0.034 |
| budget LEX5B (15500-6000) | +0.143 (0.189); OR 1.15 [0.80, 1.67]; p=0.450 | +0.175 (0.187); OR 1.19 [0.83, 1.72]; p=0.349 |
| budget SETR (15500-6000) | +0.085 (0.184); OR 1.09 [0.76, 1.56]; p=0.645 | +0.066 (0.182); OR 1.07 [0.75, 1.53]; p=0.716 |

Population-averaged accuracy (%, mean over readers): unres=0: HYBRID_6000 63.8, HR_6000 66.5, LEX5B_6000 69.9, SETR_6000 67.8, HYBRID_15500 68.9, HR_15500 70.9, LEX5B_15500 71.2, SETR_15500 68.6. unres=1: HYBRID_6000 64.8, HR_6000 68.4, LEX5B_6000 71.0, SETR_6000 68.8, HYBRID_15500 70.6, HR_15500 72.2, LEX5B_15500 72.6, SETR_15500 69.5.

### Estimator sensitivity (answerable90, form 0): LEX5B-HR and LEX5B-HYBRID @6000 log-odds ratios (SE)

| fit | LEX5B-HR | LEX5B-HYBRID | sigma_q | sigma_c |
|---|---|---|---|---|
| AGQ re=q unres=0 | +0.380 (0.186) | +0.641 (0.185) | 3.086 | - |
| LAPLACE re=q unres=0 | +0.380 (0.186) | +0.641 (0.185) | 3.029 | - |
| LAPLACE re=cq unres=0 | +0.380 (0.184) | +0.641 (0.142) | 3.029 | 0.000 |
| AGQ re=q unres=1 | +0.290 (0.185) | +0.628 (0.183) | 2.902 | - |
| LAPLACE re=q unres=1 | +0.290 (0.185) | +0.627 (0.183) | 2.856 | - |
| LAPLACE re=cq unres=1 | +0.290 (0.185) | +0.627 (0.183) | 2.848 | 0.201 |

### Likelihood-ratio tests

| test | stat | df | p |
|---|---|---|---|
| AGQ|answerable90|unres=0|re=q: form1 vs form0 (pipeline:reader) | 11.174 | 15 | 0.74 |
| AGQ|answerable90|unres=0|re=q: form2 vs form1 (budget:reader + pipeline:budget:reader) | 13.318 | 20 | 0.863 |
| AGQ|answerable90|unres=0|re=q: form2 vs form0 (all reader interactions) | 24.491 | 35 | 0.908 |
| AGQ|answerable90|unres=1|re=q: form1 vs form0 (pipeline:reader) | 11.992 | 15 | 0.68 |
| AGQ|answerable90|unres=1|re=q: form2 vs form1 (budget:reader + pipeline:budget:reader) | 17.036 | 20 | 0.651 |
| AGQ|answerable90|unres=1|re=q: form2 vs form0 (all reader interactions) | 29.028 | 35 | 0.751 |
| AGQ|all96|unres=0|re=q: form1 vs form0 (pipeline:reader) | 10.391 | 15 | 0.794 |
| AGQ|all96|unres=0|re=q: form2 vs form1 (budget:reader + pipeline:budget:reader) | 12.363 | 20 | 0.903 |
| AGQ|all96|unres=0|re=q: form2 vs form0 (all reader interactions) | 22.753 | 35 | 0.945 |
| AGQ|all96|unres=1|re=q: form1 vs form0 (pipeline:reader) | 10.987 | 15 | 0.754 |
| AGQ|all96|unres=1|re=q: form2 vs form1 (budget:reader + pipeline:budget:reader) | 16.045 | 20 | 0.714 |
| AGQ|all96|unres=1|re=q: form2 vs form0 (all reader interactions) | 27.032 | 35 | 0.83 |
| LAPLACE|answerable90|unres=0|re=q: form1 vs form0 (pipeline:reader) | 11.162 | 15 | 0.741 |
| LAPLACE|answerable90|unres=0|re=q: form2 vs form1 (budget:reader + pipeline:budget:reader) | 13.301 | 20 | 0.864 |
| LAPLACE|answerable90|unres=0|re=q: form2 vs form0 (all reader interactions) | 24.463 | 35 | 0.909 |
| LAPLACE|answerable90|unres=1|re=q: form1 vs form0 (pipeline:reader) | 11.981 | 15 | 0.68 |
| LAPLACE|answerable90|unres=1|re=q: form2 vs form1 (budget:reader + pipeline:budget:reader) | 17.019 | 20 | 0.652 |
| LAPLACE|answerable90|unres=1|re=q: form2 vs form0 (all reader interactions) | 29.000 | 35 | 0.752 |
| LAPLACE|all96|unres=0|re=q: form1 vs form0 (pipeline:reader) | 10.377 | 15 | 0.795 |
| LAPLACE|all96|unres=0|re=q: form2 vs form1 (budget:reader + pipeline:budget:reader) | 12.345 | 20 | 0.904 |
| LAPLACE|all96|unres=0|re=q: form2 vs form0 (all reader interactions) | 22.722 | 35 | 0.946 |
| LAPLACE|all96|unres=1|re=q: form1 vs form0 (pipeline:reader) | 10.974 | 15 | 0.754 |
| LAPLACE|all96|unres=1|re=q: form2 vs form1 (budget:reader + pipeline:budget:reader) | 16.026 | 20 | 0.715 |
| LAPLACE|all96|unres=1|re=q: form2 vs form0 (all reader interactions) | 27.001 | 35 | 0.831 |
| LAPLACE|answerable90|unres=0|re=cq: form1 vs form0 (pipeline:reader) | 11.162 | 15 | 0.741 |
| LAPLACE|answerable90|unres=0|re=cq: form2 vs form1 (budget:reader + pipeline:budget:reader) | 13.301 | 20 | 0.864 |
| LAPLACE|answerable90|unres=0|re=cq: form2 vs form0 (all reader interactions) | 24.463 | 35 | 0.909 |
| LAPLACE|answerable90|unres=1|re=cq: form1 vs form0 (pipeline:reader) | 11.981 | 15 | 0.68 |
| LAPLACE|answerable90|unres=1|re=cq: form2 vs form1 (budget:reader + pipeline:budget:reader) | 17.019 | 20 | 0.652 |
| LAPLACE|answerable90|unres=1|re=cq: form2 vs form0 (all reader interactions) | 28.999 | 35 | 0.752 |
| LAPLACE|all96|unres=0|re=cq: form1 vs form0 (pipeline:reader) | 10.374 | 15 | 0.796 |
| LAPLACE|all96|unres=0|re=cq: form2 vs form1 (budget:reader + pipeline:budget:reader) | 12.341 | 20 | 0.904 |
| LAPLACE|all96|unres=0|re=cq: form2 vs form0 (all reader interactions) | 22.715 | 35 | 0.946 |
| LAPLACE|all96|unres=1|re=cq: form1 vs form0 (pipeline:reader) | 10.972 | 15 | 0.755 |
| LAPLACE|all96|unres=1|re=cq: form2 vs form1 (budget:reader + pipeline:budget:reader) | 16.021 | 20 | 0.715 |
| LAPLACE|all96|unres=1|re=cq: form2 vs form0 (all reader interactions) | 26.993 | 35 | 0.832 |
| LAPLACE|answerable90|unres=0|form=0: re=cq vs re=q (component variance; boundary, chi2(1)/2) | -0.000 | 1 | 0.5 |
| LAPLACE|answerable90|unres=0|form=1: re=cq vs re=q (component variance; boundary, chi2(1)/2) | -0.000 | 1 | 0.5 |
| LAPLACE|answerable90|unres=0|form=2: re=cq vs re=q (component variance; boundary, chi2(1)/2) | -0.000 | 1 | 0.5 |
| LAPLACE|answerable90|unres=1|form=0: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 0.008 | 1 | 0.465 |
| LAPLACE|answerable90|unres=1|form=1: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 0.007 | 1 | 0.466 |
| LAPLACE|answerable90|unres=1|form=2: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 0.007 | 1 | 0.467 |
| LAPLACE|all96|unres=0|form=0: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 0.943 | 1 | 0.166 |
| LAPLACE|all96|unres=0|form=1: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 0.941 | 1 | 0.166 |
| LAPLACE|all96|unres=0|form=2: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 0.936 | 1 | 0.167 |
| LAPLACE|all96|unres=1|form=0: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 1.192 | 1 | 0.137 |
| LAPLACE|all96|unres=1|form=1: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 1.190 | 1 | 0.138 |
| LAPLACE|all96|unres=1|form=2: re=cq vs re=q (component variance; boundary, chi2(1)/2) | 1.184 | 1 | 0.138 |

### Per-reader LEX5B-HR @6000 log-odds ratio implied by form 1 (AGQ, answerable90)

| reader | unres=0 | unres=1 |
|---|---|---|
| glm-5.3-flash | -0.033 (0.378) | -0.206 (0.376) |
| gpt-oss-20b | +0.209 (0.340) | +0.157 (0.338) |
| qwen3-235b-a22b-2507 | +0.772 (0.360) | +0.775 (0.359) |
| llama-3.3-70b-instruct | +0.426 (0.352) | +0.264 (0.348) |
| gemma-3-27b-it | +0.718 (0.333) | +0.655 (0.330) |
| deepseek-v4-flash | +0.099 (0.367) | -0.009 (0.364) |

## GLMM with pack random effect (PRIMARY model): ML, Laplace approximation; random intercepts for question and for pack (question x arm) nested in question

| fit | n | fixed | logLik | converged | max abs grad | min Hess eig | sigma_question | sigma_pack |
|---|---|---|---|---|---|---|---|---|
| LAPLACE|answerable90|unres=0|form=0|re=q+pack | 4320 | 13 | -1551.645 | True | 3.1e-06 | 2.9 | 3.346 | 0.966 |
| LAPLACE|answerable90|unres=0|form=1|re=q+pack | 4320 | 28 | -1545.443 | True | 6.2e-06 | 2.9 | 3.364 | 0.975 |
| LAPLACE|answerable90|unres=0|form=2|re=q+pack | 4320 | 48 | -1538.247 | True | 4.5e-06 | 0.59 | 3.384 | 0.982 |
| LAPLACE|answerable90|unres=1|form=0|re=q+pack | 4320 | 13 | -1580.645 | True | 2.5e-06 | 3.2 | 3.123 | 0.918 |
| LAPLACE|answerable90|unres=1|form=1|re=q+pack | 4320 | 28 | -1574.142 | True | 2.8e-06 | 3.1 | 3.140 | 0.926 |
| LAPLACE|answerable90|unres=1|form=2|re=q+pack | 4320 | 48 | -1564.824 | True | 7.6e-06 | 0.61 | 3.165 | 0.937 |
| LAPLACE|all96|unres=0|form=0|re=q+pack | 4608 | 13 | -1594.064 | True | 2.4e-06 | 3 | 3.508 | 0.944 |
| LAPLACE|all96|unres=0|form=1|re=q+pack | 4608 | 28 | -1588.288 | True | 3.0e-06 | 3 | 3.526 | 0.952 |
| LAPLACE|all96|unres=0|form=2|re=q+pack | 4608 | 48 | -1581.631 | True | 5.5e-06 | 0.6 | 3.544 | 0.959 |
| LAPLACE|all96|unres=1|form=0|re=q+pack | 4608 | 13 | -1623.147 | True | 8.6e-06 | 3.3 | 3.282 | 0.897 |
| LAPLACE|all96|unres=1|form=1|re=q+pack | 4608 | 28 | -1617.190 | True | 7.4e-06 | 3.2 | 3.298 | 0.904 |
| LAPLACE|all96|unres=1|form=2|re=q+pack | 4608 | 48 | -1608.428 | True | 5.2e-06 | 0.63 | 3.321 | 0.915 |

### Contrasts: Laplace, question + pack intercepts, answerable90, form 0 (arm + reader)

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +0.747 (0.273); OR 2.11 [1.24, 3.60]; p=0.006 | +0.701 (0.262); OR 2.02 [1.21, 3.37]; p=0.008 |
| LEX5B_6000 vs HR_6000 | +0.474 (0.271); OR 1.61 [0.94, 2.73]; p=0.080 | +0.342 (0.261); OR 1.41 [0.84, 2.35]; p=0.191 |
| HR_6000 vs HYBRID_6000 | +0.273 (0.268); OR 1.31 [0.78, 2.22]; p=0.308 | +0.360 (0.258); OR 1.43 [0.86, 2.37]; p=0.163 |
| SETR_6000 vs LEX5B_6000 | -0.284 (0.272); OR 0.75 [0.44, 1.28]; p=0.297 | -0.264 (0.263); OR 0.77 [0.46, 1.29]; p=0.315 |
| SETR_6000 vs HYBRID_6000 | +0.463 (0.270); OR 1.59 [0.94, 2.70]; p=0.086 | +0.438 (0.259); OR 1.55 [0.93, 2.58]; p=0.091 |
| budget HYBRID (15500-6000) | +0.573 (0.271); OR 1.77 [1.04, 3.01]; p=0.034 | +0.622 (0.261); OR 1.86 [1.12, 3.11]; p=0.017 |
| budget HR (15500-6000) | +0.474 (0.267); OR 1.61 [0.95, 2.71]; p=0.076 | +0.360 (0.259); OR 1.43 [0.86, 2.38]; p=0.164 |
| budget LEX5B (15500-6000) | +0.125 (0.275); OR 1.13 [0.66, 1.94]; p=0.649 | +0.180 (0.265); OR 1.20 [0.71, 2.01]; p=0.498 |
| budget SETR (15500-6000) | +0.076 (0.270); OR 1.08 [0.64, 1.83]; p=0.779 | +0.050 (0.260); OR 1.05 [0.63, 1.75]; p=0.848 |

Population-averaged accuracy (%, mean over readers): unres=0: HYBRID_6000 62.0, HR_6000 64.6, LEX5B_6000 68.9, SETR_6000 66.3, HYBRID_15500 67.4, HR_15500 68.9, LEX5B_15500 70.0, SETR_15500 67.0. unres=1: HYBRID_6000 63.1, HR_6000 66.7, LEX5B_6000 69.9, SETR_6000 67.4, HYBRID_15500 69.2, HR_15500 70.1, LEX5B_15500 71.6, SETR_15500 67.9.

### Contrasts: Laplace, question + pack intercepts, answerable90, form 1 (arm + reader + pipeline:reader; arm effects averaged over readers (sum coding))

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +0.743 (0.275); OR 2.10 [1.23, 3.60]; p=0.007 | +0.699 (0.264); OR 2.01 [1.20, 3.37]; p=0.008 |
| LEX5B_6000 vs HR_6000 | +0.457 (0.273); OR 1.58 [0.92, 2.70]; p=0.095 | +0.321 (0.263); OR 1.38 [0.82, 2.31]; p=0.223 |
| HR_6000 vs HYBRID_6000 | +0.286 (0.270); OR 1.33 [0.78, 2.26]; p=0.289 | +0.378 (0.260); OR 1.46 [0.88, 2.43]; p=0.145 |
| SETR_6000 vs LEX5B_6000 | -0.278 (0.274); OR 0.76 [0.44, 1.30]; p=0.310 | -0.262 (0.264); OR 0.77 [0.46, 1.29]; p=0.322 |
| SETR_6000 vs HYBRID_6000 | +0.465 (0.271); OR 1.59 [0.94, 2.71]; p=0.087 | +0.437 (0.261); OR 1.55 [0.93, 2.58]; p=0.094 |
| budget HYBRID (15500-6000) | +0.576 (0.272); OR 1.78 [1.04, 3.03]; p=0.034 | +0.624 (0.262); OR 1.87 [1.12, 3.12]; p=0.017 |
| budget HR (15500-6000) | +0.480 (0.269); OR 1.62 [0.95, 2.74]; p=0.075 | +0.367 (0.260); OR 1.44 [0.87, 2.40]; p=0.159 |
| budget LEX5B (15500-6000) | +0.124 (0.276); OR 1.13 [0.66, 1.95]; p=0.653 | +0.180 (0.266); OR 1.20 [0.71, 2.02]; p=0.500 |
| budget SETR (15500-6000) | +0.076 (0.271); OR 1.08 [0.63, 1.84]; p=0.780 | +0.050 (0.261); OR 1.05 [0.63, 1.75]; p=0.849 |

Population-averaged accuracy (%, mean over readers): unres=0: HYBRID_6000 61.9, HR_6000 64.6, LEX5B_6000 68.9, SETR_6000 66.3, HYBRID_15500 67.3, HR_15500 69.0, LEX5B_15500 70.0, SETR_15500 67.0. unres=1: HYBRID_6000 63.1, HR_6000 66.7, LEX5B_6000 69.9, SETR_6000 67.4, HYBRID_15500 69.2, HR_15500 70.2, LEX5B_15500 71.6, SETR_15500 67.9.

### Contrasts: Laplace, question + pack intercepts, all96, form 0 (arm + reader)

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +0.726 (0.266); OR 2.07 [1.23, 3.49]; p=0.006 | +0.683 (0.256); OR 1.98 [1.20, 3.27]; p=0.008 |
| LEX5B_6000 vs HR_6000 | +0.445 (0.265); OR 1.56 [0.93, 2.62]; p=0.093 | +0.317 (0.255); OR 1.37 [0.83, 2.26]; p=0.215 |
| HR_6000 vs HYBRID_6000 | +0.282 (0.262); OR 1.33 [0.79, 2.21]; p=0.281 | +0.366 (0.252); OR 1.44 [0.88, 2.36]; p=0.146 |
| SETR_6000 vs LEX5B_6000 | -0.276 (0.266); OR 0.76 [0.45, 1.28]; p=0.300 | -0.256 (0.257); OR 0.77 [0.47, 1.28]; p=0.318 |
| SETR_6000 vs HYBRID_6000 | +0.451 (0.263); OR 1.57 [0.94, 2.63]; p=0.087 | +0.427 (0.253); OR 1.53 [0.93, 2.52]; p=0.092 |
| budget HYBRID (15500-6000) | +0.559 (0.264); OR 1.75 [1.04, 2.93]; p=0.034 | +0.607 (0.255); OR 1.84 [1.11, 3.03]; p=0.017 |
| budget HR (15500-6000) | +0.464 (0.262); OR 1.59 [0.95, 2.66]; p=0.076 | +0.354 (0.253); OR 1.42 [0.87, 2.34]; p=0.162 |
| budget LEX5B (15500-6000) | +0.123 (0.269); OR 1.13 [0.67, 1.91]; p=0.647 | +0.176 (0.259); OR 1.19 [0.72, 1.98]; p=0.497 |
| budget SETR (15500-6000) | +0.075 (0.264); OR 1.08 [0.64, 1.81]; p=0.777 | +0.050 (0.254); OR 1.05 [0.64, 1.73]; p=0.845 |

Population-averaged accuracy (%, mean over readers): unres=0: HYBRID_6000 64.3, HR_6000 66.9, LEX5B_6000 70.7, SETR_6000 68.4, HYBRID_15500 69.3, HR_15500 70.9, LEX5B_15500 71.8, SETR_15500 69.0. unres=1: HYBRID_6000 65.4, HR_6000 68.8, LEX5B_6000 71.7, SETR_6000 69.4, HYBRID_15500 71.0, HR_15500 72.0, LEX5B_15500 73.2, SETR_15500 69.8.

### Contrasts: Laplace, question + pack intercepts, all96, form 1 (arm + reader + pipeline:reader; arm effects averaged over readers (sum coding))

| contrast | unresolved = wrong | unresolved = right |
|---|---|---|
| LEX5B_6000 vs HYBRID_6000 | +0.721 (0.268); OR 2.06 [1.22, 3.48]; p=0.007 | +0.679 (0.258); OR 1.97 [1.19, 3.27]; p=0.008 |
| LEX5B_6000 vs HR_6000 | +0.427 (0.267); OR 1.53 [0.91, 2.59]; p=0.110 | +0.295 (0.257); OR 1.34 [0.81, 2.22]; p=0.251 |
| HR_6000 vs HYBRID_6000 | +0.294 (0.264); OR 1.34 [0.80, 2.25]; p=0.265 | +0.383 (0.254); OR 1.47 [0.89, 2.41]; p=0.131 |
| SETR_6000 vs LEX5B_6000 | -0.270 (0.268); OR 0.76 [0.45, 1.29]; p=0.313 | -0.254 (0.258); OR 0.78 [0.47, 1.29]; p=0.326 |
| SETR_6000 vs HYBRID_6000 | +0.450 (0.265); OR 1.57 [0.93, 2.64]; p=0.089 | +0.425 (0.255); OR 1.53 [0.93, 2.52]; p=0.096 |
| budget HYBRID (15500-6000) | +0.562 (0.266); OR 1.75 [1.04, 2.95]; p=0.034 | +0.610 (0.256); OR 1.84 [1.11, 3.04]; p=0.017 |
| budget HR (15500-6000) | +0.469 (0.263); OR 1.60 [0.95, 2.68]; p=0.075 | +0.360 (0.255); OR 1.43 [0.87, 2.36]; p=0.158 |
| budget LEX5B (15500-6000) | +0.122 (0.270); OR 1.13 [0.67, 1.92]; p=0.651 | +0.176 (0.260); OR 1.19 [0.72, 1.99]; p=0.499 |
| budget SETR (15500-6000) | +0.075 (0.265); OR 1.08 [0.64, 1.81]; p=0.778 | +0.049 (0.256); OR 1.05 [0.64, 1.73]; p=0.847 |

Population-averaged accuracy (%, mean over readers): unres=0: HYBRID_6000 64.3, HR_6000 66.9, LEX5B_6000 70.7, SETR_6000 68.4, HYBRID_15500 69.3, HR_15500 70.9, LEX5B_15500 71.7, SETR_15500 69.0. unres=1: HYBRID_6000 65.4, HR_6000 68.9, LEX5B_6000 71.6, SETR_6000 69.4, HYBRID_15500 71.0, HR_15500 72.1, LEX5B_15500 73.2, SETR_15500 69.8.

### LRTs (question + pack model)

| test | stat | df | p |
|---|---|---|---|
| LAPLACE|answerable90|unres=0|re=q+pack: form1 vs form0 (pipeline:reader) | 12.404 | 15 | 0.648 |
| LAPLACE|answerable90|unres=0|re=q+pack: form2 vs form1 (budget:reader + pipeline:budget:reader) | 14.393 | 20 | 0.81 |
| LAPLACE|answerable90|unres=0|re=q+pack: form2 vs form0 (all reader interactions) | 26.797 | 35 | 0.838 |
| LAPLACE|answerable90|unres=0|form=0: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 84.014 | 1 | 2.46e-20 |
| LAPLACE|answerable90|unres=0|form=1: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 85.257 | 1 | 1.31e-20 |
| LAPLACE|answerable90|unres=0|form=2: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 86.349 | 1 | 7.54e-21 |
| LAPLACE|answerable90|unres=1|re=q+pack: form1 vs form0 (pipeline:reader) | 13.006 | 15 | 0.602 |
| LAPLACE|answerable90|unres=1|re=q+pack: form2 vs form1 (budget:reader + pipeline:budget:reader) | 18.636 | 20 | 0.546 |
| LAPLACE|answerable90|unres=1|re=q+pack: form2 vs form0 (all reader interactions) | 31.642 | 35 | 0.631 |
| LAPLACE|answerable90|unres=1|form=0: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 75.433 | 1 | 1.89e-18 |
| LAPLACE|answerable90|unres=1|form=1: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 76.458 | 1 | 1.12e-18 |
| LAPLACE|answerable90|unres=1|form=2: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 78.075 | 1 | 4.96e-19 |
| LAPLACE|all96|unres=0|re=q+pack: form1 vs form0 (pipeline:reader) | 11.553 | 15 | 0.712 |
| LAPLACE|all96|unres=0|re=q+pack: form2 vs form1 (budget:reader + pipeline:budget:reader) | 13.314 | 20 | 0.863 |
| LAPLACE|all96|unres=0|re=q+pack: form2 vs form0 (all reader interactions) | 24.868 | 35 | 0.898 |
| LAPLACE|all96|unres=0|form=0: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 80.066 | 1 | 1.81e-19 |
| LAPLACE|all96|unres=0|form=1: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 81.242 | 1 | 9.98e-20 |
| LAPLACE|all96|unres=0|form=2: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 82.211 | 1 | 6.11e-20 |
| LAPLACE|all96|unres=1|re=q+pack: form1 vs form0 (pipeline:reader) | 11.914 | 15 | 0.686 |
| LAPLACE|all96|unres=1|re=q+pack: form2 vs form1 (budget:reader + pipeline:budget:reader) | 17.524 | 20 | 0.619 |
| LAPLACE|all96|unres=1|re=q+pack: form2 vs form0 (all reader interactions) | 29.438 | 35 | 0.733 |
| LAPLACE|all96|unres=1|form=0: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 71.712 | 1 | 1.25e-17 |
| LAPLACE|all96|unres=1|form=1: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 72.651 | 1 | 7.74e-18 |
| LAPLACE|all96|unres=1|form=2: re=q+pack vs re=q (pack variance; boundary, chi2(1)/2) | 74.149 | 1 | 3.62e-18 |

### Per-reader LEX5B-HR @6000 log-odds ratio implied by form 1 (question + pack, answerable90)

| reader | unres=0 | unres=1 |
|---|---|---|
| glm-5.3-flash | +0.019 (0.444) | -0.205 (0.435) |
| gpt-oss-20b | +0.275 (0.406) | +0.188 (0.397) |
| qwen3-235b-a22b-2507 | +0.918 (0.426) | +0.883 (0.418) |
| llama-3.3-70b-instruct | +0.529 (0.418) | +0.315 (0.407) |
| gemma-3-27b-it | +0.834 (0.399) | +0.729 (0.389) |
| deepseek-v4-flash | +0.167 (0.433) | +0.016 (0.422) |

## TOST / non-inferiority, LEX5B vs HR @6000 (GLM reader), margins post hoc

### machine LEX5B_6000 vs HR_6000 (net -3..+0 of 90)

| bootstrap | 90% CI pp (union over extremes) | 90% CI points of 90 | smallest equiv. margin (pts) | smallest NI margin (pts) |
|---|---|---|---|---|
| component | [-9.23, 6.32] | [-8.31, 5.68] | 8.31 | 8.31 |
| question | [-10.00, 6.67] | [-9.00, 6.00] | 9.00 | 9.00 |

| margin (pts / pp) | TOST p comp. | equivalent (comp.) | TOST p quest. | NI p comp. | non-inferior (comp.) | NI p quest. | cluster-t TOST p | cluster-t NI p |
|---|---|---|---|---|---|---|---|---|
| 3 / 3.33 | 0.5022 | False | 0.5465 | 0.5022 | False | 0.5465 | 0.5000 | 0.5000 |
| 5 / 5.56 | 0.2715 | False | 0.3476 | 0.2715 | False | 0.3476 | 0.2749 | 0.2749 |
| 8 / 8.89 | 0.0617 | False | 0.1204 | 0.0617 | False | 0.1204 | 0.0722 | 0.0722 |

### human_sym LEX5B_6000 vs HR_6000 (net +0..+0 of 90)

| bootstrap | 90% CI pp (union over extremes) | 90% CI points of 90 | smallest equiv. margin (pts) | smallest NI margin (pts) |
|---|---|---|---|---|
| component | [-6.12, 6.84] | [-5.51, 6.15] | 6.15 | 5.51 |
| question | [-5.56, 5.56] | [-5.00, 5.00] | 5.00 | 5.00 |

| margin (pts / pp) | TOST p comp. | equivalent (comp.) | TOST p quest. | NI p comp. | non-inferior (comp.) | NI p quest. | cluster-t TOST p | cluster-t NI p |
|---|---|---|---|---|---|---|---|---|
| 3 / 3.33 | 0.2090 | False | 0.2193 | 0.1904 | False | 0.2046 | 0.2046 | 0.2046 |
| 5 / 5.56 | 0.0920 | False | 0.0776 | 0.0681 | False | 0.0721 | 0.0878 | 0.0878 |
| 8 / 8.89 | 0.0170 | True | 0.0081 | 0.0043 | True | 0.0066 | 0.0182 | 0.0182 |

## Minimum detectable effect (paired, exact two-sided sign test, alpha .05, power .80, n=90)

| comparison | discordance basis | psi | MDE exact (pts of 90 / pp) | MDE normal (pts / pp) | MDE exact x sqrt(deff) (pts) | power for 5/90 at n=90 | n for 5/90 exact | n normal | n normal x deff |
|---|---|---|---|---|---|---|---|---|---|
| LEX5B_6000 vs HYBRID_6000 (b/w 11/4 of 88 resolved; deff 1.25) | resolved | 0.170 | 11.12 / 12.35 | 10.83 / 12.03 | 12.41 | 0.176 | 458 | 431 | 537 |
| LEX5B_6000 vs HYBRID_6000 (b/w 11/4 of 88 resolved; deff 1.25) | min_over_assignments | 0.167 | 10.99 / 12.21 | 10.71 / 11.90 | 12.27 | 0.178 | 448 | 421 | 525 |
| LEX5B_6000 vs HYBRID_6000 (b/w 11/4 of 88 resolved; deff 1.25) | max_over_assignments | 0.189 | 11.72 / 13.02 | 11.40 / 12.67 | 13.08 | 0.163 | 506 | 478 | 596 |
| LEX5B_6000 vs HR_6000 (b/w 6/7 of 87 resolved; deff 0.72) | resolved | 0.149 | 10.38 / 11.53 | 10.14 / 11.27 | 8.80 | 0.192 | 401 | 378 | 271 |
| LEX5B_6000 vs HR_6000 (b/w 6/7 of 87 resolved; deff 0.72) | min_over_assignments | 0.144 | 10.19 / 11.32 | 9.97 / 11.08 | 8.64 | 0.197 | 387 | 365 | 262 |
| LEX5B_6000 vs HR_6000 (b/w 6/7 of 87 resolved; deff 0.72) | max_over_assignments | 0.178 | 11.36 / 12.62 | 11.06 / 12.29 | 9.63 | 0.170 | 478 | 450 | 323 |

## Multiplicity across readers (tables/readers.tex); raw and adjusted p as min-max over unresolved assignments

| reader | comparison | net | raw p | Holm (6 readers) | BH (6 readers) | Holm (24 LEX-vs-control) | BH (24) | Holm (48) | BH (48) |
|---|---|---|---|---|---|---|---|---|---|
| glm-5.3-flash | LEX5B_6000 vs HYBRID_6000 | +5..+7 | 0.1185-0.3323 | 0.474-1.000 | 0.237-0.431 | 1.000-1.000 | 0.539-1.000 | 1.000-1.000 | 0.507-0.997 |
| glm-5.3-flash | LEX5B_6000 vs HR_6000 | -3..+0 | 0.6072-1.0000 | 1.000-1.000 | 0.678-1.000 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.834-1.000 |
| glm-5.3-flash | LEX5B_15500 vs HYBRID_15500 | -1..-1 | 1.0000-1.0000 | 1.000-1.000 | 1.000-1.000 | 1.000-1.000 | 1.000-1.000 | 1.000-1.000 | 1.000-1.000 |
| glm-5.3-flash | LEX5B_15500 vs HR_15500 | -3..-2 | 0.6291-0.8036 | 1.000-1.000 | 0.832-1.000 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.834-1.000 |
| glm-5.3-flash | HYBRID_15500 vs HYBRID_6000 | +5..+6 | 0.1796-0.3018 | 0.592-0.905 | 0.252-0.453 | - | - | 1.000-1.000 | 0.507-0.997 |
| glm-5.3-flash | HR_15500 vs HR_6000 | -2..+1 | 0.7905-1.0000 | 1.000-1.000 | 0.839-1.000 | - | - | 1.000-1.000 | 0.895-1.000 |
| glm-5.3-flash | interaction vs HYBRID | +6..+8 | 0.1523-0.3286 | 0.761-1.000 | 0.457-0.986 | - | - | 1.000-1.000 | 0.507-0.997 |
| glm-5.3-flash | interaction vs HR | -1..+3 | 0.6636-1.0000 | 1.000-1.000 | 0.872-1.000 | - | - | 1.000-1.000 | 0.834-1.000 |
| gpt-oss-20b | LEX5B_6000 vs HYBRID_6000 | +10..+11 | 0.0266-0.0414 | 0.160-0.248 | 0.160-0.248 | 0.638-0.993 | 0.523-0.993 | 1.000-1.000 | 0.422-0.662 |
| gpt-oss-20b | LEX5B_6000 vs HR_6000 | +1..+4 | 0.5572-1.0000 | 1.000-1.000 | 0.678-1.000 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.834-1.000 |
| gpt-oss-20b | LEX5B_15500 vs HYBRID_15500 | +1..+3 | 0.6072-1.0000 | 1.000-1.000 | 0.777-1.000 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.834-1.000 |
| gpt-oss-20b | LEX5B_15500 vs HR_15500 | -2..-1 | 0.8036-1.0000 | 1.000-1.000 | 0.832-1.000 | 1.000-1.000 | 0.868-1.000 | 1.000-1.000 | 0.895-1.000 |
| gpt-oss-20b | HYBRID_15500 vs HYBRID_6000 | +6..+7 | 0.1185-0.1796 | 0.592-0.718 | 0.252-0.359 | - | - | 1.000-1.000 | 0.507-0.997 |
| gpt-oss-20b | HR_15500 vs HR_6000 | +1..+3 | 0.6072-1.0000 | 1.000-1.000 | 0.839-1.000 | - | - | 1.000-1.000 | 0.834-1.000 |
| gpt-oss-20b | interaction vs HYBRID | +7..+10 | 0.1105-0.2831 | 0.663-1.000 | 0.457-0.986 | - | - | 1.000-1.000 | 0.507-0.997 |
| gpt-oss-20b | interaction vs HR | +2..+6 | 0.3449-0.8450 | 1.000-1.000 | 0.759-1.000 | - | - | 1.000-1.000 | 0.690-1.000 |
| qwen3-235b-a22b-2507 | LEX5B_6000 vs HYBRID_6000 | +5..+5 | 0.3593-0.3593 | 0.719-1.000 | 0.431-0.431 | 1.000-1.000 | 0.719-1.000 | 1.000-1.000 | 0.690-1.000 |
| qwen3-235b-a22b-2507 | LEX5B_6000 vs HR_6000 | +3..+5 | 0.3593-0.6476 | 1.000-1.000 | 0.678-1.000 | 1.000-1.000 | 0.719-1.000 | 1.000-1.000 | 0.690-1.000 |
| qwen3-235b-a22b-2507 | LEX5B_15500 vs HYBRID_15500 | +2..+7 | 0.1435-0.7905 | 0.861-1.000 | 0.777-1.000 | 1.000-1.000 | 0.539-1.000 | 1.000-1.000 | 0.507-1.000 |
| qwen3-235b-a22b-2507 | LEX5B_15500 vs HR_15500 | +6..+9 | 0.0490-0.2101 | 0.294-1.000 | 0.294-0.630 | 1.000-1.000 | 0.523-1.000 | 1.000-1.000 | 0.471-0.997 |
| qwen3-235b-a22b-2507 | HYBRID_15500 vs HYBRID_6000 | +7..+9 | 0.0352-0.1185 | 0.211-0.711 | 0.211-0.359 | - | - | 1.000-1.000 | 0.422-0.997 |
| qwen3-235b-a22b-2507 | HR_15500 vs HR_6000 | +3..+5 | 0.3323-0.6291 | 1.000-1.000 | 0.665-1.000 | - | - | 1.000-1.000 | 0.690-1.000 |
| qwen3-235b-a22b-2507 | interaction vs HYBRID | -2..+3 | 0.7313-1.0000 | 1.000-1.000 | 0.878-1.000 | - | - | 1.000-1.000 | 0.878-1.000 |
| qwen3-235b-a22b-2507 | interaction vs HR | -6..-1 | 0.3794-1.0000 | 1.000-1.000 | 0.759-1.000 | - | - | 1.000-1.000 | 0.700-1.000 |
| llama-3.3-70b-instruct | LEX5B_6000 vs HYBRID_6000 | +4..+4 | 0.5034-0.5034 | 0.719-1.000 | 0.503-0.503 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.806-1.000 |
| llama-3.3-70b-instruct | LEX5B_6000 vs HR_6000 | +2..+3 | 0.6776-0.8388 | 1.000-1.000 | 0.678-1.000 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.834-1.000 |
| llama-3.3-70b-instruct | LEX5B_15500 vs HYBRID_15500 | +3..+3 | 0.6476-0.6476 | 1.000-1.000 | 0.777-1.000 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.834-1.000 |
| llama-3.3-70b-instruct | LEX5B_15500 vs HR_15500 | +1..+2 | 0.8318-1.0000 | 1.000-1.000 | 0.832-1.000 | 1.000-1.000 | 0.868-1.000 | 1.000-1.000 | 0.895-1.000 |
| llama-3.3-70b-instruct | HYBRID_15500 vs HYBRID_6000 | +7..+7 | 0.1435-0.1435 | 0.592-0.717 | 0.252-0.359 | - | - | 1.000-1.000 | 0.507-0.997 |
| llama-3.3-70b-instruct | HR_15500 vs HR_6000 | +6..+8 | 0.0768-0.2101 | 0.384-1.000 | 0.230-0.630 | - | - | 1.000-1.000 | 0.507-0.997 |
| llama-3.3-70b-instruct | interaction vs HYBRID | +1..+1 | 1.0000-1.0000 | 1.000-1.000 | 1.000-1.000 | - | - | 1.000-1.000 | 1.000-1.000 |
| llama-3.3-70b-instruct | interaction vs HR | +0..+2 | 0.8720-1.0000 | 1.000-1.000 | 0.872-1.000 | - | - | 1.000-1.000 | 0.910-1.000 |
| gemma-3-27b-it | LEX5B_6000 vs HYBRID_6000 | +6..+7 | 0.1671-0.2632 | 0.501-1.000 | 0.251-0.431 | 1.000-1.000 | 0.539-1.000 | 1.000-1.000 | 0.507-0.997 |
| gemma-3-27b-it | LEX5B_6000 vs HR_6000 | +7..+7 | 0.2478-0.2478 | 1.000-1.000 | 0.678-1.000 | 1.000-1.000 | 0.661-1.000 | 1.000-1.000 | 0.626-0.997 |
| gemma-3-27b-it | LEX5B_15500 vs HYBRID_15500 | +0..+3 | 0.6476-1.0000 | 1.000-1.000 | 0.777-1.000 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.834-1.000 |
| gemma-3-27b-it | LEX5B_15500 vs HR_15500 | +4..+6 | 0.3075-0.5572 | 1.000-1.000 | 0.615-1.000 | 1.000-1.000 | 0.719-1.000 | 1.000-1.000 | 0.690-1.000 |
| gemma-3-27b-it | HYBRID_15500 vs HYBRID_6000 | +3..+6 | 0.2101-0.6072 | 0.592-1.000 | 0.252-0.729 | - | - | 1.000-1.000 | 0.560-1.000 |
| gemma-3-27b-it | HR_15500 vs HR_6000 | +1..+2 | 0.8388-1.0000 | 1.000-1.000 | 0.839-1.000 | - | - | 1.000-1.000 | 0.895-1.000 |
| gemma-3-27b-it | interaction vs HYBRID | +3..+7 | 0.2672-0.7132 | 1.000-1.000 | 0.534-1.000 | - | - | 1.000-1.000 | 0.641-1.000 |
| gemma-3-27b-it | interaction vs HR | +1..+3 | 0.7624-1.0000 | 1.000-1.000 | 0.872-1.000 | - | - | 1.000-1.000 | 0.893-1.000 |
| deepseek-v4-flash | LEX5B_6000 vs HYBRID_6000 | +5..+7 | 0.0654-0.2266 | 0.327-1.000 | 0.196-0.431 | 1.000-1.000 | 0.523-1.000 | 1.000-1.000 | 0.507-0.997 |
| deepseek-v4-flash | LEX5B_6000 vs HR_6000 | +4..+6 | 0.1796-0.4545 | 1.000-1.000 | 0.678-1.000 | 1.000-1.000 | 0.539-1.000 | 1.000-1.000 | 0.507-1.000 |
| deepseek-v4-flash | LEX5B_15500 vs HYBRID_15500 | +2..+4 | 0.4545-0.8036 | 1.000-1.000 | 0.777-1.000 | 1.000-1.000 | 0.774-1.000 | 1.000-1.000 | 0.766-1.000 |
| deepseek-v4-flash | LEX5B_15500 vs HR_15500 | -7..-6 | 0.1185-0.2101 | 0.592-1.000 | 0.355-0.630 | 1.000-1.000 | 0.539-1.000 | 1.000-1.000 | 0.507-0.997 |
| deepseek-v4-flash | HYBRID_15500 vs HYBRID_6000 | +1..+4 | 0.4545-1.0000 | 0.592-1.000 | 0.454-1.000 | - | - | 1.000-1.000 | 0.766-1.000 |
| deepseek-v4-flash | HR_15500 vs HR_6000 | +10..+12 | 0.0005-0.0063 | 0.003-0.038 | 0.003-0.038 | - | - | 0.023-0.305 | 0.023-0.305 |
| deepseek-v4-flash | interaction vs HYBRID | +1..+5 | 0.4627-1.0000 | 1.000-1.000 | 0.694-1.000 | - | - | 1.000-1.000 | 0.766-1.000 |
| deepseek-v4-flash | interaction vs HR | +10..+13 | 0.0065-0.0405 | 0.039-0.243 | 0.039-0.243 | - | - | 0.306-1.000 | 0.156-0.662 |
