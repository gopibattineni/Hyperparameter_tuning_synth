# Bayesian sampler comparison (Cancer & Alzheimer's)

Objective is **maximized**: `0.4×Fidelity + 0.4×Utility − 0.2×PrivacyRisk`.

## Scope
- Datasets: cancer, alzheimers
- Generators: gaussian_copula, copulagan, ctgan, ctabgan, tvae, wgan_gp, forest_diffusion, tabddpm
- Samplers: Standard TPE (baseline), Multivariate TPE, Gaussian Process (GPSampler)

## Pruning note
The shared `TrialObjective` does **not** call `trial.report()` / `should_prune()`.
Configured `MedianPruner` therefore cannot prune on intermediate metrics.
Generator exceptions are recorded as Optuna `PRUNED` trials via `TrialPruned`.

## GPSampler compatibility
Optuna 5.0 GPSampler requires `with torch.device(...):` (PyTorch ≥ 2.1).
This environment uses torch 1.13.x; `gaussian_process/run.py` applies a
context-manager shim so GPSampler still runs (not a sampler substitute).

## Aggregate best objective (higher is better)

|                                    |   Gaussian Process |   Multivariate TPE |   Standard TPE |
|:-----------------------------------|-------------------:|-------------------:|---------------:|
| ('alzheimers', 'copulagan')        |           0.597712 |           0.619537 |       0.56853  |
| ('alzheimers', 'ctabgan')          |           0.203241 |           0.214048 |       0.561733 |
| ('alzheimers', 'ctgan')            |           0.663467 |           0.636579 |       0.627764 |
| ('alzheimers', 'forest_diffusion') |           0.618762 |           0.618009 |       0.237894 |
| ('alzheimers', 'gaussian_copula')  |           0.589056 |           0.589056 |       0.593771 |
| ('alzheimers', 'tabddpm')          |           0.55223  |           0.56329  |       0.55223  |
| ('alzheimers', 'tvae')             |           0.631754 |           0.637642 |       0.652424 |
| ('alzheimers', 'wgan_gp')          |           0.623352 |           0.623494 |       0.233268 |
| ('cancer', 'copulagan')            |           0.613039 |           0.597223 |       0.597583 |
| ('cancer', 'ctabgan')              |           0.52668  |           0.522526 |       0.550884 |
| ('cancer', 'ctgan')                |           0.636524 |           0.63778  |       0.6349   |
| ('cancer', 'forest_diffusion')     |           0.661983 |           0.661686 |       0.317402 |
| ('cancer', 'gaussian_copula')      |           0.568597 |           0.568597 |       0.506908 |
| ('cancer', 'tabddpm')              |           0.543344 |           0.544518 |       0.550037 |
| ('cancer', 'tvae')                 |           0.642463 |           0.654602 |       0.653185 |
| ('cancer', 'wgan_gp')              |           0.680187 |           0.683819 |       0.306408 |

## Aggregate runtime (seconds)

|                                    |   Gaussian Process |   Multivariate TPE |   Standard TPE |
|:-----------------------------------|-------------------:|-------------------:|---------------:|
| ('alzheimers', 'copulagan')        |           2775.22  |           2999.48  |       611.248  |
| ('alzheimers', 'ctabgan')          |            199.058 |            892.007 |      1010.6    |
| ('alzheimers', 'ctgan')            |          12015.9   |           9894.55  |      1673.97   |
| ('alzheimers', 'forest_diffusion') |          18245.2   |          15860.7   |      2185.23   |
| ('alzheimers', 'gaussian_copula')  |            237.06  |            201.325 |        19.2215 |
| ('alzheimers', 'tabddpm')          |           1229.59  |           1326.47  |      1059.81   |
| ('alzheimers', 'tvae')             |           1237.1   |            948.609 |       204.902  |
| ('alzheimers', 'wgan_gp')          |           1554.71  |           1410.81  |       405.925  |
| ('cancer', 'copulagan')            |           1413.84  |            238.928 |      1034.75   |
| ('cancer', 'ctabgan')              |            773.764 |            733.815 |      2705.41   |
| ('cancer', 'ctgan')                |           6790.75  |           5853.28  |      2471.52   |
| ('cancer', 'forest_diffusion')     |          11473.7   |           7562.79  |     12256.5    |
| ('cancer', 'gaussian_copula')      |            464.734 |            427.5   |        30.0227 |
| ('cancer', 'tabddpm')              |           1586.7   |           1520.09  |      1126.3    |
| ('cancer', 'tvae')                 |           2287.05  |           2582.11  |      1250.54   |
| ('cancer', 'wgan_gp')              |           4456.46  |           4096.79  |      1526.05   |

## Headlines
- **cancer/gaussian_copula**: best objective = 0.5686 (Multivariate TPE); fastest wall-clock = 30.0s (Standard TPE)
- **cancer/copulagan**: best objective = 0.6130 (Gaussian Process); fastest wall-clock = 238.9s (Multivariate TPE)
- **cancer/ctgan**: best objective = 0.6378 (Multivariate TPE); fastest wall-clock = 2471.5s (Standard TPE)
- **cancer/ctabgan**: best objective = 0.5509 (Standard TPE); fastest wall-clock = 733.8s (Multivariate TPE)
- **cancer/tvae**: best objective = 0.6546 (Multivariate TPE); fastest wall-clock = 1250.5s (Standard TPE)
- **cancer/wgan_gp**: best objective = 0.6838 (Multivariate TPE); fastest wall-clock = 1526.1s (Standard TPE)
- **cancer/forest_diffusion**: best objective = 0.6620 (Gaussian Process); fastest wall-clock = 7562.8s (Multivariate TPE)
- **cancer/tabddpm**: best objective = 0.5500 (Standard TPE); fastest wall-clock = 1126.3s (Standard TPE)
- **alzheimers/gaussian_copula**: best objective = 0.5938 (Standard TPE); fastest wall-clock = 19.2s (Standard TPE)
- **alzheimers/copulagan**: best objective = 0.6195 (Multivariate TPE); fastest wall-clock = 611.2s (Standard TPE)
- **alzheimers/ctgan**: best objective = 0.6635 (Gaussian Process); fastest wall-clock = 1674.0s (Standard TPE)
- **alzheimers/ctabgan**: best objective = 0.5617 (Standard TPE); fastest wall-clock = 199.1s (Gaussian Process)
- **alzheimers/tvae**: best objective = 0.6524 (Standard TPE); fastest wall-clock = 204.9s (Standard TPE)
- **alzheimers/wgan_gp**: best objective = 0.6235 (Multivariate TPE); fastest wall-clock = 405.9s (Standard TPE)
- **alzheimers/forest_diffusion**: best objective = 0.6188 (Gaussian Process); fastest wall-clock = 2185.2s (Standard TPE)
- **alzheimers/tabddpm**: best objective = 0.5633 (Multivariate TPE); fastest wall-clock = 1059.8s (Standard TPE)
