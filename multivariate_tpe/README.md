# Multivariate TPE experiments

Optuna `TPESampler(multivariate=True)` HPO — same catalogue as Standard TPE.

## Catalogue

| Split | Count | Datasets |
|-------|------:|----------|
| Classification | **9** | cancer, alzheimers, adult, forest_cover, bank_marketing, wine_quality, cdc_diabetes, mushroom, magic_gamma |
| Regression | **6** | metro_interstate, online_shopping, air_quality, concrete, energy_efficiency, real_estate |
| Generators | **7** | gaussian_copula, copulagan, ctgan, ctabgan, tvae, wgan_gp, tabddpm *(no ForestDiffusion)* |

Defined in `datasets_catalog.py` and checked against `config/datasets.yaml` at runtime.

Results under `multivariate_tpe/results/` never overwrite `standard_tpe/results/`.

## Commands

```bash
cd /home/gopi_b/Hyperparameter_tuning_synth
export SYNTH_REPO_ROOT=/home/gopi_b/SYNTH_BENCHMARK
PY=/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python

# All 9 classification datasets (resume-safe)
$PY multivariate_tpe/run.py --classification-only

# All 6 regression datasets
$PY multivariate_tpe/run.py --regression-only

# Full 15
$PY multivariate_tpe/run.py --all-datasets

# Multi-GPU packed queue (default: classification 9, 7 gens)
$PY scripts/run_multivariate_tpe_packed.py start
$PY scripts/run_multivariate_tpe_packed.py status

# Include regression in the packed queue
CLASSIFICATION_ONLY=0 $PY scripts/run_multivariate_tpe_packed.py start
```

## Outputs

`results/{classification|regression}/{n}_{dataset}/{generator}/`
