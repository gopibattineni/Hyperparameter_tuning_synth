# Optuna HPO for synthetic tabular generators

Find hyperparameters that produce high-quality synthetic tabular data across multiple generators and datasets.

**Full project documentation:** [DOCUMENTATION.md](DOCUMENTATION.md)

## Layout

```text
Hyperparameter_tuning_synth/
├── config/                 # YAML configs + generator search spaces
├── generators/             # BaseGenerator + wrappers
├── datasets/               # DatasetSpec + loader
├── evaluation/             # fidelity / privacy / utility / objective
├── tuning/                 # Optuna objective, optimizer, artifacts, HPO reports
├── scripts/                # CLI entry points + multi-GPU helpers
├── results/                # run outputs + Excel/notation report packs
├── config_utils.py
├── requirements-hpo.txt
├── DOCUMENTATION.md
└── README.md
```

## Objective

```text
0.4 × Fidelity + 0.4 × Utility − 0.2 × PrivacyRisk   (maximize)
```

## Generators (7)

GaussianCopula · CopulaGAN · CTGAN · CTAB-GAN+ · TVAE · WGAN-GP · TabDDPM

## Datasets (15)

**Classification (9):** cancer, alzheimers, adult, forest_cover, bank_marketing, wine_quality, cdc_diabetes, mushroom, magic_gamma

**Regression (6):** metro_interstate, online_shopping, air_quality, concrete, energy_efficiency, real_estate

Full sweep: **15 × 7 = 105** dataset × generator pairs.

## Quick start

```bash
cd /home/gopi_b/Hyperparameter_tuning_synth
pip install -r requirements-hpo.txt

# Priority sweep
python scripts/run_experiments.py --datasets cancer alzheimers adult --n-trials 20 --resume

# Single pair
python scripts/run_tuning.py --generator ctgan --dataset cancer --n-trials 10

# Fill missing pairs across healthy GPUs (default: TabDDPM)
scripts/run_6gpu_parallel.sh start

# Rebuild Adult-style HPO Excel + notation for every dataset
scripts/write_all_8gen_reports.sh
```

Outputs land in `results/`:

- `all_experiments.csv` — master log
- `{classification|regression}/{n}_{dataset}/{generator}/best_params.json`, `metrics.json`, `best_synthetic.csv`, `trials.csv`, `completed.json`
- Report packs: `{dataset}_HPO_results.xlsx` + `{dataset}_HPO_notation.txt` (7 generators ranked)

## Quick check

```bash
python -c "import sys; sys.path.insert(0,'.'); \
from datasets import load_train_test, list_datasets; \
print(list_datasets()); \
train, test, spec = load_train_test('cancer'); \
print(spec.n_train, spec.n_test, spec.target)"
```
