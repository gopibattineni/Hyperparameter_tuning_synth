# Optuna HPO for synthetic tabular generators

All framework files live in this single folder:

`SYNTH_BENCHMARK/hyper parameter tuning/`

## Layout

```text
hyper parameter tuning/
├── config/                 # YAML configs + generator search spaces
├── generators/             # BaseGenerator + wrappers
├── datasets/               # DatasetSpec + loader
├── evaluation/             # fidelity / privacy / utility / objective
├── tuning/                 # Optuna objective, optimizer, artifacts
├── scripts/                # CLI entry points
├── results/                # run outputs
├── config_utils.py
├── requirements-hpo.txt
└── README.md
```

## Generators (8)

GaussianCopula · CopulaGAN · CTGAN · CTAB-GAN+ · TVAE · WGAN-GP · ForestDiffusion · TabDDPM

## Run the 15 × 8 sweep

```bash
cd "hyper parameter tuning"
pip install -r requirements-hpo.txt

python scripts/run_experiments.py --n-trials 50
# resume (default): skips completed pairs
python scripts/run_experiments.py --resume
# subset
python scripts/run_experiments.py --datasets cancer adult --generators ctgan tvae --n-trials 10
# single pair
python scripts/run_tuning.py --generator ctgan --dataset cancer --n-trials 10
```

Outputs land in `results/`:
- `all_experiments.csv` — master log
- `experiment_summary.csv` — this run
- `{dataset}/{generator}/best_params.json`, `best_synthetic.csv`, `metrics.json`, `trials.csv`, `timing.json`, `completed.json`

## Quick check

```bash
cd "hyper parameter tuning"
python -c "import sys; sys.path.insert(0,'.'); \
from datasets import load_train_test, list_datasets; \
print(list_datasets()); \
train, test, spec = load_train_test('cancer'); \
print(spec.n_train, spec.n_test, spec.target)"
```
