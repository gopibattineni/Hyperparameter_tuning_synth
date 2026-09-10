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
├── tuning/                 # Optuna objective, optimizer, artifacts
├── scripts/                # CLI entry points
├── results/                # run outputs
├── config_utils.py
├── requirements-hpo.txt
├── DOCUMENTATION.md
└── README.md
```

## Objective

```text
0.4 × Fidelity + 0.4 × Utility − 0.2 × PrivacyRisk   (maximize)
```

## Generators (8)

GaussianCopula · CopulaGAN · CTGAN · CTAB-GAN+ · TVAE · WGAN-GP · ForestDiffusion · TabDDPM

## Quick start

```bash
cd /home/gopi.battineni/Hyperparameter_tuning_synth
pip install -r requirements-hpo.txt

# Priority sweep
python scripts/run_experiments.py --datasets cancer alzheimers adult --n-trials 20 --resume

# Single pair
python scripts/run_tuning.py --generator ctgan --dataset cancer --n-trials 10
```

Outputs land in `results/`:

- `all_experiments.csv` — master log
- `{classification|regression}/{n}_{dataset}/{generator}/best_params.json`, `metrics.json`, `best_synthetic.csv`, `trials.csv`, `completed.json`
- Report packs: `results/classification/1_cancer/cancer_HPO_*.{txt,xlsx}`, `results/classification/2_alzheimers/alzheimers_HPO_*.{txt,xlsx}`

## Quick check

```bash
python -c "import sys; sys.path.insert(0,'.'); \
from datasets import load_train_test, list_datasets; \
print(list_datasets()); \
train, test, spec = load_train_test('cancer'); \
print(spec.n_train, spec.n_test, spec.target)"
```
