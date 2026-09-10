# Hyperparameter Tuning for Synthetic Tabular Data Generators

## 1. Project overview

This project finds **high-quality hyperparameters** for synthetic tabular data generators using **Optuna**.

For every **dataset × generator** pair it:

1. Samples hyperparameters from a YAML search space
2. Trains the generator on the **real training split only**
3. Generates a synthetic dataset of the same size
4. Scores quality on **fidelity**, **utility**, and **privacy risk**
5. Maximizes a weighted objective
6. Saves best parameters, metrics, and synthetic data

**Workspace root**

```text
/home/gopi.battineni/Hyperparameter_tuning_synth
```

**Data / vendor root** (configured in `config_utils.py`)

```text
/home/gopi.battineni/SYNTH_BENCHMARK/SYNTH
```

---

## 2. Goals

- Compare 8 research generators fairly under the same evaluation protocol
- Discover hyperparameters that tune well for high-quality synthetic data
- Support resumeable large sweeps (15 datasets × 8 generators)
- Produce reusable artefacts: best params, metrics, synthetic CSVs, master CSV

---

## 3. Repository layout

```text
Hyperparameter_tuning_synth/
├── config/
│   ├── datasets.yaml              # 15 dataset definitions
│   ├── objective.yaml             # objective weights + classifiers
│   ├── optuna.yaml                # trials, sampler, pruner, artefacts
│   └── generators/
│       ├── gaussian_copula.yaml
│       ├── copulagan.yaml
│       ├── ctgan.yaml
│       ├── ctabgan.yaml
│       ├── tvae.yaml
│       ├── wgan_gp.yaml
│       ├── forest_diffusion.yaml
│       ├── tabddpm.yaml
│       └── bootstrap_noise.yaml   # smoke / baseline only
├── datasets/
│   ├── loader.py                  # load CSV / Excel / UCI + train/test split
│   └── schema.py                  # DatasetSpec
├── generators/
│   ├── base.py                    # BaseGenerator interface
│   ├── registry.py                # name → class lookup
│   ├── sdv_wrappers.py            # GaussianCopula, CopulaGAN, CTGAN, TVAE
│   ├── advanced_wrappers.py       # CTAB-GAN+, ForestDiffusion, TabDDPM, WGAN-GP
│   └── bootstrap.py               # noise baseline
├── evaluation/
│   ├── fidelity.py                # KS + correlation similarity
│   ├── utility.py                 # TSTR vs TRTR (F1 / R²)
│   ├── privacy.py                 # MIA AUC, NNDR, DCR → privacy_risk
│   └── objective.py               # weighted scalar score
├── tuning/
│   ├── objective_fn.py            # one Optuna trial
│   ├── optimizer.py               # study runner + final retrain
│   ├── search_space.py            # YAML → trial.suggest_*
│   └── artifacts.py               # save / resume markers
├── scripts/
│   ├── run_experiments.py         # multi-pair sweep CLI
│   ├── run_tuning.py              # single pair CLI
│   └── run_all.py
├── results/
│   ├── classification/{n}_{dataset}/{generator}/   # 1–9
│   ├── regression/{n}_{dataset}/{generator}/       # 10–15
│   ├── all_experiments.csv
│   └── _archive/  _smoke/
├── config_utils.py
├── requirements-hpo.txt
├── README.md
└── DOCUMENTATION.md               # this file
```

---

## 4. End-to-end pipeline

```text
Datasets + Generator YAML search space
                │
                ▼
        Optuna Study (TPE)
                │
                ▼
   ┌────────────────────────────────┐
   │  Trial loop (N trials)         │
   │  1. Sample hyperparameters     │
   │  2. generator.fit(train)       │
   │  3. generator.sample(n_train)  │
   │  4. Evaluate F / U / P         │
   │  5. Score objective            │
   │  6. Feedback to TPE sampler    │
   └────────────────────────────────┘
                │
                ▼
        Best trial hyperparameters
                │
                ▼
   Retrain best → save artefacts
   (best_params.json, metrics.json,
    best_synthetic.csv, trials.csv)
```

### Important rules

- Generators never see the **held-out test set** during training
- Synthetic size defaults to **match training set size**
- Failed generator trials are **pruned** so Optuna continues
- Pairs with `completed.json` are skipped when `--resume` is on (default)

---

## 5. Objective formula

Configured in `config/objective.yaml`:

```text
Objective = 0.4 × Fidelity + 0.4 × Utility − 0.2 × PrivacyRisk
```

Direction: **maximize**.

| Component | Meaning | Range | Better when |
|-----------|---------|-------|-------------|
| **Fidelity** | Statistical similarity (KS + correlation) | [0, 1] | Higher |
| **Utility** | Downstream ML usefulness (TSTR vs TRTR) | [0, 1] | Higher |
| **Privacy Risk** | Leakage risk (MIA / NNDR / DCR mix) | [0, 1] | Lower (penalized) |

### Fidelity

- **KS similarity**: `1 − KS distance`, averaged over columns
- **Corr similarity**: how well pairwise correlations are preserved
- Combined into a single fidelity score in `[0, 1]`

### Utility

- Train classifiers on **synthetic** data → test on **real test** (**TSTR**)
- Train same classifiers on **real train** → test on **real test** (**TRTR**)
- `mean_f1_gap = mean(TRTR_F1 − TSTR_F1)`
- `utility = 1 − clip(mean_f1_gap, 0, 1)`

Default classifiers used during HPO:

1. Logistic Regression
2. Random Forest
3. Decision Tree

For regression datasets, utility uses **R² gap** instead of F1.

### Privacy risk

Weighted mix (higher = worse):

| Metric | Weight | Role |
|--------|--------|------|
| MIA AUC | 0.50 | Membership inference attack success |
| NNDR | 0.30 | Nearest-neighbour distance ratio |
| DCR | 0.20 | Distance to closest record |

---

## 6. Generators

| Key | Backend | Notes |
|-----|---------|-------|
| `gaussian_copula` | SDV GaussianCopula | Fast parametric baseline |
| `copulagan` | SDV CopulaGAN | Copula transform + CTGAN |
| `ctgan` | SDV CTGAN | Conditional GAN |
| `ctabgan` | CTAB-GAN+ vendor | CNN tabular GAN |
| `tvae` | SDV TVAE | Tabular VAE |
| `wgan_gp` | Torch MLP WGAN-GP | Wasserstein + gradient penalty |
| `forest_diffusion` | ForestDiffusion | Tree-based diffusion / flow |
| `tabddpm` | TabDDPM vendor | Diffusion (often unavailable if deps missing) |

Each generator has:

- **defaults** in `config/generators/{name}.yaml`
- **search_space** mapped to Optuna `suggest_int` / `suggest_float` / `suggest_categorical`

Unavailable generators are skipped automatically (`--skip-unavailable`, default on).

---

## 7. Datasets

Defined in `config/datasets.yaml` (15 total). Sources: local CSV/Excel under SYNTH `Datasets/`, or UCI via `ucimlrepo`.

Priority / completed reporting so far:

| Dataset | Task | Status |
|---------|------|--------|
| `cancer` | classification | Full report ready |
| `alzheimers` | classification | Full report ready |
| `adult` | classification | Partial (some generators) |
| `concrete` | regression | Pending |

Other registered datasets include forest cover, bank marketing, wine quality, CDC diabetes, mushroom, MAGIC gamma, metro interstate, online shopping, air quality, energy efficiency, real estate.

---

## 8. Optuna settings

From `config/optuna.yaml`:

| Setting | Default |
|---------|---------|
| Trials | 50 (often overridden to 20 for priority sweeps) |
| Sampler | TPESampler (seed 42) |
| Pruner | MedianPruner (startup = 5) |
| Jobs | 1 |
| Storage | in-memory |

Artefacts saved per pair under `results/{classification|regression}/{n}_{dataset}/{generator}/` (dataset numbers 1–15):

- `best_params.json`
- `metrics.json`
- `best_synthetic.csv`
- `trials.csv`
- `timing.json`
- `completed.json` / `failed.json`
- optional Optuna plots + config snapshots

Master logs:

- `results/all_experiments.csv`
- `results/experiment_summary.csv`

---

## 9. How to run

### Setup

```bash
cd /home/gopi.battineni/Hyperparameter_tuning_synth

# SYNTH base deps (from SYNTH_BENCHMARK) + HPO extras
pip install -r /home/gopi.battineni/SYNTH_BENCHMARK/SYNTH/requirements.txt
pip install -r requirements-hpo.txt

# Optional TabDDPM extras (often fragile with NumPy versions)
pip install "libzero==0.0.8" "rtdl==0.0.13" --no-deps

# Prefer NumPy 1.x for SDV / compiled deps compatibility
pip install "numpy>=1.24,<2"
```

### Multi-pair sweep

```bash
python scripts/run_experiments.py \
  --datasets cancer adult concrete \
  --n-trials 20 \
  --resume
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--datasets ...` | Subset of dataset keys |
| `--generators ...` | Subset of generator keys |
| `--n-trials N` | Override Optuna trial budget |
| `--resume` | Skip completed pairs (default) |
| `--no-resume` | Force re-run |
| `--include-unavailable` | Attempt generators with missing deps |

### Single pair

```bash
python scripts/run_tuning.py \
  --generator ctgan \
  --dataset cancer \
  --n-trials 20
```

### Smoke-check data load

```bash
python -c "
import sys; sys.path.insert(0, '.')
from datasets import load_train_test, list_datasets
print(list_datasets())
train, test, spec = load_train_test('cancer')
print(spec.task, spec.target, train.shape, test.shape)
"
```

---

## 10. Results already produced

### Cancer report

- `results/classification/1_cancer/cancer_HPO_notation.txt`
- `results/classification/1_cancer/cancer_HPO_results.xlsx`

Best by objective: **TVAE** (~0.64)

### Alzheimers report

- `results/classification/2_alzheimers/alzheimers_HPO_notation.txt`
- `results/classification/2_alzheimers/alzheimers_HPO_results.xlsx`

Best by objective: **TVAE** (~0.63)

Excel sheets in each report:

1. **Performance Metrics** — objective + fidelity/utility/privacy + F1 + accuracy
2. **Best Hyperparameters** — Optuna winners
3. **Objective Formula** — metric definitions
4. **Search Space (Optuna)** — ranges explored

---

## 11. Module responsibilities

| Module | Role |
|--------|------|
| `scripts/run_experiments.py` | Outer loop over pairs; resume; master CSV |
| `tuning/optimizer.py` | Create study, optimize, retrain best, save artefacts |
| `tuning/objective_fn.py` | One trial: sample → fit → sample → evaluate → score |
| `tuning/search_space.py` | YAML search space + restore Optuna stringified lists/bools |
| `generators/*` | Thin wrappers around SDV / vendor / torch backends |
| `evaluation/*` | Metric computation and objective combination |
| `datasets/loader.py` | Load + split + schema inference |
| `config_utils.py` | Paths + YAML loaders (`REPO_ROOT`, `PACKAGE_ROOT`) |

---

## 12. Design notes / known caveats

1. **Optuna list categoricals** may be stored as strings like `'[256, 256]'`.  
   Restored via `restore_optuna_params()` before final retrain.

2. **NumPy 2.x** often breaks SDV / numexpr / bottleneck stacks. Prefer NumPy `<2`.

3. **TabDDPM** frequently unavailable unless `libzero` / `rtdl` are installed carefully.

4. Some generators (notably **WGAN-GP**, **ForestDiffusion**) can achieve high fidelity but near-zero utility on small/clinical tables — synthetic lookalike data that does not train useful classifiers.

5. Adult / large tables make GAN trials slow (tens of minutes per trial). Prefer resume and smaller trial budgets for exploration.

6. Disk space matters: Optuna study pickles and synthetic CSVs accumulate under `results/`.

---

## 13. Recommended workflow

1. Fix / verify `REPO_ROOT` and dataset paths
2. Install deps and smoke-load target datasets
3. Run a priority subset (e.g. 3 datasets × available generators × 20 trials)
4. Inspect `results/{classification|regression}/{n}_{dataset}/{generator}/metrics.json` and master CSV
5. Export notation + Excel reports for analysis
6. Scale remaining datasets to 50 trials once the playbook is stable

---

## 14. Quick reference commands

```bash
# Priority subset
python scripts/run_experiments.py --datasets cancer alzheimers adult --n-trials 20 --resume

# One generator on three datasets
python scripts/run_experiments.py --datasets cancer alzheimers adult --generators ctgan --n-trials 20 --resume

# List registered generators
python -c "import sys; sys.path.insert(0,'.'); from generators import list_generators; print(list_generators())"
```

---

*Last updated: 2026-08-20*
