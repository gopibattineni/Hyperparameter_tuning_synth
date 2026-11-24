# Multi-seed sampler comparison

Compare **Standard TPE**, **Multivariate TPE**, and **GPSampler** with **3 paired seeds**
`[42, 123, 2024]`, without changing the objective, metrics, search spaces, or data splits.

## Design notes

| Item | Choice |
|------|--------|
| Objective | Unchanged: \(J = 0.4F + 0.4U - 0.2P\) |
| Trials / study | 20 (override with `--n-trials`) |
| `n_startup_trials` | **10** for all three samplers |
| Multivariate `group` | **Off** — YAML search spaces are flat (no Optuna conditionals) |
| Data splits | Fixed YAML split (unchanged). Seed controls sampler + generator/torch/numpy only |
| GP / torch | Optuna 5.0 already has `GPSampler`; torch 1.13 uses the existing compatibility patch |
| Layout | New tree only: `results/multiseed_YYYYMMDD/` (never overwrites prior experiments) |

## How to run

```bash
cd /home/gopi_b/Hyperparameter_tuning_synth
export SYNTH_REPO_ROOT=/home/gopi_b/SYNTH_BENCHMARK
PY=/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python

# End-to-end dry run (2 trials, cancer × gaussian_copula, 3×3)
$PY multiseed_comparison/run.py --dry-run

# Subset
$PY multiseed_comparison/run.py \
  --datasets cancer alzheimers \
  --generators gaussian_copula ctgan \
  --samplers standard_tpe multivariate_tpe gp \
  --seeds 42 123 2024 \
  --n-trials 20 \
  --trial-timeout-s 7200

# Resume (same --results-root): studies with ≥ n_trials COMPLETE are skipped
$PY multiseed_comparison/run.py --results-root results/multiseed_20261027 ...

# Parallel workers sharing one results root: skip analysis in each worker
$PY multiseed_comparison/run.py --no-analyze --gpu 4 --seeds 123 ...
$PY multiseed_comparison/run.py --no-analyze --gpu 5 --seeds 2024 ...

# Re-analyze only (after all workers finish)
$PY multiseed_comparison/run.py --analyze-only --results-root results/multiseed_20261027
# or:
$PY multiseed_comparison/analyze_samplers.py --results-root results/multiseed_20261027

# Import original seed-42 grids (Standard or Multivariate TPE) into the multiseed tree
$PY multiseed_comparison/import_legacy_st.py \
  --results-root results/multiseed_20261027 --sampler standard_tpe --force --analyze
$PY multiseed_comparison/import_legacy_st.py \
  --results-root results/multiseed_20261027 --sampler multivariate_tpe --force

# Extra seeds (e.g. 7, 99, 1337): queue keeps one study-grid per free GPU
$PY multiseed_comparison/run_extra_seeds_queue.py \
  --results-root results/multiseed_20261027 \
  --seeds 7 99 1337 \
  --samplers standard_tpe multivariate_tpe gp \
  --gpus 0 2 3 4 5
```

Sequential by default (`--n-jobs` is accepted but studies stay one-after-another so timings stay comparable). Optional `--gpu 0` sets `CUDA_VISIBLE_DEVICES`. Use `--no-analyze` when several processes write into the same `results-root`.

## Outputs

```
results/multiseed_YYYYMMDD/
  raw/*.csv              # every trial
  studies/.../optuna.db  # resumable SQLite
  summary_runs.csv       # one row per (dataset, generator, sampler, seed)
  summary_agg.csv        # mean±std across seeds
  pairwise.csv           # MT−ST, GP−ST, GP−MT
  figures/*.png|pdf
  stats_report.txt       # Friedman + Wilcoxon (Holm)
```
