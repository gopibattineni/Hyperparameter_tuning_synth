# Experiment status

## Running now

Full Cancer + Alzheimer’s sweeps for:

1. `multivariate_tpe/run.py` — Optuna `TPESampler(multivariate=True)`
2. `gaussian_process/run.py` — Optuna `GPSampler` (with torch 1.13 context-manager shim)

Baseline **standard TPE** results already exist under `results/classification/{1_cancer,2_alzheimers}/` and are **not** modified.

## Monitor

```bash
# Process status
ps -p $(cat multivariate_tpe/logs/full_run.pid) -o pid,etime,cmd
ps -p $(cat gaussian_process/logs/full_run.pid) -o pid,etime,cmd

# Live logs
tail -f multivariate_tpe/logs/full_run.out
tail -f gaussian_process/logs/full_run.out

# Completed pairs so far
find multivariate_tpe/results gaussian_process/results -name completed.json | sort

# Refresh comparison tables/figures
python scripts/compare_bayesian_samplers.py
```

## Expected wall-clock

Roughly several hours per sampler (7 generators × 2 datasets; CTGAN/CTAB-GAN+/TabDDPM dominate). Runs are resume-safe via per-pair `optuna.db`.
