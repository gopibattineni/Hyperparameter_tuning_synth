# Objective weight sensitivity (focused pilot)

Investigate whether the default FORGE objective weights \(0.4 / 0.4 / 0.2\) are
appropriate, **without** redesigning utility, privacy, data splits, or the
multiseed sampler comparison.

## Why

The scalar Optuna objective is

\[
J = w_F\,F + w_U\,U - w_P\,P
\]

(maximised). Different \((w_F, w_U, w_P)\) change which hyperparameters win.
This package runs **independent** Standard-TPE studies for five weight presets
and compares the underlying \(F\), \(U\), and \(P\) (not raw \(J\) across
configs as if they shared a scale).

## Presets (must sum to 1)

| Config | Fidelity | Utility | Privacy |
|--------|---------:|--------:|--------:|
| O1 – Original | 0.400 | 0.400 | 0.200 |
| O2 – Utility-focused | 0.200 | 0.600 | 0.200 |
| O3 – Fidelity-focused | 0.600 | 0.200 | 0.200 |
| O4 – Privacy-focused | 0.300 | 0.300 | 0.400 |
| O5 – Equal weighting | 0.333 | 0.333 | 0.334 |

Defined in [`config/objective.yaml`](../config/objective.yaml) under `presets:`.

## Fidelity (new for this pilot)

Sensitivity studies use:

\[
F = \mathrm{clip}_{[0,1]}(\mathrm{SDV\ QualityReport})
\]

KS and correlation similarities are still **computed and stored** as diagnostics
but are **not** part of \(F\) or \(J\). If SDV is unavailable, the trial is
**failed/pruned** explicitly — there is **no** silent fallback to KS/corr.

All objectives (O1–O5) and the default multiseed runner use
`fidelity.aggregation: sdv_only`. KS/corr never enter \(F\) or \(J\).

## Unchanged

- **Utility**: TRTR−TSTR mean F1 gap (classification) / R² gap (regression).
- **Privacy**: \(P = 0.5\,\mathrm{MIA} + 0.3\,\mathrm{NNDR\_risk} + 0.2\,\mathrm{DCR\_risk}\).
- **Data**: fixed YAML 80/20 train/test (`seed: 42`); no validation redesign.
- **Sampler**: Standard TPE only (not Multivariate TPE / GP).
- **Search spaces / generators**: existing YAML spaces.
- **Historical results**: `results/multiseed_*` are never written by this package.

## Commands

```bash
# Smoke (2 trials)
python -m objective_weight_sensitivity.run --dry-run

# Small pilot — all five configs
python -m objective_weight_sensitivity.run \
  --configs O1 O2 O3 O4 O5 \
  --datasets cancer \
  --generators gaussian_copula tvae \
  --n-trials 5 --seed 42

# Rebuild tables/plots only
python -m objective_weight_sensitivity.run --analyze-only \
  --results-root results/objective_weight_sensitivity
```

## Outputs

Under `results/objective_weight_sensitivity/`:

- `objective_weight_comparison.csv`
- `best_trials.csv`
- `all_trials.csv`
- `objective_metadata.json`
- `objective_weight_comparison.xlsx` — styled workbook (Overall_Comparison + O1–O5 sheets)
- `figures/` — grouped F/U/P bars, weight chart, F vs U, U vs P scatters
- Per study: `{dataset}/{generator}/{Oi}/seed{N}/optuna.db|summary.json|meta.json`

### Excel export (no Optuna)

```bash
python -m objective_weight_sensitivity.export_excel \
  --results-root results/objective_weight_sensitivity

# or
python -m objective_weight_sensitivity.run --export-excel --analyze-only \
  --results-root results/objective_weight_sensitivity
```

Workbook sheets: `Overall_Comparison`, `O1_Original`, `O2_Utility`, `O3_Fidelity`,
`O4_Privacy`, `O5_Equal`. All five objective sheets share the same A–E layout.
Formatting matches `templates/Sampler_results_10_seeds.xlsx` (FORGE navy / blue /
green). Missing metrics are left blank (never coerced to zero).
