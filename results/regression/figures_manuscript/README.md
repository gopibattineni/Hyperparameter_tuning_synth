# Manuscript figures — regression Optuna HPO

Generated from existing Optuna studies (`study.pkl`) under each
`results/regression/{n}_{dataset}/{generator}/` folder.
**No HPO was re-run.**

## Reproduce

```bash
python3 scripts/plot_regression_optuna_figures.py
```

Script: `scripts/plot_regression_optuna_figures.py`

## Main figures

| File | Content |
|------|---------|
| `Fig1_optimization_history.{png,pdf}` | Trial objective vs trial #; 6 dataset panels; all 6 generators overlaid; running-best dashed; overall best marked |
| `Fig2_hyperparameter_importance.{png,pdf}` | Fanova importance from the **best generator** study on each dataset |
| `Fig3_best_objective_comparison.{png,pdf}` | Best objective per dataset (max over generators); bar labels + winning generator |

## Supplementary

| File | Content |
|------|---------|
| `Supp_parallel_coordinates.{png,pdf}` | Parallel coordinates for the best-generator study per dataset; best trial in red |

## Supporting tables

| File | Content |
|------|---------|
| `study_inventory.csv` | Per dataset×generator: COMPLETE/PRUNED/FAIL counts, best value, best trial, params |
| `Fig3_best_objective_summary.csv` | Best objective used in Fig 3 |

## Design notes

- Each dataset has **six independent** Optuna studies (one per generator).
- Fig 2 / Supp use the generator with the highest `completed.json` / study best value on that dataset.
- Only `TrialState.COMPLETE` trials with a numeric value enter plots; pruned/failed are excluded.
- Importance uses Optuna Fanova; constant or non-estimable parameters are omitted (not fabricated).
- Raster: 300 dpi PNG; vector: PDF (Type 42 fonts).
- Body text uses Times New Roman when installed; otherwise Liberation Serif (metric-compatible substitute).
- Fig 3 winning-generator legend is outside the axes (upper right) so it does not overlap bars.
