# Gaussian Process (GPSampler) experiments

Optuna `GPSampler` HPO — same catalogue as Standard / Multivariate TPE.

## Catalogue

| Split | Count | Notes |
|-------|------:|-------|
| Classification | **9** | cancer → magic_gamma |
| Regression | **6** | metro_interstate → real_estate |
| Generators | **7** | no ForestDiffusion |

Cancer + Alzheimer's already complete. Remaining pairs resume-safe.

```bash
# Multi-GPU packed (default: classification)
python scripts/run_gaussian_process_packed.py start
python scripts/run_gaussian_process_packed.py status

# Include regression
CLASSIFICATION_ONLY=0 python scripts/run_gaussian_process_packed.py start
```
