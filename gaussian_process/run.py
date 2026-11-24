#!/usr/bin/env python3
"""Run Gaussian Process (Optuna GPSampler) HPO.

Catalogue (same as Standard / Multivariate TPE):
  • 9 classification + 6 regression datasets
  • 7 generators (ForestDiffusion excluded by default)

Results under ``gaussian_process/results/`` never overwrite other experiment folders.

Compatibility: Optuna 5 GPSampler needs ``torch.device`` context (PyTorch ≥ 2.1).
This env has torch 1.13, so ``force_torch_compat=True`` patches GPSampler only
(still GPSampler — not a substitute sampler).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multivariate_tpe.datasets_catalog import (
    ALL_DATASETS,
    CLASSIFICATION_DATASETS,
    GENERATORS_NO_FOREST,
    REGRESSION_DATASETS,
    assert_catalog,
)
from tuning.alt_sampler_runner import DEFAULT_GENERATORS, run_pairs

DONE_DATASETS = ("cancer", "alzheimers")
REMAINING_DATASETS = [d for d in ALL_DATASETS if d not in DONE_DATASETS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "GPSampler HPO — 9 classification + 6 regression; "
            "7 generators (no ForestDiffusion by default)"
        )
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--generators", nargs="*", default=None)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--quiet-trials", action="store_true")
    p.add_argument("--include-forest", action="store_true")
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--classification-only", action="store_true")
    p.add_argument("--regression-only", action="store_true")
    return p.parse_args()


def main() -> int:
    assert_catalog()
    args = parse_args()

    if args.datasets is not None:
        datasets = list(args.datasets)
    elif args.all_datasets:
        datasets = list(ALL_DATASETS)
    elif args.classification_only:
        datasets = list(CLASSIFICATION_DATASETS)
    elif args.regression_only:
        datasets = list(REGRESSION_DATASETS)
    else:
        datasets = list(REMAINING_DATASETS)

    if args.generators is not None:
        generators = list(args.generators)
    elif args.include_forest:
        generators = list(DEFAULT_GENERATORS)
    else:
        generators = list(GENERATORS_NO_FOREST)

    n_cls = sum(1 for d in datasets if d in CLASSIFICATION_DATASETS)
    n_reg = sum(1 for d in datasets if d in REGRESSION_DATASETS)
    print(
        f"gaussian_process catalogue: classification={n_cls}/9 "
        f"regression={n_reg}/6 generators={len(generators)} "
        f"datasets={datasets}"
    )

    rows = run_pairs(
        experiment_root=_EXP,
        optuna_cfg_path=_EXP / "config" / "optuna.yaml",
        budget_path=_EXP / "config" / "trials_budget.yaml",
        datasets=datasets,
        generators=generators,
        n_trials_override=args.n_trials,
        resume=not args.no_resume,
        show_trial_progress=not args.quiet_trials,
        force_torch_compat=True,
    )
    n_ok = sum(1 for r in rows if r.get("status") == "completed")
    n_skip = sum(1 for r in rows if r.get("status") == "skipped_resume")
    n_fail = sum(1 for r in rows if r.get("status") == "failed")
    print(f"gaussian_process done: completed={n_ok} skipped={n_skip} failed={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
