#!/usr/bin/env python3
"""Run Multivariate TPE HPO (Optuna TPESampler with multivariate=True).

Catalogue (same as Standard TPE):
  • 9 classification datasets
  • 6 regression datasets
  • 7 generators (ForestDiffusion excluded by default)

Results under ``multivariate_tpe/results/`` never overwrite ``standard_tpe/results/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multivariate_tpe.datasets_catalog import (  # noqa: E402
    ALL_DATASETS,
    CLASSIFICATION_DATASETS,
    GENERATORS_NO_FOREST,
    REGRESSION_DATASETS,
    assert_catalog,
)
from tuning.alt_sampler_runner import DEFAULT_GENERATORS, run_pairs  # noqa: E402

# Already finished (incl. forest_diffusion) for the Cancer / Alzheimer's pilot.
DONE_DATASETS = ("cancer", "alzheimers")
REMAINING_DATASETS = [d for d in ALL_DATASETS if d not in DONE_DATASETS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multivariate TPE HPO — 9 classification + 6 regression datasets; "
            "7 generators (no ForestDiffusion by default)"
        )
    )
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset keys (default: remaining 13 after cancer/alzheimers)",
    )
    p.add_argument(
        "--generators",
        nargs="*",
        default=None,
        help="Generator keys (default: 7 gens without forest_diffusion)",
    )
    p.add_argument("--n-trials", type=int, default=None, help="Override per-pair budget")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--quiet-trials", action="store_true")
    p.add_argument(
        "--include-forest",
        action="store_true",
        help="Also run forest_diffusion (off by default)",
    )
    p.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run all 15 datasets (resume-skips completed pairs)",
    )
    p.add_argument(
        "--classification-only",
        action="store_true",
        help="Run all 9 classification datasets",
    )
    p.add_argument(
        "--regression-only",
        action="store_true",
        help="Run all 6 regression datasets",
    )
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
        f"multivariate_tpe catalogue: classification={n_cls}/9 "
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
        force_torch_compat=False,
    )
    n_ok = sum(1 for r in rows if r.get("status") == "completed")
    n_skip = sum(1 for r in rows if r.get("status") == "skipped_resume")
    n_fail = sum(1 for r in rows if r.get("status") == "failed")
    print(f"multivariate_tpe done: completed={n_ok} skipped={n_skip} failed={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
