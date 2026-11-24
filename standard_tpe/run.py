#!/usr/bin/env python3
"""Run Standard TPE HPO for Cancer and Alzheimer's (comparison subset).

Results are written under ``standard_tpe/results/``.
For the full 15×8 sweep use ``standard_tpe/run_experiments.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tuning.alt_sampler_runner import DEFAULT_DATASETS, DEFAULT_GENERATORS, run_pairs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standard TPE experiments (Cancer, Alzheimer's)")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--generators", nargs="*", default=None)
    p.add_argument("--n-trials", type=int, default=None, help="Override per-pair budget")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--quiet-trials", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rows = run_pairs(
        experiment_root=_EXP,
        optuna_cfg_path=_EXP / "config" / "optuna.yaml",
        budget_path=_EXP / "config" / "trials_budget.yaml",
        datasets=args.datasets or DEFAULT_DATASETS,
        generators=args.generators or DEFAULT_GENERATORS,
        n_trials_override=args.n_trials,
        resume=not args.no_resume,
        show_trial_progress=not args.quiet_trials,
        force_torch_compat=False,
    )
    n_ok = sum(1 for r in rows if r.get("status") == "completed")
    n_skip = sum(1 for r in rows if r.get("status") == "skipped_resume")
    n_fail = sum(1 for r in rows if r.get("status") == "failed")
    print(f"standard_tpe done: completed={n_ok} skipped={n_skip} failed={n_fail}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
