#!/usr/bin/env python3
"""Quick smoke test: Cancer × gaussian_copula, 2 Standard-TPE trials."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_ROOT = _EXP.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tuning.alt_sampler_runner import run_pairs


def main() -> int:
    smoke_root = _EXP / "_smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
    smoke_root.mkdir(parents=True)
    rows = run_pairs(
        experiment_root=smoke_root,
        optuna_cfg_path=_EXP / "config" / "optuna.yaml",
        budget_path=_EXP / "config" / "trials_budget.yaml",
        datasets=["cancer"],
        generators=["gaussian_copula"],
        n_trials_override=2,
        resume=False,
        show_trial_progress=True,
        force_torch_compat=False,
    )
    ok = rows and rows[0].get("status") == "completed"
    print("SMOKE", "PASS" if ok else "FAIL", rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
