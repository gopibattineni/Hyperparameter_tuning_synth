#!/usr/bin/env python3
"""Quick smoke test: Cancer × gaussian_copula, 2 multivariate-TPE trials."""

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
    # Reuse configs but write into _smoke via a temporary experiment layout
    cfg = _EXP / "config" / "optuna.yaml"
    budget = _EXP / "config" / "trials_budget.yaml"
    # Point results under _smoke by copying runner call with custom root
    rows = run_pairs(
        experiment_root=smoke_root,
        optuna_cfg_path=cfg,
        budget_path=budget,
        datasets=["cancer"],
        generators=["gaussian_copula"],
        n_trials_override=2,
        resume=False,
        show_trial_progress=True,
    )
    ok = rows and rows[0].get("status") == "completed"
    print("SMOKE", "PASS" if ok else "FAIL", rows)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
