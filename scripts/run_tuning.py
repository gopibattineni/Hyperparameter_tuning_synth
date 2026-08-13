#!/usr/bin/env python3
"""CLI entry point for a single generator × dataset Optuna study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tuning.optimizer import run_study


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Optuna HPO for one generator × dataset")
    p.add_argument("--generator", required=True, help="Registry key, e.g. ctgan")
    p.add_argument("--dataset", required=True, help="Dataset key from datasets.yaml")
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    row = run_study(
        args.generator,
        args.dataset,
        n_trials=args.n_trials,
        results_root=args.results_dir,
    )
    print(row)
    return 0 if row.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
