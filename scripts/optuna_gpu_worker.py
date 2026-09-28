#!/usr/bin/env python3
"""Single-GPU Optuna worker that contributes trials to a shared study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--n-trials", type=int, required=True)
    p.add_argument("--storage", required=True)
    p.add_argument("--study-name", required=True)
    p.add_argument("--results-dir", default=None)
    return p.parse_args()


def main() -> int:
    import os

    args = parse_args()
    print(
        f"[worker] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"dataset={args.dataset} generator={args.generator} n_trials={args.n_trials}",
        flush=True,
    )
    from tuning.optimizer import run_worker_trials

    run_worker_trials(
        generator_name=args.generator,
        dataset_key=args.dataset,
        n_trials=args.n_trials,
        storage=args.storage,
        study_name=args.study_name,
        results_root=args.results_dir,
    )
    print("[worker] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
