#!/usr/bin/env python3
"""Run Optuna trials for ONE dataset × generator across multiple GPUs.

Each healthy GPU gets its own process with ``CUDA_VISIBLE_DEVICES`` pinned.
Workers share an SQLite Optuna study so TPE still sees every completed trial.

Example (forest_cover × CTAB-GAN+, 10 trials on GPUs 0,1,2,3,5)::

    scripts/run_multigpu_trials.py \\
        --dataset forest_cover --generator ctabgan \\
        --n-trials 10 --gpus 0,1,2,3,5

After all workers finish, the best params are refit once and artifacts are
written the same way as ``run_experiments.py``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _healthy_gpus(requested: str) -> list[str]:
    out: list[str] = []
    for raw in requested.split(","):
        g = raw.strip()
        if not g:
            continue
        r = subprocess.run(
            ["nvidia-smi", "-i", g],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if r.returncode == 0:
            out.append(g)
        else:
            print(f"skip GPU {g} (nvidia-smi failed)", flush=True)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-GPU Optuna trials for one pair")
    p.add_argument("--dataset", required=True)
    p.add_argument("--generator", required=True)
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--gpus", default="0,1,2,3,4,5", help="Comma-separated GPU ids")
    p.add_argument("--results-dir", default=None)
    p.add_argument(
        "--force",
        action="store_true",
        help="Delete prior completed/failed markers and recreate the study DB",
    )
    p.add_argument("--python", default=None, help="Python executable (default: SYNTH_PYTHON or this interpreter)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from config_utils import PACKAGE_ROOT
    from datasets.loader import load_dataset_config
    from tuning.artifacts import run_dir
    from tuning.optimizer import finalize_study_from_storage, study_storage_url

    ds_cfg = load_dataset_config()
    if args.dataset not in ds_cfg:
        raise SystemExit(f"Unknown dataset: {args.dataset}")
    task = ds_cfg[args.dataset].get("task", "classification")

    results_root = Path(args.results_dir) if args.results_dir else PACKAGE_ROOT / "results"
    out_dir = run_dir(results_root, args.dataset, args.generator, task=task)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpus = _healthy_gpus(args.gpus)
    if not gpus:
        raise SystemExit("No healthy GPUs found")

    n_workers = len(gpus)
    # Split trials as evenly as possible across GPUs
    base, rem = divmod(args.n_trials, n_workers)
    trials_per_gpu = [base + (1 if i < rem else 0) for i in range(n_workers)]
    trials_per_gpu = [t for t in trials_per_gpu if t > 0]
    gpus = gpus[: len(trials_per_gpu)]

    storage = study_storage_url(out_dir)
    db_path = out_dir / "optuna.db"
    study_name = f"{args.dataset}__{args.generator}"

    if args.force:
        for name in ("completed.json", "failed.json", "optuna.db", "optuna.db-journal"):
            p = out_dir / name
            if p.is_file():
                p.unlink()
                print(f"removed {p.name}")

    py = args.python or os.environ.get("SYNTH_PYTHON") or sys.executable
    env_base = os.environ.copy()
    env_base["SYNTH_REPO_ROOT"] = env_base.get(
        "SYNTH_REPO_ROOT", "/home/gopi_b/SYNTH_BENCHMARK"
    )

    log_dir = out_dir / "multigpu_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Study {study_name}  storage={storage}\n"
        f"Total trials={args.n_trials} across GPUs={gpus} → per-GPU {trials_per_gpu}",
        flush=True,
    )

    procs: list[subprocess.Popen] = []
    for gpu, n_local in zip(gpus, trials_per_gpu):
        log = log_dir / f"worker_gpu{gpu}.log"
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            py,
            "-u",
            str(_ROOT / "scripts" / "optuna_gpu_worker.py"),
            "--dataset",
            args.dataset,
            "--generator",
            args.generator,
            "--n-trials",
            str(n_local),
            "--storage",
            storage,
            "--study-name",
            study_name,
            "--results-dir",
            str(results_root),
        ]
        fh = open(log, "w", encoding="utf-8")
        print(f"  launch GPU={gpu} n_trials={n_local} log={log}", flush=True)
        procs.append(
            subprocess.Popen(cmd, cwd=str(_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        )

    t0 = time.perf_counter()
    codes = [p.wait() for p in procs]
    wall = time.perf_counter() - t0
    print(f"Workers done in {wall:.1f}s  exit_codes={codes}", flush=True)
    if any(c != 0 for c in codes):
        print("WARNING: one or more workers failed — check multigpu_logs/", flush=True)

    # Finalize: refit best + write Excel-ready artifacts
    summary = finalize_study_from_storage(
        generator_name=args.generator,
        dataset_key=args.dataset,
        storage=storage,
        study_name=study_name,
        results_root=results_root,
    )
    print(
        f"Finalize status={summary.get('status')} best={summary.get('best_value')} "
        f"n_trials={summary.get('n_trials')} → {summary.get('run_dir')}",
        flush=True,
    )
    return 0 if summary.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
