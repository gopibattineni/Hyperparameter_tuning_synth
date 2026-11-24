#!/usr/bin/env python3
"""Queue launcher: run extra multiseed (sampler × seed) grids, 1 worker per GPU.

Keeps free GPUs busy without sharing a device across studies (fairer timings).
Default extra seeds: 7, 99, 1337 (paired across samplers).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("SYNTH_PYTHON", "/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python")
DEFAULT_EXTRA_SEEDS = [7, 99, 1337]
DEFAULT_SAMPLERS = ["standard_tpe", "multivariate_tpe", "gp"]
# GPU1 has historically been broken on this host.
DEFAULT_GPUS = [0, 2, 3, 4, 5]


def _thread_env() -> dict[str, str]:
    keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
        "FOREST_XGB_NTHREAD",
    ]
    env = os.environ.copy()
    for k in keys:
        env.setdefault(k, "1")
    env.setdefault("JOBLIB_MULTIPROCESSING", "0")
    env.setdefault("SYNTH_REPO_ROOT", "/home/gopi_b/SYNTH_BENCHMARK")
    env.setdefault("SYNTH_REQUIRE_CUDA", "1")
    return env


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_EXTRA_SEEDS)
    p.add_argument("--samplers", nargs="*", default=DEFAULT_SAMPLERS)
    p.add_argument("--gpus", nargs="*", type=int, default=DEFAULT_GPUS)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--trial-timeout-s", type=float, default=7200.0)
    p.add_argument(
        "--log-dir",
        type=str,
        default=str(_ROOT / "multiseed_comparison" / "logs"),
    )
    args = p.parse_args()

    results_root = Path(args.results_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Job order: all ST first (faster baseline), then MT, then GP — within each, seeds.
    jobs: list[tuple[str, int]] = []
    for samp in args.samplers:
        for seed in args.seeds:
            jobs.append((samp, int(seed)))

    print(
        f"queue: {len(jobs)} jobs × 105 studies | seeds={args.seeds} "
        f"samplers={args.samplers} gpus={args.gpus} → {results_root}",
        flush=True,
    )

    pending = list(jobs)
    active: dict[int, tuple[subprocess.Popen, str, int, Path]] = {}  # gpu -> ...
    base_env = _thread_env()

    def start_one(gpu: int, samp: str, seed: int) -> None:
        log = log_dir / f"extra_{samp}_seed{seed}_gpu{gpu}.log"
        cmd = [
            PY,
            str(_ROOT / "multiseed_comparison" / "run.py"),
            "--samplers",
            samp,
            "--seeds",
            str(seed),
            "--results-root",
            str(results_root),
            "--n-trials",
            str(args.n_trials),
            "--trial-timeout-s",
            str(args.trial_timeout_s),
            "--no-analyze",
        ]
        env = dict(base_env)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        proc = subprocess.Popen(
            cmd,
            cwd=str(_ROOT),
            env=env,
            stdout=log.open("w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        active[gpu] = (proc, samp, seed, log)
        print(
            f"  start gpu={gpu} {samp} seed={seed} pid={proc.pid} log={log.name}",
            flush=True,
        )

    while pending or active:
        # Reap finished workers
        for gpu in list(active):
            proc, samp, seed, log = active[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            print(
                f"  done  gpu={gpu} {samp} seed={seed} rc={rc} log={log.name}",
                flush=True,
            )
            del active[gpu]

        # Fill free GPUs
        for gpu in args.gpus:
            if gpu in active or not pending:
                continue
            samp, seed = pending.pop(0)
            start_one(gpu, samp, seed)

        if not pending and not active:
            break
        time.sleep(30)

    print("queue empty — all extra-seed jobs finished", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
