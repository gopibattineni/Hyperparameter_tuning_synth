#!/usr/bin/env python3
"""Pack objective-weight sensitivity studies across GPUs (CUDA required).

Queue TSV lines: ``config\\tdataset\\tgenerator``
One worker process per GPU; claims jobs with flock. Sets
``SYNTH_REQUIRE_CUDA=1`` and ``CUDA_VISIBLE_DEVICES=<gpu>`` so neural
generators never fall back to host CPU.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("SYNTH_PYTHON", "/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python")


def _claim_line(queue_path: Path) -> str | None:
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    if not queue_path.exists():
        return None
    with queue_path.open("r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            lines = f.readlines()
            while lines and not lines[0].strip():
                lines.pop(0)
            if not lines:
                return None
            claimed = lines.pop(0).strip()
            f.seek(0)
            f.truncate()
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
            return claimed
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue", type=str, required=True)
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--gpu", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-trials", type=int, default=2)
    p.add_argument("--n-startup-trials", type=int, default=1)
    p.add_argument("--trial-timeout-s", type=float, default=7200.0)
    p.add_argument("--log-dir", type=str, required=True)
    args = p.parse_args()

    queue_path = Path(args.queue)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_log = log_dir / f"sens_gpu{args.gpu}.out"

    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
    ):
        env.setdefault(k, "1")
    env.setdefault("JOBLIB_MULTIPROCESSING", "0")
    env.setdefault("SYNTH_REPO_ROOT", "/home/gopi_b/SYNTH_BENCHMARK")
    env["SYNTH_REQUIRE_CUDA"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with worker_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"worker start gpu={args.gpu} queue={queue_path} CUDA_VISIBLE_DEVICES={args.gpu}")
    done = 0
    while True:
        claimed = _claim_line(queue_path)
        if not claimed:
            log(f"queue empty — exit (finished {done} jobs)")
            break
        parts = claimed.split("\t")
        if len(parts) != 3:
            log(f"skip bad line: {claimed!r}")
            continue
        config, dataset, generator = parts
        job_log = log_dir / f"sens_{config}_{dataset}_{generator}_gpu{args.gpu}.log"
        log(f"claim {config} {dataset}×{generator} → {job_log.name}")
        cmd = [
            PY,
            "-m",
            "objective_weight_sensitivity.run",
            "--configs",
            config,
            "--datasets",
            dataset,
            "--generators",
            generator,
            "--seed",
            str(args.seed),
            "--n-trials",
            str(args.n_trials),
            "--n-startup-trials",
            str(args.n_startup_trials),
            "--trial-timeout-s",
            str(args.trial_timeout_s),
            "--results-root",
            args.results_root,
            "--no-analyze",
        ]
        with job_log.open("w", encoding="utf-8") as out:
            rc = subprocess.call(cmd, cwd=str(_ROOT), env=env, stdout=out, stderr=subprocess.STDOUT)
        log(f"done  {config} {dataset}×{generator} rc={rc}")
        done += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
