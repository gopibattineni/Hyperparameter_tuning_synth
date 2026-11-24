#!/usr/bin/env python3
"""Pull individual (seed, dataset, generator, sampler) studies from a shared queue.

One process per GPU. Uses an exclusive flock on the queue file so workers
never claim the same study. Studies already complete (≥ n_trials) are skipped
by the underlying runner.
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
    """Atomically pop the first non-empty line from ``queue_path``."""
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
    p.add_argument("--queue", type=str, required=True, help="TSV: seed\\tdataset\\tgenerator")
    p.add_argument("--results-root", type=str, required=True)
    p.add_argument("--sampler", type=str, default="gp")
    p.add_argument("--gpu", type=str, required=True)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--trial-timeout-s", type=float, default=7200.0)
    p.add_argument("--log-dir", type=str, required=True)
    p.add_argument(
        "--objective-preset",
        type=str,
        default=None,
        help="O1–O5 weight preset forwarded to run.py",
    )
    p.add_argument(
        "--force-sdv-only",
        action="store_true",
        default=True,
        help="Forward SDV-only fidelity to run.py (default; KS excluded from F).",
    )
    p.add_argument(
        "--legacy-composite-fidelity",
        action="store_true",
        help="Forward legacy_composite fidelity (includes KS).",
    )
    args = p.parse_args()

    queue_path = Path(args.queue)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_log = log_dir / f"packed_{args.sampler}_gpu{args.gpu}.log"

    env = os.environ.copy()
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
        "TORCH_NUM_INTEROP_THREADS",
        "FOREST_XGB_NTHREAD",
    ):
        env.setdefault(k, "1")
    env.setdefault("JOBLIB_MULTIPROCESSING", "0")
    env.setdefault("SYNTH_REPO_ROOT", "/home/gopi_b/SYNTH_BENCHMARK")
    # Neural generators must not fall back to host CPU.
    env.setdefault("SYNTH_REQUIRE_CUDA", "1")
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with worker_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    log(f"worker start gpu={args.gpu} sampler={args.sampler} queue={queue_path}")
    done = 0
    while True:
        claimed = _claim_line(queue_path)
        if not claimed:
            log(f"queue empty — worker exit (finished {done} jobs)")
            break
        parts = claimed.split("\t")
        if len(parts) != 3:
            log(f"skip bad line: {claimed!r}")
            continue
        seed, dataset, generator = parts
        job_log = log_dir / f"packed_{args.sampler}_seed{seed}_{dataset}_{generator}_gpu{args.gpu}.log"
        log(f"claim seed={seed} {dataset}×{generator} → {job_log.name}")
        cmd = [
            PY,
            str(_ROOT / "multiseed_comparison" / "run.py"),
            "--samplers",
            args.sampler,
            "--seeds",
            str(seed),
            "--datasets",
            dataset,
            "--generators",
            generator,
            "--results-root",
            args.results_root,
            "--n-trials",
            str(args.n_trials),
            "--trial-timeout-s",
            str(args.trial_timeout_s),
            "--no-analyze",
        ]
        if args.objective_preset:
            cmd.extend(["--objective-preset", str(args.objective_preset)])
        if args.legacy_composite_fidelity:
            cmd.append("--legacy-composite-fidelity")
        else:
            cmd.append("--force-sdv-only")
        with job_log.open("w", encoding="utf-8") as out:
            rc = subprocess.call(cmd, cwd=str(_ROOT), env=env, stdout=out, stderr=subprocess.STDOUT)
        log(f"done  seed={seed} {dataset}×{generator} rc={rc}")
        done += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
