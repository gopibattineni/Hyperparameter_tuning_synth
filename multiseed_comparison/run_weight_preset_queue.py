#!/usr/bin/env python3
"""GPU worker for O2–O5 multiseed queues (same setup as O1).

Queue TSV columns (tab-separated)::

    preset\\tsampler\\tseed\\tdataset\\tgenerator

Each job calls ``multiseed_comparison/run.py`` with ``--objective-preset``,
``--all-10-seeds`` seeds already expanded per line, and a **separate**
results root ``results/multiseed_weights_{PRESET}/`` so O1 studies are never
touched. Fidelity defaults to O1 legacy (no ``--force-sdv-only``).
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
    p.add_argument(
        "--results-root-template",
        type=str,
        default="results/multiseed_weights_{preset}",
        help="Format string with {preset} placeholder",
    )
    p.add_argument("--gpu", type=str, required=True)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-startup-trials", type=int, default=10)
    p.add_argument("--trial-timeout-s", type=float, default=7200.0)
    p.add_argument("--log-dir", type=str, required=True)
    p.add_argument(
        "--force-sdv-only",
        action="store_true",
        default=True,
        help="Use fidelity.aggregation=sdv_only (default; KS excluded from F).",
    )
    p.add_argument(
        "--legacy-composite-fidelity",
        action="store_true",
        help="Replay legacy_composite fidelity (includes KS). Not for O1–O5.",
    )
    p.add_argument(
        "--gpu-only",
        action="store_true",
        default=True,
        help="Skip CPU-only generators (gaussian_copula); park them on --cpu-queue (default on)",
    )
    p.add_argument(
        "--allow-cpu-generators",
        action="store_true",
        help="Disable --gpu-only (allows gaussian_copula on this worker)",
    )
    p.add_argument(
        "--cpu-queue",
        type=str,
        default=None,
        help="Where to append skipped CPU-only jobs (default <log-dir>/queue_cpu_only.tsv)",
    )
    args = p.parse_args()
    gpu_only = bool(args.gpu_only) and not bool(args.allow_cpu_generators)

    queue_path = Path(args.queue)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    worker_log = log_dir / f"weights_gpu{args.gpu}.log"
    cpu_queue = Path(args.cpu_queue) if args.cpu_queue else (log_dir / "queue_cpu_only.tsv")

    # Import here so workers fail fast if path is wrong
    sys.path.insert(0, str(_ROOT))
    from generators.device import CPU_ONLY_GENERATORS

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
        env[k] = "1"
    env["JOBLIB_MULTIPROCESSING"] = "0"
    env["SYNTH_REPO_ROOT"] = "/home/gopi_b/SYNTH_BENCHMARK"
    # Hard-require CUDA for neural gens — never silent CPU fallback.
    env["SYNTH_REQUIRE_CUDA"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with worker_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def park_cpu_job(line: str) -> None:
        cpu_queue.parent.mkdir(parents=True, exist_ok=True)
        with cpu_queue.open("a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(line.rstrip() + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    log(
        f"worker start gpu={args.gpu} queue={queue_path} "
        f"gpu_only={gpu_only} SYNTH_REQUIRE_CUDA=1 CUDA_VISIBLE_DEVICES={args.gpu}"
    )
    done = 0
    parked = 0
    while True:
        claimed = _claim_line(queue_path)
        if not claimed:
            log(f"queue empty — exit (finished {done} jobs, parked_cpu={parked})")
            break
        parts = claimed.split("\t")
        if len(parts) != 5:
            log(f"skip bad line: {claimed!r}")
            continue
        preset, sampler, seed, dataset, generator = parts
        preset = preset.upper()
        if gpu_only and generator in CPU_ONLY_GENERATORS:
            park_cpu_job(claimed)
            parked += 1
            log(f"park CPU-only {generator} → {cpu_queue.name} ({preset} {sampler} seed={seed} {dataset})")
            continue
        results_root = args.results_root_template.format(preset=preset)
        job_log = (
            log_dir
            / f"w_{preset}_{sampler}_seed{seed}_{dataset}_{generator}_gpu{args.gpu}.log"
        )
        log(
            f"claim {preset} {sampler} seed={seed} {dataset}×{generator} → {job_log.name}"
        )
        cmd = [
            PY,
            str(_ROOT / "multiseed_comparison" / "run.py"),
            "--objective-preset",
            preset,
            "--samplers",
            sampler,
            "--seeds",
            str(seed),
            "--datasets",
            dataset,
            "--generators",
            generator,
            "--results-root",
            results_root,
            "--n-trials",
            str(args.n_trials),
            "--n-startup-trials",
            str(args.n_startup_trials),
            "--trial-timeout-s",
            str(args.trial_timeout_s),
            "--no-analyze",
            "--gpu",
            str(args.gpu),
        ]
        if args.legacy_composite_fidelity:
            cmd.append("--legacy-composite-fidelity")
        else:
            cmd.append("--force-sdv-only")
        with job_log.open("w", encoding="utf-8") as out:
            rc = subprocess.call(
                cmd, cwd=str(_ROOT), env=env, stdout=out, stderr=subprocess.STDOUT
            )
        log(
            f"done  {preset} {sampler} seed={seed} {dataset}×{generator} rc={rc}"
        )
        done += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
