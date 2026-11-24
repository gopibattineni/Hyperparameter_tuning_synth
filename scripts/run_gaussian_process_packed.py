#!/usr/bin/env python3
"""Pack Gaussian-Process (GPSampler) HPO across free GPUs.

Default scope: **classification datasets only** (7 remaining after cancer /
Alzheimer's), **7 generators** (no forest_diffusion). Set
``CLASSIFICATION_ONLY=0`` to include regression as well.

CPU / RAM safety (this host has ~8 cores / 23 GiB; 18× workers previously
SIGKILL'd jobs and hard-rebooted the box):

  * Default **1 worker per GPU** (override with WORKERS_PER_GPU).
  * Global ``MAX_WORKERS`` capped by CPU count and free RAM.
  * Every worker gets ``OMP/MKL/TORCH_*_THREADS=1``.

Usage::

    python scripts/run_gaussian_process_packed.py start
    python scripts/run_gaussian_process_packed.py status
    python scripts/run_gaussian_process_packed.py stop
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tuning.cpu_limits import thread_limit_env

PY = os.environ.get("SYNTH_PYTHON", "/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python")
SAMPLER = "gaussian_process"
RUN_DIR = _ROOT / "gaussian_process" / "packed_gpu"
QUEUE = RUN_DIR / "queue.txt"
PIDS = RUN_DIR / "pids"
LOGS = RUN_DIR / "logs"
MASTER = RUN_DIR / "launcher.log"

from multivariate_tpe.datasets_catalog import (
    ALL_DATASETS,
    CLASSIFICATION_DATASETS,
    GENERATORS_NO_FOREST,
    REGRESSION_DATASETS,
    assert_catalog,
)

DONE_DATASETS = ("cancer", "alzheimers")
GENERATORS = tuple(GENERATORS_NO_FOREST)

# Scope:
#   CLASSIFICATION_ONLY=1 (default) → 9 classification datasets
#   REGRESSION_ONLY=1               → 6 regression datasets
#   both 0                          → all 15
_truthy = ("1", "true", "True", "yes")
REGRESSION_ONLY = os.environ.get("REGRESSION_ONLY", "0").strip() in _truthy
CLASSIFICATION_ONLY = (
    not REGRESSION_ONLY
    and os.environ.get("CLASSIFICATION_ONLY", "1").strip() not in ("0", "false", "False", "no")
)

# VRAM packing (secondary). Primary limit is host CPU/RAM below.
TARGET_MB = int(os.environ.get("TARGET_GPU_MB", "2100"))
PROCESS_MB = int(os.environ.get("PROCESS_GPU_MB", "700"))
MAX_FOREIGN_USED_MB = int(os.environ.get("MAX_FOREIGN_USED_MB", "1500"))
# Lower default reserve so 1 worker/GPU does not waste VRAM needlessly.
RESERVE_MB = int(os.environ.get("SYNTH_CUDA_RESERVE_MB", "0"))

# Host safety: 1 slot/GPU by default (was 3 → 18 procs → reboot).
WORKERS_PER_GPU = int(os.environ.get("WORKERS_PER_GPU", "1"))
# Leave headroom for OS + Cursor; each HPO proc can peak ~1.5–2.5 GiB RSS.
PROCESS_RSS_MB = int(os.environ.get("PROCESS_RSS_MB", "2000"))
RAM_RESERVE_MB = int(os.environ.get("RAM_RESERVE_MB", "4096"))
# Hard cap; default = cpu_count - 1 so the machine stays interactive.
def _default_max_workers() -> int:
    cpus = os.cpu_count() or 4
    return max(1, cpus - 1)


MAX_WORKERS = int(os.environ.get("MAX_WORKERS", str(_default_max_workers())))


def _log(msg: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with MASTER.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _available_ram_mb() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        # /proc/meminfo fallback
        try:
            with open("/proc/meminfo", encoding="utf-8") as fh:
                info = {}
                for line in fh:
                    key, val = line.split(":")
                    info[key] = int(val.strip().split()[0])  # KiB
            # MemAvailable preferred
            avail_kib = info.get("MemAvailable") or info.get("MemFree", 0)
            return avail_kib // 1024
        except Exception:
            return 8192


def _nvidia_query() -> list[dict[str, int]]:
    r = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "used": int(float(parts[1])),
                "free": int(float(parts[2])),
                "total": int(float(parts[3])),
                "util": int(float(parts[4])),
            }
        )
    return rows


def _compute_apps() -> dict[int, list[tuple[int, int]]]:
    uuid_to_idx: dict[str, int] = {}
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    for line in r.stdout.strip().splitlines():
        idx, uuid = [x.strip() for x in line.split(",")]
        uuid_to_idx[uuid] = int(idx)
    out: dict[int, list[tuple[int, int]]] = {i: [] for i in uuid_to_idx.values()}
    r2 = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    for line in (r2.stdout or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid, uuid, mem = int(parts[0]), parts[1], int(float(parts[2]))
        except ValueError:
            continue
        if uuid in uuid_to_idx:
            out.setdefault(uuid_to_idx[uuid], []).append((pid, mem))
    return out


def _our_pids() -> set[int]:
    ours: set[int] = set()
    if PIDS.is_dir():
        for p in PIDS.glob("*.pid"):
            try:
                ours.add(int(p.read_text().strip()))
            except ValueError:
                pass
    r = subprocess.run(
        ["pgrep", "-f", "Hyperparameter_tuning_synth/gaussian_process/run.py"],
        capture_output=True,
        text=True,
    )
    for line in (r.stdout or "").split():
        try:
            ours.add(int(line))
        except ValueError:
            pass
    return ours


def remaining_datasets() -> list[str]:
    """Datasets still to queue (excludes completed cancer/alzheimers for cls scope)."""
    assert_catalog()
    if REGRESSION_ONLY:
        return list(REGRESSION_DATASETS)
    if CLASSIFICATION_ONLY:
        keys = list(CLASSIFICATION_DATASETS)
        return [k for k in keys if k not in DONE_DATASETS]
    return [k for k in ALL_DATASETS if k not in DONE_DATASETS]


def scoped_datasets() -> list[str]:
    """Full scope for status."""
    assert_catalog()
    if REGRESSION_ONLY:
        return list(REGRESSION_DATASETS)
    if CLASSIFICATION_ONLY:
        return list(CLASSIFICATION_DATASETS)
    return list(ALL_DATASETS)


def max_workers_budget() -> int:
    """Host-safe concurrent process budget from CPU + free RAM."""
    by_cpu = MAX_WORKERS
    avail = _available_ram_mb()
    by_ram = max(1, (avail - RAM_RESERVE_MB) // max(PROCESS_RSS_MB, 1))
    budget = max(1, min(by_cpu, by_ram))
    _log(
        f"host budget: cpus={os.cpu_count()} MAX_WORKERS={by_cpu} "
        f"RAM_avail≈{avail}MiB → by_ram={by_ram} → budget={budget}"
    )
    return budget


def select_gpus() -> list[tuple[int, int]]:
    """Return (gpu_index, workers_on_that_gpu), total slots ≤ host budget."""
    requested = os.environ.get("GPUS")
    apps = _compute_apps()
    ours = _our_pids()
    budget = max_workers_budget()
    per_gpu_cap = max(1, WORKERS_PER_GPU)

    selected: list[tuple[int, int]] = []
    remaining = budget
    for g in _nvidia_query():
        if remaining <= 0:
            break
        idx = g["index"]
        if requested is not None:
            allow = {int(x) for x in requested.split(",") if x.strip()}
            if idx not in allow:
                continue
        foreign = sum(mem for pid, mem in apps.get(idx, []) if pid not in ours)
        if foreign >= MAX_FOREIGN_USED_MB:
            _log(f"skip GPU {idx}: foreign VRAM {foreign} MiB >= {MAX_FOREIGN_USED_MB}")
            continue
        by_vram = max(
            1,
            min(
                per_gpu_cap,
                TARGET_MB // PROCESS_MB,
                max(1, (g["free"] - 400) // PROCESS_MB),
            ),
        )
        n = min(by_vram, remaining)
        selected.append((idx, n))
        remaining -= n
    return selected


def unfinished_jobs(*, include_done_datasets: bool = False) -> list[tuple[str, str]]:
    from tuning.artifacts import is_completed

    results = _ROOT / SAMPLER / "results"
    datasets = scoped_datasets() if include_done_datasets else remaining_datasets()
    jobs: list[tuple[str, str]] = []
    for ds in datasets:
        for gen in GENERATORS:
            if is_completed(results, ds, gen):
                continue
            jobs.append((ds, gen))
    return jobs


def write_queue(jobs: list[tuple[str, str]]) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PIDS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with QUEUE.open("w", encoding="utf-8") as f:
        for ds, gen in jobs:
            f.write(f"{ds} {gen}\n")
    _log(f"queue size={len(jobs)} → {QUEUE}")


def pop_job() -> tuple[str, str] | None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    lock = Path(str(QUEUE) + ".lock")
    for _ in range(100):
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            time.sleep(0.05)
    else:
        return None
    try:
        if not QUEUE.is_file():
            return None
        lines = QUEUE.read_text(encoding="utf-8").splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return None
        head, rest = lines[0], lines[1:]
        QUEUE.write_text("\n".join(rest) + ("\n" if rest else ""), encoding="utf-8")
        parts = head.split()
        if len(parts) != 2:
            return None
        return parts[0], parts[1]
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def stop_workers() -> None:
    patterns = [
        "Hyperparameter_tuning_synth/gaussian_process/run.py",
        "scripts/run_gaussian_process_packed.py worker",
    ]
    for pat in patterns:
        subprocess.run(["pkill", "-f", pat], check=False)
    if PIDS.is_dir():
        for p in PIDS.glob("*.pid"):
            try:
                pid = int(p.read_text().strip())
                os.kill(pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, OSError):
                pass
            p.unlink(missing_ok=True)
    time.sleep(2)
    for pat in patterns:
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(1)


def _worker_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["SYNTH_CUDA_RESERVE_MB"] = str(RESERVE_MB)
    env["SYNTH_REPO_ROOT"] = env.get("SYNTH_REPO_ROOT", "/home/gopi_b/SYNTH_BENCHMARK")
    env["SYNTH_PYTHON"] = PY
    env["SYNTH_NUM_THREADS"] = "1"
    env.update(thread_limit_env(1))
    return env


def launch_slot(gpu: int, slot: int) -> subprocess.Popen:
    env = _worker_env(gpu)
    tag = f"gpu{gpu}_slot{slot}"
    log = LOGS / f"{tag}.out"
    pidf = PIDS / f"{tag}.pid"
    cmd = [PY, "-u", str(Path(__file__).resolve()), "worker", str(gpu), str(slot)]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
        env=env,
        stdout=log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pidf.write_text(str(proc.pid), encoding="utf-8")
    _log(f"launch {tag} pid={proc.pid} reserve_mb={RESERVE_MB} threads=1 log={log}")
    return proc


def worker_loop(gpu: int, slot: int) -> int:
    _log(
        f"worker GPU={gpu} slot={slot} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"RESERVE={os.environ.get('SYNTH_CUDA_RESERVE_MB')} "
        f"OMP={os.environ.get('OMP_NUM_THREADS')}"
    )
    results = _ROOT / SAMPLER / "results"
    run_py = _ROOT / SAMPLER / "run.py"
    from tuning.artifacts import is_completed

    while True:
        # Bail if host RAM is critically low (avoid another hard reboot).
        avail = _available_ram_mb()
        if avail < RAM_RESERVE_MB // 2:
            _log(
                f"worker GPU={gpu} slot={slot} PAUSE: RAM_avail≈{avail}MiB < "
                f"{RAM_RESERVE_MB // 2} — sleeping 60s"
            )
            time.sleep(60)
            continue

        job = pop_job()
        if job is None:
            _log(f"worker GPU={gpu} slot={slot} queue empty — exit")
            return 0
        dataset, generator = job
        if is_completed(results, dataset, generator):
            _log(f"skip completed {dataset}/{generator}")
            continue
        cmd = [
            PY,
            "-u",
            str(run_py),
            "--datasets",
            dataset,
            "--generators",
            generator,
            "--quiet-trials",
        ]
        _log(f"GPU{gpu}/s{slot} START {dataset} {generator}")
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(_ROOT), env=os.environ.copy())
        _log(
            f"GPU{gpu}/s{slot} END {dataset} {generator} rc={rc} elapsed={time.time() - t0:.0f}s"
        )
    return 0


def cmd_start() -> int:
    stop_workers()
    jobs = unfinished_jobs()
    if not jobs:
        _log("nothing to run — all remaining pairs complete")
        return 0
    write_queue(jobs)
    gpus = select_gpus()
    if not gpus:
        _log("no free GPUs / host budget=0")
        return 1
    total_slots = sum(n for _, n in gpus)
    _log(
        f"gaussian_process packs (CPU-safe): WORKERS_PER_GPU={WORKERS_PER_GPU} "
        f"MAX_WORKERS={MAX_WORKERS} RESERVE/slot={RESERVE_MB}MiB → "
        f"GPUs={gpus} slots={total_slots} jobs={len(jobs)}"
    )
    for p in PIDS.glob("*.pid"):
        p.unlink()
    procs = []
    for gpu, n in gpus:
        for slot in range(n):
            procs.append(launch_slot(gpu, slot))
            time.sleep(0.5)
    summary = RUN_DIR / "launch_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "slots", "workers_per_gpu_cap", "max_workers", "reserve_mb"])
        for gpu, n in gpus:
            w.writerow([gpu, n, WORKERS_PER_GPU, MAX_WORKERS, RESERVE_MB])
    _log(f"launched {len(procs)} workers; summary={summary}")
    _log(f"datasets={remaining_datasets()}")
    _log(f"generators={list(GENERATORS)}")
    return 0


def cmd_status() -> int:
    jobs_left = 0
    if QUEUE.is_file():
        jobs_left = sum(1 for line in QUEUE.read_text().splitlines() if line.strip())
    print(f"queue_remaining={jobs_left}")
    print(f"RAM_avail_MiB≈{_available_ram_mb()}  load={os.getloadavg()}")
    alive = 0
    if PIDS.is_dir():
        for p in sorted(PIDS.glob("*.pid")):
            try:
                pid = int(p.read_text().strip())
            except ValueError:
                continue
            running = True
            try:
                os.kill(pid, 0)
            except OSError:
                running = False
            print(f"  {p.name}: pid={pid} {'alive' if running else 'dead'}")
            alive += int(running)
    print(f"workers_alive={alive}")
    unfinished = unfinished_jobs(include_done_datasets=True)
    scope = scoped_datasets()
    total_pairs = len(scope) * len(GENERATORS)
    scope_name = (
        "regression"
        if REGRESSION_ONLY
        else ("classification" if CLASSIFICATION_ONLY else "all")
    )
    print(
        f"catalogue: classification={len(CLASSIFICATION_DATASETS)} "
        f"regression={len(REGRESSION_DATASETS)} "
        f"scope={scope_name} generators={len(GENERATORS)}"
    )
    print(f"pairs_unfinished={len(unfinished)} / {total_pairs}")
    from collections import Counter

    c = Counter(ds for ds, _ in unfinished)
    if not REGRESSION_ONLY:
        print("classification (9):")
        for ds in CLASSIFICATION_DATASETS:
            if ds not in scope:
                continue
            rem = c.get(ds, 0)
            print(f"  {ds:20} {len(GENERATORS) - rem}/{len(GENERATORS)} done  ({rem} left)")
    if REGRESSION_ONLY or not CLASSIFICATION_ONLY:
        print("regression (6):")
        for ds in REGRESSION_DATASETS:
            if ds not in scope:
                continue
            rem = c.get(ds, 0)
            print(f"  {ds:20} {len(GENERATORS) - rem}/{len(GENERATORS)} done  ({rem} left)")
    print("--- GPUs ---")
    apps = _compute_apps()
    ours = _our_pids()
    for g in _nvidia_query():
        our_mem = sum(m for pid, m in apps.get(g["index"], []) if pid in ours)
        print(
            f"  GPU{g['index']}: util={g['util']}% used={g['used']}MiB "
            f"free={g['free']}MiB our≈{our_mem}MiB"
        )
    return 0


def cmd_stop() -> int:
    stop_workers()
    _log("stopped gaussian_process packed workers")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "status", "stop", "worker"])
    p.add_argument("gpu", nargs="?", type=int, default=None)
    p.add_argument("slot", nargs="?", type=int)
    args = p.parse_args()
    if args.action == "start":
        return cmd_start()
    if args.action == "status":
        return cmd_status()
    if args.action == "stop":
        return cmd_stop()
    if args.action == "worker":
        if args.gpu is None or args.slot is None:
            raise SystemExit("worker requires gpu slot")
        return worker_loop(args.gpu, args.slot)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
