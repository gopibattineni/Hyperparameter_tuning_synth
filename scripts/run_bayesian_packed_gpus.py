#!/usr/bin/env python3
"""Pack Bayesian HPO workers onto free GPUs (~2.0–2.3 GiB each).

Each neural-net trial process uses ~0.7 GiB on GTX 1070, so we run
``workers_per_gpu ≈ 3`` concurrent processes per free card to land near
2.1–2.3 GiB while fanning unfinished (sampler × dataset × generator) pairs
across all available GPUs.

Usage::

    python scripts/run_bayesian_packed_gpus.py start
    python scripts/run_bayesian_packed_gpus.py status
    python scripts/run_bayesian_packed_gpus.py stop
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

PY = os.environ.get("SYNTH_PYTHON", "/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python")
RUN_DIR = _ROOT / "sampler_comparison" / "packed_gpu"
QUEUE = RUN_DIR / "queue.txt"
LOCK = RUN_DIR / "queue.lock"
PIDS = RUN_DIR / "pids"
LOGS = RUN_DIR / "logs"
MASTER = RUN_DIR / "launcher.log"

SAMPLERS = ("multivariate_tpe", "gaussian_process")
DATASETS = ("cancer", "alzheimers")
# GPU-heavy only on CUDA slots. forest_diffusion is CPU — separate queue.
GPU_GENERATORS = (
    "ctabgan",
    "tvae",
    "wgan_gp",
    "tabddpm",
)
CPU_GENERATORS = ("forest_diffusion",)

# Target resident VRAM per GPU for *our* workers (~2.0–2.3 GiB).
TARGET_MB = int(os.environ.get("TARGET_GPU_MB", "2100"))
PROCESS_MB = int(os.environ.get("PROCESS_GPU_MB", "700"))
# Skip cards already heavily used by foreign jobs (leave them alone).
MAX_FOREIGN_USED_MB = int(os.environ.get("MAX_FOREIGN_USED_MB", "1500"))
# Default 0: avoid pinning VRAM when packing lightly; set SYNTH_CUDA_RESERVE_MB to raise.
RESERVE_MB = int(os.environ.get("SYNTH_CUDA_RESERVE_MB", "0"))
CPU_SLOTS = int(os.environ.get("CPU_SLOTS", "1"))
# Host safety (8-core / 23 GiB box previously rebooted with 18 workers).
WORKERS_PER_GPU = int(os.environ.get("WORKERS_PER_GPU", "1"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", str(max(1, (os.cpu_count() or 4) - 1))))
PROCESS_RSS_MB = int(os.environ.get("PROCESS_RSS_MB", "2000"))
RAM_RESERVE_MB = int(os.environ.get("RAM_RESERVE_MB", "4096"))


def _log(msg: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    with MASTER.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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
    """Map physical GPU index → list of (pid, used_mb)."""
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
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
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
    # Also match run.py under experiment folders
    r = subprocess.run(["pgrep", "-f", "Hyperparameter_tuning_synth/.*/run.py"], capture_output=True, text=True)
    for line in (r.stdout or "").split():
        try:
            ours.add(int(line))
        except ValueError:
            pass
    return ours


def select_gpus() -> list[tuple[int, int]]:
    """Return list of (gpu_index, workers_per_gpu) for free cards.

    Caps total slots by CPU count and free host RAM so packed runs cannot
    thrash an 8-core / 23 GiB machine (previous 18-worker pack hard-rebooted).
    """
    requested = os.environ.get("GPUS")
    apps = _compute_apps()
    ours = _our_pids()
    try:
        import psutil

        avail_mb = int(psutil.virtual_memory().available / (1024 * 1024))
    except Exception:
        avail_mb = 8192
    by_ram = max(1, (avail_mb - RAM_RESERVE_MB) // max(PROCESS_RSS_MB, 1))
    budget = max(1, min(MAX_WORKERS, by_ram))
    _log(f"host budget: MAX_WORKERS={MAX_WORKERS} RAM_avail≈{avail_mb}MiB → budget={budget}")

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
        n = max(1, TARGET_MB // PROCESS_MB)
        by_free = max(1, (g["free"] - 400) // PROCESS_MB)
        n = max(1, min(n, by_free, WORKERS_PER_GPU, remaining))
        selected.append((idx, n))
        remaining -= n
    return selected


def unfinished_jobs(generators: tuple[str, ...] | None = None) -> list[tuple[str, str, str]]:
    from tuning.artifacts import is_completed

    gens = generators or (GPU_GENERATORS + CPU_GENERATORS)
    jobs: list[tuple[str, str, str]] = []
    for sampler in SAMPLERS:
        results = _ROOT / sampler / "results"
        for ds in DATASETS:
            for gen in gens:
                if is_completed(results, ds, gen):
                    continue
                jobs.append((sampler, ds, gen))
    return jobs


def write_queue(jobs: list[tuple[str, str, str]], path: Path | None = None) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    PIDS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    qpath = path or QUEUE
    with qpath.open("w", encoding="utf-8") as f:
        for sampler, ds, gen in jobs:
            f.write(f"{sampler} {ds} {gen}\n")
    _log(f"queue size={len(jobs)} → {qpath}")


def pop_job(queue_path: Path | None = None) -> tuple[str, str, str] | None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    qpath = queue_path or QUEUE
    lock = Path(str(qpath) + ".lock")
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
        if not qpath.is_file():
            return None
        lines = qpath.read_text(encoding="utf-8").splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return None
        head, rest = lines[0], lines[1:]
        qpath.write_text("\n".join(rest) + ("\n" if rest else ""), encoding="utf-8")
        parts = head.split()
        if len(parts) != 3:
            return None
        return parts[0], parts[1], parts[2]
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def stop_legacy_workers() -> None:
    """Stop old sequential 2-GPU / packed workers we own."""
    patterns = [
        "Hyperparameter_tuning_synth/multivariate_tpe/run.py",
        "Hyperparameter_tuning_synth/gaussian_process/run.py",
        "scripts/run_bayesian_packed_gpus.py worker",
    ]
    for pat in patterns:
        subprocess.run(["pkill", "-f", pat], check=False)
    # Clean legacy pid files
    for d in (_ROOT / "multivariate_tpe" / "logs", _ROOT / "gaussian_process" / "logs"):
        if d.is_dir():
            for p in d.glob("*_gpu*.pid"):
                try:
                    pid = int(p.read_text().strip())
                    os.kill(pid, signal.SIGTERM)
                except (ValueError, ProcessLookupError, OSError):
                    pass
                p.unlink(missing_ok=True)
    time.sleep(2)
    # force leftovers
    for pat in patterns:
        subprocess.run(["pkill", "-9", "-f", pat], check=False)
    time.sleep(1)


def fail_running_trials() -> None:
    try:
        import optuna
        from optuna.trial import TrialState
    except Exception as exc:
        _log(f"skip fail_running_trials: {exc}")
        return
    for sampler in SAMPLERS:
        for db in (_ROOT / sampler).rglob("*/optuna.db"):
            if "_smoke" in str(db):
                continue
            storage = f"sqlite:///{db.resolve()}"
            try:
                summaries = optuna.study.get_all_study_summaries(storage=storage)
                if not summaries:
                    continue
                study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
                n = 0
                for t in study.trials:
                    if t.state == TrialState.RUNNING:
                        study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
                        n += 1
                if n:
                    _log(f"marked FAIL {n} RUNNING trials in {db}")
            except Exception as exc:
                _log(f"optuna cleanup skip {db}: {exc}")


def launch_slot(gpu: int | str, slot: int, *, cpu: bool = False) -> subprocess.Popen:
    from tuning.cpu_limits import thread_limit_env

    env = os.environ.copy()
    if cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["SYNTH_CUDA_RESERVE_MB"] = "0"
        tag = f"cpu_slot{slot}"
        worker_args = ["worker", "cpu", str(slot)]
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["SYNTH_CUDA_RESERVE_MB"] = str(RESERVE_MB)
        tag = f"gpu{gpu}_slot{slot}"
        worker_args = ["worker", str(gpu), str(slot)]
    env["SYNTH_REPO_ROOT"] = env.get("SYNTH_REPO_ROOT", "/home/gopi_b/SYNTH_BENCHMARK")
    env["SYNTH_PYTHON"] = PY
    env["SYNTH_NUM_THREADS"] = "1"
    env.update(thread_limit_env(1))
    env["PACKED_QUEUE"] = str(RUN_DIR / ("cpu_queue.txt" if cpu else "queue.txt"))
    log = LOGS / f"{tag}.out"
    pidf = PIDS / f"{tag}.pid"
    cmd = [PY, "-u", str(Path(__file__).resolve()), *worker_args]
    proc = subprocess.Popen(
        cmd,
        cwd=str(_ROOT),
        env=env,
        stdout=log.open("w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    pidf.write_text(str(proc.pid), encoding="utf-8")
    _log(f"launch {tag} pid={proc.pid} reserve_mb={env.get('SYNTH_CUDA_RESERVE_MB')} log={log}")
    return proc


def worker_loop(gpu: int | str, slot: int) -> int:
    """Pull jobs until queue empty; each job is one sampler×dataset×generator."""
    qpath = Path(os.environ.get("PACKED_QUEUE", str(QUEUE)))
    _log(
        f"worker GPU={gpu} slot={slot} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"RESERVE={os.environ.get('SYNTH_CUDA_RESERVE_MB')} queue={qpath}"
    )
    while True:
        job = pop_job(qpath)
        if job is None:
            _log(f"worker GPU={gpu} slot={slot} queue empty — exit")
            return 0
        sampler, dataset, generator = job
        results = _ROOT / sampler / "results"
        from tuning.artifacts import is_completed

        if is_completed(results, dataset, generator):
            _log(f"skip completed {sampler}/{dataset}/{generator}")
            continue
        run_py = _ROOT / sampler / "run.py"
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
        _log(f"GPU{gpu}/s{slot} START {sampler} {dataset} {generator}")
        t0 = time.time()
        rc = subprocess.call(cmd, cwd=str(_ROOT), env=os.environ.copy())
        _log(
            f"GPU{gpu}/s{slot} END {sampler} {dataset} {generator} "
            f"rc={rc} elapsed={time.time()-t0:.0f}s"
        )
    return 0


def cmd_start() -> int:
    stop_legacy_workers()
    fail_running_trials()
    gpu_jobs = unfinished_jobs(GPU_GENERATORS)
    cpu_jobs = unfinished_jobs(CPU_GENERATORS)
    if not gpu_jobs and not cpu_jobs:
        _log("nothing to run — all pairs complete")
        return 0
    write_queue(gpu_jobs, QUEUE)
    write_queue(cpu_jobs, RUN_DIR / "cpu_queue.txt")
    gpus = select_gpus()
    if not gpus and gpu_jobs:
        _log("no free GPUs for GPU queue")
        return 1
    total_slots = sum(n for _, n in gpus)
    _log(
        f"packing TARGET={TARGET_MB}MiB PROCESS≈{PROCESS_MB}MiB RESERVE/slot={RESERVE_MB}MiB → "
        f"GPUs={gpus} total_slots={total_slots} gpu_jobs={len(gpu_jobs)} cpu_jobs={len(cpu_jobs)}"
    )
    for p in PIDS.glob("*.pid"):
        p.unlink()
    procs = []
    for gpu, n in gpus:
        for slot in range(n):
            procs.append(launch_slot(gpu, slot, cpu=False))
            time.sleep(0.35)
    for slot in range(CPU_SLOTS if cpu_jobs else 0):
        procs.append(launch_slot("cpu", slot, cpu=True))
        time.sleep(0.2)
    summary = RUN_DIR / "launch_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gpu", "slots", "target_mb", "reserve_mb_per_slot"])
        for gpu, n in gpus:
            w.writerow([gpu, n, n * RESERVE_MB, RESERVE_MB])
    _log(f"launched {len(procs)} workers; summary={summary}")
    return 0


def cmd_status() -> int:
    jobs_left = 0
    if QUEUE.is_file():
        jobs_left = sum(1 for line in QUEUE.read_text().splitlines() if line.strip())
    print(f"queue_remaining={jobs_left}")
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
    unfinished = unfinished_jobs()
    print(f"pairs_unfinished={len(unfinished)}")
    for row in unfinished:
        print(" ", "/".join(row))
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
    stop_legacy_workers()
    if PIDS.is_dir():
        for p in PIDS.glob("*.pid"):
            try:
                pid = int(p.read_text().strip())
                os.kill(pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, OSError):
                pass
            p.unlink(missing_ok=True)
    time.sleep(1)
    subprocess.run(["pkill", "-9", "-f", "scripts/run_bayesian_packed_gpus.py worker"], check=False)
    _log("stopped packed workers")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "status", "stop", "worker"])
    p.add_argument("gpu", nargs="?", default=None, help="GPU index or 'cpu'")
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
        gpu: int | str
        if args.gpu == "cpu":
            gpu = "cpu"
        else:
            gpu = int(args.gpu)
        return worker_loop(gpu, args.slot)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
