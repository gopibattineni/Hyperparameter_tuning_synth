"""Cap BLAS / OpenMP / Torch threads so packed HPO workers do not thrash the host.

With many concurrent Optuna processes, libraries that default to ``n_cpus`` threads
(OpenBLAS, MKL, Torch, XGBoost) can spawn 18×8 = 144+ threads on an 8-core box and
push the machine into OOM / hard reboot. Call ``apply_thread_limits()`` early in
every worker process, and set the same env vars in the launcher before spawn.
"""

from __future__ import annotations

import os
from typing import Mapping


# Env vars read by NumPy/SciPy/Torch/XGBoost/joblib before they spawn pools.
_THREAD_ENV_DEFAULTS: Mapping[str, str] = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "TORCH_NUM_THREADS": "1",
    "TORCH_NUM_INTEROP_THREADS": "1",
    "FOREST_XGB_NTHREAD": "1",
    "JOBLIB_MULTIPROCESSING": "0",
}


def thread_limit_env(n: int = 1) -> dict[str, str]:
    """Return env mapping suitable for ``subprocess`` / ``os.environ.update``."""
    s = str(max(1, int(n)))
    return {k: s if k != "JOBLIB_MULTIPROCESSING" else "0" for k in _THREAD_ENV_DEFAULTS}


def apply_thread_limits(n: int = 1) -> str:
    """Pin process-wide thread pools. Safe to call multiple times.

    Returns a short note for logging.
    """
    n = max(1, int(n))
    for key, val in thread_limit_env(n).items():
        # Do not override an explicit user choice already set higher in the env.
        os.environ.setdefault(key, val)

    notes: list[str] = [f"thread_env=1"]
    try:
        import torch

        torch.set_num_threads(n)
        # Interop threads must be set before any parallel work; ignore if too late.
        try:
            torch.set_num_interop_threads(n)
        except RuntimeError:
            pass
        notes.append(f"torch_threads={torch.get_num_threads()}")
    except Exception as exc:  # pragma: no cover
        notes.append(f"torch_skip={type(exc).__name__}")

    try:
        import cv2  # type: ignore

        cv2.setNumThreads(n)
        notes.append("cv2_threads=1")
    except Exception:
        pass

    return "CPU limits: " + ", ".join(notes)
