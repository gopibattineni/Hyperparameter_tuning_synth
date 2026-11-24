"""Reproducible seeding for multi-seed HPO studies."""

from __future__ import annotations

import os
import random


def set_global_seed(seed: int) -> None:
    """Seed ``random``, NumPy, and Torch (CPU + CUDA) for a study.

    Call once at the start of each (dataset, generator, sampler, seed) study.
    Data splits remain the YAML-fixed split (see project constraint); this seed
    controls sampler + generator/training stochasticity only.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN can slow GANs; keep default nondeterministic kernels
        # but still seed generators. Callers may set CUBLAS_WORKSPACE_CONFIG.
    except Exception:
        pass


def trial_seed(study_seed: int, trial_number: int) -> int:
    """Derive a per-trial seed so a single trial can be reproduced alone."""
    # Stable mix; stays in uint32 range for libraries that expect that.
    return int((int(study_seed) * 1_000_003 + int(trial_number) * 97) % (2**31 - 1))
