"""Objective-weight sensitivity pilot (Standard TPE, presets O1–O5).

Does not modify historical multiseed results or the default data loader.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_N_TRIALS",
    "DEFAULT_SEED",
    "N_STARTUP_TRIALS",
    "SAMPLER_NAME",
]

SAMPLER_NAME = "standard_tpe"
DEFAULT_N_TRIALS = 5
DEFAULT_SEED = 42
N_STARTUP_TRIALS = 3
