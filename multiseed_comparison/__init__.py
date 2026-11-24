"""Multi-seed sampler comparison: Standard TPE vs Multivariate TPE vs GPSampler."""

from __future__ import annotations

__all__ = [
    "SEEDS",
    "ALL_SEEDS_10",
    "SAMPLER_NAMES",
    "OBJECTIVE_PRESETS",
    "DEFAULT_N_TRIALS",
    "N_STARTUP_TRIALS",
]

SEEDS: list[int] = [42, 123, 2024]
# Full 10-seed set used for the curated O1 multiseed report (multiseed_10seeds_all).
ALL_SEEDS_10: list[int] = [42, 123, 2024, 7, 99, 1337, 13, 21, 77, 314]
SAMPLER_NAMES: tuple[str, ...] = ("standard_tpe", "multivariate_tpe", "gp")
DEFAULT_N_TRIALS: int = 20
N_STARTUP_TRIALS: int = 10
OBJECTIVE_PRESETS: tuple[str, ...] = ("O1", "O2", "O3", "O4", "O5")
