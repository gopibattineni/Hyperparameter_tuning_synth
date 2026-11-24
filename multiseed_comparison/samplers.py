"""Sampler factory for the multi-seed comparison."""

from __future__ import annotations

from typing import Any

import optuna
from optuna.samplers import GPSampler, TPESampler

from multiseed_comparison import N_STARTUP_TRIALS


def make_sampler(name: str, seed: int, *, n_startup_trials: int = N_STARTUP_TRIALS) -> Any:
    """Build an Optuna sampler by name.

    Parameters
    ----------
    name:
        ``standard_tpe`` | ``multivariate_tpe`` | ``gp``.
    seed:
        Sampler RNG seed (paired across samplers for the same run index).
    n_startup_trials:
        Shared random-startup budget so only the model-based phase differs.

    Notes
    -----
    Multivariate TPE does **not** set ``group=True``: generator YAML search
    spaces are flat (no Optuna conditional parameters), so grouping is a no-op.
    """
    seed = int(seed)
    n_startup = int(n_startup_trials)
    key = name.strip().lower().replace("-", "_")

    if key in {"standard_tpe", "tpe", "st"}:
        return TPESampler(
            seed=seed,
            multivariate=False,
            n_startup_trials=n_startup,
        )
    if key in {"multivariate_tpe", "mtpe", "mt"}:
        return TPESampler(
            seed=seed,
            multivariate=True,
            n_startup_trials=n_startup,
            # group=False: no conditional search-space structure in YAML.
        )
    if key in {"gp", "gpsampler", "gaussian_process"}:
        return GPSampler(
            seed=seed,
            n_startup_trials=n_startup,
            deterministic_objective=False,
            warn_independent_sampling=True,
        )
    raise ValueError(
        f"Unknown sampler {name!r}. Expected one of: standard_tpe, multivariate_tpe, gp"
    )


def ensure_gp_torch_compat() -> str | None:
    """Apply the existing torch<2.1 GPSampler patch if needed."""
    try:
        from tuning.alt_sampler_runner import apply_torch_device_compat

        return apply_torch_device_compat()
    except Exception as exc:  # pragma: no cover
        return f"GP torch compat skipped: {type(exc).__name__}: {exc}"
