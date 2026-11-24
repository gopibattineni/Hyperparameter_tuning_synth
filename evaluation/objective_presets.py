"""Resolve named objective-weight presets (O1–O5) for sensitivity experiments.

Presets live in ``config/objective.yaml`` under ``presets:``. Resolving a preset
returns an independent objective config copy with:
  - selected weights
  - ``fidelity.aggregation = sdv_only`` (KS excluded from F / J)
  - ``fidelity.require_sdv = true``
  - ``direction = maximize``
  - validated weights summing to 1 (±1e-6)

All O1–O5 presets use SDV-only fidelity. ``force_sdv_only=False`` is retained
only for rare replay of legacy_composite runs.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

from config_utils import load_config

PRESET_NAMES = ("O1", "O2", "O3", "O4", "O5")
WEIGHT_SUM_TOL = 1e-6


def validate_weights(weights: Mapping[str, Any], *, tol: float = WEIGHT_SUM_TOL) -> dict[str, float]:
    """Return normalized weight dict; raise if keys missing or sum ≠ 1."""
    required = ("fidelity", "utility", "privacy_risk")
    out: dict[str, float] = {}
    for key in required:
        if key not in weights:
            # Accept legacy ``privacy`` alias
            if key == "privacy_risk" and "privacy" in weights:
                out[key] = float(weights["privacy"])
                continue
            raise ValueError(f"Missing weight key {key!r} in {dict(weights)}")
        out[key] = float(weights[key])
    total = out["fidelity"] + out["utility"] + out["privacy_risk"]
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Objective weights must sum to 1.0 (±{tol}); got sum={total} for {out}"
        )
    return out


def list_presets(base_cfg: Mapping[str, Any] | None = None) -> list[str]:
    cfg = dict(base_cfg or load_config("objective"))
    presets = cfg.get("presets") or {}
    names = [n for n in PRESET_NAMES if n in presets]
    # Include any extra named presets in YAML (stable order: O* first then rest)
    extras = sorted(k for k in presets if k not in PRESET_NAMES)
    return names + extras


def resolve_preset(
    name: str,
    *,
    base_cfg: Mapping[str, Any] | None = None,
    force_sdv_only: bool = True,
) -> dict[str, Any]:
    """Deep-copy objective config with preset weights and sensitivity fidelity mode."""
    key = str(name).strip().upper()
    cfg = copy.deepcopy(dict(base_cfg or load_config("objective")))
    presets = cfg.get("presets") or {}
    if key not in presets:
        raise KeyError(
            f"Unknown objective preset {name!r}. Available: {list_presets(cfg)}"
        )
    weights = validate_weights(presets[key])
    cfg["weights"] = weights
    cfg["direction"] = "maximize"
    cfg["preset"] = key
    fid = dict(cfg.get("fidelity") or {})
    if force_sdv_only:
        fid["aggregation"] = "sdv_only"
        fid["require_sdv"] = True
    cfg["fidelity"] = fid
    # Re-validate after merge
    validate_weights(cfg["weights"])
    return cfg
