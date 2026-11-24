"""Combine fidelity / utility / privacy_risk into a scalar Optuna score.

Default formula (maximize)::

    Objective = 0.4 × Fidelity + 0.4 × Utility − 0.2 × PrivacyRisk

All three components are assumed normalized to ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any, Mapping

from .preprocess import clip01


DEFAULT_WEIGHTS = {
    "fidelity": 0.4,
    "utility": 0.4,
    "privacy_risk": 0.2,
}


def _resolve_weights(cfg: Mapping[str, Any]) -> dict[str, float]:
    weights = dict(DEFAULT_WEIGHTS)
    w_cfg = cfg.get("weights") or {}
    # Support both legacy ``privacy`` weight and explicit ``privacy_risk``
    if "fidelity" in w_cfg:
        weights["fidelity"] = float(w_cfg["fidelity"])
    if "utility" in w_cfg:
        weights["utility"] = float(w_cfg["utility"])
    if "privacy_risk" in w_cfg:
        weights["privacy_risk"] = float(w_cfg["privacy_risk"])
    elif "privacy" in w_cfg:
        weights["privacy_risk"] = float(w_cfg["privacy"])
    return weights


def compute_objective(
    fidelity: Mapping[str, float],
    privacy: Mapping[str, float],
    utility: Mapping[str, float],
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Normalize components and compute the scalar objective.

    Returns
    -------
    dict
        Includes ``objective``, ``fidelity``, ``utility``, ``privacy_risk``.
    """
    cfg = cfg or {}
    weights = _resolve_weights(cfg)

    raw_f = fidelity.get("fidelity", 0.0)
    # Refuse to silently score when fidelity was marked unavailable (NaN).
    try:
        raw_f_f = float(raw_f)
    except (TypeError, ValueError):
        raw_f_f = float("nan")
    if raw_f_f != raw_f_f:  # NaN
        raise ValueError(
            "Cannot compute objective: fidelity is unavailable (NaN). "
            "Under sdv_only / require_sdv the trial must be failed or pruned."
        )

    f = clip01(raw_f_f)
    u = clip01(float(utility.get("utility", 0.0)))
    r = clip01(float(privacy.get("privacy_risk", 0.5)))

    objective = (
        weights["fidelity"] * f
        + weights["utility"] * u
        - weights["privacy_risk"] * r
    )

    return {
        "objective": float(objective),
        "fidelity": f,
        "utility": u,
        "privacy_risk": r,
        "w_fidelity": weights["fidelity"],
        "w_utility": weights["utility"],
        "w_privacy_risk": weights["privacy_risk"],
    }
