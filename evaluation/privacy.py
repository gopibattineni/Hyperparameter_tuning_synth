"""Privacy evaluation — returns ``privacy_risk`` ∈ [0, 1] (higher = worse).

Weighted mix (from ``config/objective.yaml``):
  - MIA AUC (risk)
  - NNDR (safety → risk = 1 − normalized NNDR)
  - DCR (safety → risk = 1 − normalized DCR)
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

from .preprocess import align_columns, build_feature_matrix, clip01


def _pairwise_nn_distances(a: np.ndarray, b: np.ndarray, k: int = 1) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(b)
    dists, _ = nn.kneighbors(a, return_distance=True)
    return dists[:, 0]


def _dcr_score(real_X: np.ndarray, synth_X: np.ndarray) -> float:
    """Median distance from synthetic rows to nearest real row (higher safer)."""
    if len(real_X) == 0 or len(synth_X) == 0:
        return 0.0
    d = _pairwise_nn_distances(synth_X, real_X, k=1)
    return float(np.median(d))


def _nndr_score(real_X: np.ndarray, synth_X: np.ndarray) -> float:
    """Median ratio d1/d2 of synth→real nearest distances (higher safer)."""
    if len(real_X) < 2 or len(synth_X) == 0:
        return 0.0
    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(real_X)
    dists, _ = nn.kneighbors(synth_X, return_distance=True)
    d1, d2 = dists[:, 0], dists[:, 1]
    ratio = np.divide(d1, np.maximum(d2, 1e-12))
    return float(np.median(ratio))


def _mia_auc(real_X: np.ndarray, synth_X: np.ndarray, seed: int = 42) -> float:
    """Simple membership-inference AUC via RF shadow attack (higher = riskier)."""
    n = min(len(real_X), len(synth_X), 2000)
    if n < 20:
        return 0.5
    rng = np.random.default_rng(seed)
    r_idx = rng.choice(len(real_X), size=n, replace=False)
    s_idx = rng.choice(len(synth_X), size=n, replace=False)
    X = np.vstack([real_X[r_idx], synth_X[s_idx]])
    y = np.array([1] * n + [0] * n)  # 1 = member (real)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )
        clf = RandomForestClassifier(
            n_estimators=50, max_depth=8, random_state=seed, n_jobs=1
        )
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        return float(roc_auc_score(y_te, proba))
    except Exception:
        return 0.5


def _normalize_distance(value: float, ref_scale: float) -> float:
    """Map a non-negative distance-like score into roughly [0, 1]."""
    if ref_scale <= 1e-12:
        return 0.0
    return clip01(value / (value + ref_scale))


def evaluate_privacy(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: Any = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return privacy metrics including ``privacy_risk`` ∈ [0, 1] (higher worse)."""
    cfg = cfg or {}
    priv_cfg = cfg.get("privacy", {}) if isinstance(cfg, Mapping) else {}
    metric_cfg = priv_cfg.get("metrics") or {
        "mia_auc": {"weight": 0.5, "invert": False},  # already a risk
        "nndr": {"weight": 0.3, "invert": True},
        "dcr": {"weight": 0.2, "invert": True},
    }

    target = getattr(metadata, "target", None)
    real, synth = align_columns(real_train, synthetic)
    if target and target in real.columns:
        real_X = real.drop(columns=[target])
        synth_X = synth.drop(columns=[target], errors="ignore")
    else:
        real_X, synth_X = real, synth

    real_m, synth_m = build_feature_matrix(real_X, synth_X)

    mia = _mia_auc(real_m, synth_m)
    dcr_raw = _dcr_score(real_m, synth_m)
    nndr_raw = _nndr_score(real_m, synth_m)

    # Normalize distance metrics using median pairwise scale from real data
    if len(real_m) >= 2:
        sample = real_m[np.random.default_rng(0).choice(len(real_m), size=min(200, len(real_m)), replace=False)]
        ref = float(np.median(_pairwise_nn_distances(sample, real_m, k=2)))
    else:
        ref = 1.0
    dcr_n = _normalize_distance(dcr_raw, ref)
    nndr_n = clip01(nndr_raw)  # already a ratio in ~[0,1]

    # Convert each metric to a RISK in [0,1] (higher worse)
    risks = {
        "mia_auc": clip01(mia),
        "nndr": clip01(1.0 - nndr_n),
        "dcr": clip01(1.0 - dcr_n),
    }

    # Weighted average of risks
    num = 0.0
    den = 0.0
    for name, spec in metric_cfg.items():
        w = float(spec.get("weight", 0.0))
        if name not in risks or w <= 0:
            continue
        r = risks[name]
        # If YAML marks invert:true on a safety metric we already converted to risk;
        # mia_auc invert:true in old YAML meant "lower better" → treat as risk directly.
        num += w * r
        den += w
    privacy_risk = clip01(num / den) if den > 0 else 0.5

    return {
        "privacy_risk": privacy_risk,
        "mia_auc": clip01(mia),
        "nndr": float(nndr_raw),
        "dcr": float(dcr_raw),
        "nndr_norm": nndr_n,
        "dcr_norm": dcr_n,
    }
