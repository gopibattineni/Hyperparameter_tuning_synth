"""Fidelity evaluation — higher score is better, normalized to ``[0, 1]``.

Uses reusable statistical checks aligned with the benchmark (KS / correlation).
Falls back gracefully when optional SDV QualityReport is unavailable.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .preprocess import align_columns, clip01, numeric_feature_frame


def _ks_similarity(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean (1 − KS statistic) over numeric columns."""
    real_n = numeric_feature_frame(real)
    synth_n = numeric_feature_frame(synth)
    cols = [c for c in real_n.columns if c in synth_n.columns]
    if not cols:
        return 0.5
    sims = []
    for c in cols:
        r = real_n[c].dropna().to_numpy()
        s = synth_n[c].dropna().to_numpy()
        if len(r) < 2 or len(s) < 2:
            continue
        stat = ks_2samp(r, s).statistic
        sims.append(1.0 - float(stat))
    return float(np.mean(sims)) if sims else 0.5


def _corr_similarity(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Correlation-matrix fidelity: 1 − mean |Δρ| over numeric pairs."""
    real_n = numeric_feature_frame(real)
    synth_n = numeric_feature_frame(synth)
    cols = [c for c in real_n.columns if c in synth_n.columns]
    if len(cols) < 2:
        return 0.5
    rc = real_n[cols].corr().to_numpy()
    sc = synth_n[cols].corr().to_numpy()
    # Ignore diagonal
    mask = ~np.eye(len(cols), dtype=bool)
    diff = np.abs(rc[mask] - sc[mask])
    diff = diff[~np.isnan(diff)]
    if diff.size == 0:
        return 0.5
    return clip01(1.0 - float(np.mean(diff)))


def _sdv_quality(real: pd.DataFrame, synth: pd.DataFrame) -> float | None:
    try:
        from sdv.evaluation.single_table import evaluate_quality
        from sdv.metadata import SingleTableMetadata
    except ImportError:
        return None
    try:
        meta = SingleTableMetadata()
        meta.detect_from_dataframe(real)
        report = evaluate_quality(real, synth, meta)
        # SDV API varies: QualityReport object or float
        if hasattr(report, "get_score"):
            return clip01(float(report.get_score()))
        return clip01(float(report))
    except Exception:
        return None


def evaluate_fidelity(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: Any = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return fidelity metrics; primary ``fidelity`` ∈ [0, 1] (higher better)."""
    _ = metadata, cfg
    real, synth = align_columns(real_train, synthetic)

    ks = _ks_similarity(real, synth)
    corr = _corr_similarity(real, synth)
    sdv = _sdv_quality(real, synth)

    if sdv is not None:
        fidelity = clip01(0.5 * sdv + 0.25 * ks + 0.25 * corr)
    else:
        fidelity = clip01(0.5 * ks + 0.5 * corr)

    return {
        "fidelity": fidelity,
        "ks_similarity": clip01(ks),
        "corr_similarity": clip01(corr),
        "sdv_quality": clip01(sdv) if sdv is not None else float("nan"),
    }
