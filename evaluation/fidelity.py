"""Fidelity evaluation — higher score is better, normalized to ``[0, 1]``.

Aggregation is config-driven via ``cfg["fidelity"]``:

- ``sdv_only`` (default): ``F = SDV QualityReport``; KS / correlation are
  diagnostics only and never enter the objective. When ``require_sdv: true``
  (default under ``sdv_only``) and SDV is unavailable, ``fidelity`` is NaN and
  ``sdv_unavailable`` is set — callers must fail/prune the trial (no silent fallback).
- ``legacy_composite``: historical FORGE formula
  ``0.5·SDV + 0.25·KS + 0.25·Corr`` (or ``0.5·KS + 0.5·Corr`` if SDV missing).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .preprocess import align_columns, clip01, numeric_feature_frame

AGG_LEGACY = "legacy_composite"
AGG_SDV_ONLY = "sdv_only"


class FidelityUnavailableError(RuntimeError):
    """Raised when SDV QualityReport is required but unavailable."""


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
    """SDV / sdmetrics QualityReport overall score in [0, 1], or None if unavailable.

    SDV 1.3x+ changed metadata / evaluation APIs. Prefer the stable
    ``sdmetrics`` single-table QualityReport with ``SingleTableMetadata.to_dict()``.
    """
    # Path A (preferred): sdmetrics single-table QualityReport + SingleTableMetadata dict
    try:
        from sdv.metadata import SingleTableMetadata
        from sdmetrics.reports.single_table import QualityReport as STQualityReport

        meta = SingleTableMetadata()
        meta.detect_from_dataframe(real)
        report = STQualityReport()
        report.generate(real, synth, meta.to_dict(), verbose=False)
        return clip01(float(report.get_score()))
    except Exception:
        pass

    # Path B: newer Metadata wrapper + evaluate_quality
    try:
        from sdv.evaluation.single_table import evaluate_quality
        from sdv.metadata import Metadata, SingleTableMetadata

        stm = SingleTableMetadata()
        stm.detect_from_dataframe(real)
        md = Metadata.load_from_dict({"tables": {"table": stm.to_dict()}})
        report = evaluate_quality(real, synth, md)
        if hasattr(report, "get_score"):
            return clip01(float(report.get_score()))
        return clip01(float(report))
    except Exception:
        pass

    # Path C: legacy evaluate_quality(SingleTableMetadata)
    try:
        from sdv.evaluation.single_table import evaluate_quality
        from sdv.metadata import SingleTableMetadata

        meta = SingleTableMetadata()
        meta.detect_from_dataframe(real)
        report = evaluate_quality(real, synth, meta)
        if hasattr(report, "get_score"):
            return clip01(float(report.get_score()))
        return clip01(float(report))
    except Exception:
        return None


def _fidelity_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = {}
    if isinstance(cfg, Mapping):
        raw = dict(cfg.get("fidelity") or {})
    aggregation = str(raw.get("aggregation") or AGG_SDV_ONLY).strip().lower()
    if aggregation not in {AGG_LEGACY, AGG_SDV_ONLY}:
        raise ValueError(
            f"Unknown fidelity.aggregation={aggregation!r}; "
            f"expected {AGG_LEGACY!r} or {AGG_SDV_ONLY!r}"
        )
    # sdv_only implies require_sdv unless explicitly overridden to false
    if "require_sdv" in raw:
        require_sdv = bool(raw.get("require_sdv"))
    else:
        require_sdv = aggregation == AGG_SDV_ONLY
    return {
        "aggregation": aggregation,
        "require_sdv": require_sdv,
    }


def sdv_required_but_missing(
    fidelity: Mapping[str, Any],
    cfg: Mapping[str, Any] | None = None,
) -> bool:
    """True when config requires SDV and evaluation marked it unavailable."""
    fcfg = _fidelity_cfg(cfg)
    if not fcfg["require_sdv"] and fcfg["aggregation"] != AGG_SDV_ONLY:
        return False
    if fcfg["require_sdv"] or fcfg["aggregation"] == AGG_SDV_ONLY:
        return bool(fidelity.get("sdv_unavailable"))
    return False


def evaluate_fidelity(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    metadata: Any = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return fidelity metrics; primary ``fidelity`` ∈ [0, 1] when available.

    Under ``sdv_only`` + missing SDV, ``fidelity`` is NaN and
    ``sdv_unavailable=True`` (no KS/corr fallback into the objective score).
    """
    _ = metadata
    fcfg = _fidelity_cfg(cfg)
    aggregation = fcfg["aggregation"]
    require_sdv = fcfg["require_sdv"]

    real, synth = align_columns(real_train, synthetic)

    ks = _ks_similarity(real, synth)
    corr = _corr_similarity(real, synth)
    sdv = _sdv_quality(real, synth)
    sdv_unavailable = sdv is None

    if aggregation == AGG_SDV_ONLY:
        if sdv is None:
            fidelity = float("nan")
        else:
            fidelity = clip01(sdv)
    else:
        # legacy_composite — historical behaviour
        if sdv is not None:
            fidelity = clip01(0.5 * sdv + 0.25 * ks + 0.25 * corr)
        else:
            fidelity = clip01(0.5 * ks + 0.5 * corr)

    out: dict[str, Any] = {
        "fidelity": fidelity,
        "ks_similarity": clip01(ks),
        "corr_similarity": clip01(corr),
        "sdv_quality": clip01(sdv) if sdv is not None else float("nan"),
        "fidelity_aggregation": aggregation,
        "sdv_unavailable": bool(sdv_unavailable),
        "require_sdv": bool(require_sdv),
    }
    return out
