"""Unit tests for objective-weight sensitivity (no GPU / no full Optuna grid)."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evaluation.fidelity import AGG_LEGACY, AGG_SDV_ONLY, evaluate_fidelity, sdv_required_but_missing
from evaluation.objective import compute_objective
from evaluation.objective_presets import (
    PRESET_NAMES,
    list_presets,
    resolve_preset,
    validate_weights,
)


def test_all_presets_sum_to_one():
    for name in PRESET_NAMES:
        cfg = resolve_preset(name)
        w = validate_weights(cfg["weights"])
        assert abs(sum(w.values()) - 1.0) <= 1e-6
        assert cfg["direction"] == "maximize"
        assert cfg["fidelity"]["aggregation"] == "sdv_only"
        assert cfg["fidelity"]["require_sdv"] is True


def test_list_presets_contains_o1_o5():
    names = list_presets()
    for n in PRESET_NAMES:
        assert n in names


def test_validate_weights_rejects_bad_sum():
    with pytest.raises(ValueError, match="sum"):
        validate_weights({"fidelity": 0.5, "utility": 0.5, "privacy_risk": 0.5})


def _toy_frames(n: int = 40, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    real = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "y": rng.normal(size=n),
            "z": rng.integers(0, 3, size=n),
        }
    )
    synth = real.copy()
    synth["x"] = synth["x"] + rng.normal(scale=0.1, size=n)
    return real, synth


def test_legacy_composite_matches_old_formula_when_sdv_present():
    real, synth = _toy_frames()
    with patch("evaluation.fidelity._sdv_quality", return_value=0.8), patch(
        "evaluation.fidelity._ks_similarity", return_value=0.6
    ), patch("evaluation.fidelity._corr_similarity", return_value=0.4):
        out = evaluate_fidelity(
            real, synth, cfg={"fidelity": {"aggregation": AGG_LEGACY}}
        )
    expected = 0.5 * 0.8 + 0.25 * 0.6 + 0.25 * 0.4
    assert out["fidelity"] == pytest.approx(expected)
    assert out["sdv_unavailable"] is False
    assert out["fidelity_aggregation"] == AGG_LEGACY


def test_legacy_fallback_when_sdv_missing():
    real, synth = _toy_frames()
    with patch("evaluation.fidelity._sdv_quality", return_value=None), patch(
        "evaluation.fidelity._ks_similarity", return_value=0.6
    ), patch("evaluation.fidelity._corr_similarity", return_value=0.4):
        out = evaluate_fidelity(
            real, synth, cfg={"fidelity": {"aggregation": AGG_LEGACY}}
        )
    expected = 0.5 * 0.6 + 0.5 * 0.4
    assert out["fidelity"] == pytest.approx(expected)
    assert out["sdv_unavailable"] is True


def test_sdv_only_uses_sdv_and_keeps_diagnostics():
    real, synth = _toy_frames()
    with patch("evaluation.fidelity._sdv_quality", return_value=0.77), patch(
        "evaluation.fidelity._ks_similarity", return_value=0.55
    ), patch("evaluation.fidelity._corr_similarity", return_value=0.66):
        out = evaluate_fidelity(
            real,
            synth,
            cfg={"fidelity": {"aggregation": AGG_SDV_ONLY, "require_sdv": True}},
        )
    assert out["fidelity"] == pytest.approx(0.77)
    assert out["sdv_quality"] == pytest.approx(0.77)
    assert out["ks_similarity"] == pytest.approx(0.55)
    assert out["corr_similarity"] == pytest.approx(0.66)
    assert out["sdv_unavailable"] is False


def test_sdv_only_no_silent_fallback_when_sdv_missing():
    real, synth = _toy_frames()
    with patch("evaluation.fidelity._sdv_quality", return_value=None), patch(
        "evaluation.fidelity._ks_similarity", return_value=0.9
    ), patch("evaluation.fidelity._corr_similarity", return_value=0.9):
        out = evaluate_fidelity(
            real,
            synth,
            cfg={"fidelity": {"aggregation": AGG_SDV_ONLY, "require_sdv": True}},
        )
    assert math.isnan(out["fidelity"])
    assert out["sdv_unavailable"] is True
    assert sdv_required_but_missing(out, {"fidelity": {"aggregation": "sdv_only", "require_sdv": True}})
    with pytest.raises(ValueError, match="unavailable"):
        compute_objective(
            out,
            {"privacy_risk": 0.2},
            {"utility": 0.5},
            resolve_preset("O1"),
        )


def test_compute_objective_uses_preset_weights():
    cfg = resolve_preset("O2")
    out = compute_objective(
        {"fidelity": 1.0},
        {"privacy_risk": 0.0},
        {"utility": 1.0},
        cfg,
    )
    # O2: 0.2*1 + 0.6*1 - 0.2*0 = 0.8
    assert out["objective"] == pytest.approx(0.8)
    assert out["w_utility"] == pytest.approx(0.6)
