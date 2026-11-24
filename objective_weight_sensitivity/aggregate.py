"""Aggregate sensitivity studies into comparison CSVs / metadata JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from optuna.trial import TrialState

from objective_weight_sensitivity.runner import study_dir, study_name


def _iter_summary_paths(results_root: Path) -> list[Path]:
    root = Path(results_root)
    if not root.is_dir():
        return []
    return sorted(root.glob("*/*/*/seed*/summary.json"))


def load_summaries(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in _iter_summary_paths(results_root):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        d = dict(d)
        d["source_summary"] = str(path)
        rows.append(d)
    return pd.DataFrame(rows)


def _trial_rows_from_db(
    *,
    results_root: Path,
    dataset: str,
    generator: str,
    config: str,
    seed: int,
) -> list[dict[str, Any]]:
    out = study_dir(results_root, dataset, generator, config, seed)
    db = out / "optuna.db"
    if not db.is_file():
        return []
    name = study_name(dataset, generator, config, seed)
    storage = f"sqlite:///{db.resolve()}"
    try:
        study = optuna.load_study(study_name=name, storage=storage)
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        attrs = dict(trial.user_attrs or {})
        rows.append(
            {
                "dataset": dataset,
                "generator": generator,
                "config": str(config).upper(),
                "seed": int(seed),
                "trial_number": trial.number,
                "state": trial.state.name,
                "objective": trial.value if trial.state == TrialState.COMPLETE else None,
                "fidelity": attrs.get("fidelity"),
                "sdv_quality": attrs.get("sdv_quality"),
                "ks_similarity": attrs.get("ks_similarity"),
                "corr_similarity": attrs.get("corr_similarity"),
                "utility": attrs.get("utility"),
                "privacy_risk": attrs.get("privacy_risk"),
                "mia_auc": attrs.get("mia_auc"),
                "nndr": attrs.get("nndr"),
                "dcr": attrs.get("dcr"),
                "duration_s": attrs.get("duration_s"),
                "error": attrs.get("error"),
                "params_json": json.dumps(dict(trial.params), sort_keys=True, default=str),
            }
        )
    return rows


def build_all_trials(results_root: Path, summaries: pd.DataFrame | None = None) -> pd.DataFrame:
    summaries = summaries if summaries is not None else load_summaries(results_root)
    rows: list[dict[str, Any]] = []
    if summaries.empty:
        return pd.DataFrame(rows)
    keys = summaries[["dataset", "generator", "config", "seed"]].drop_duplicates()
    for _, r in keys.iterrows():
        rows.extend(
            _trial_rows_from_db(
                results_root=results_root,
                dataset=str(r["dataset"]),
                generator=str(r["generator"]),
                config=str(r["config"]),
                seed=int(r["seed"]),
            )
        )
    return pd.DataFrame(rows)


def build_best_trials(summaries: pd.DataFrame) -> pd.DataFrame:
    if summaries.empty:
        return summaries.copy()
    cols = [
        "dataset",
        "generator",
        "config",
        "seed",
        "status",
        "best_objective",
        "best_fidelity",
        "best_sdv_quality",
        "best_utility",
        "best_privacy_risk",
        "best_mia_auc",
        "best_nndr",
        "best_dcr",
        "best_ks_similarity",
        "best_corr_similarity",
        "best_params",
        "n_complete",
        "n_fail",
        "n_pruned",
        "n_attempted",
        "study_wall_clock_s",
        "weights",
        "fidelity_aggregation",
        "sampler",
    ]
    keep = [c for c in cols if c in summaries.columns]
    out = summaries[keep].copy()
    if "best_params" in out.columns:
        out["best_params"] = out["best_params"].apply(
            lambda x: json.dumps(x, sort_keys=True, default=str) if isinstance(x, dict) else x
        )
    if "weights" in out.columns:
        out["weights"] = out["weights"].apply(
            lambda x: json.dumps(x, sort_keys=True, default=str) if isinstance(x, dict) else x
        )
    return out


def build_objective_weight_comparison(summaries: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, generator, config) focusing on underlying F/U/P (not raw J alone)."""
    if summaries.empty:
        return summaries.copy()
    df = summaries.copy()
    # Expand weights if present as dict
    for key in ("fidelity", "utility", "privacy_risk"):
        col = f"w_{key}"
        df[col] = df["weights"].apply(
            lambda w, k=key: (w or {}).get(k) if isinstance(w, dict) else None
        )
    cols = [
        "dataset",
        "generator",
        "config",
        "seed",
        "w_fidelity",
        "w_utility",
        "w_privacy_risk",
        "best_objective",
        "best_fidelity",
        "best_sdv_quality",
        "best_utility",
        "best_privacy_risk",
        "best_mia_auc",
        "best_nndr",
        "best_dcr",
        "n_complete",
        "n_fail",
        "n_pruned",
        "status",
        "fidelity_aggregation",
        "sampler",
    ]
    keep = [c for c in cols if c in df.columns]
    return df[keep].sort_values(["dataset", "generator", "config"]).reset_index(drop=True)


def build_metadata(results_root: Path, summaries: pd.DataFrame) -> dict[str, Any]:
    configs = sorted({str(c) for c in summaries["config"].dropna().unique()}) if not summaries.empty else []
    return {
        "results_root": str(results_root),
        "n_studies": int(len(summaries)),
        "configs": configs,
        "sampler": "standard_tpe",
        "fidelity": {
            "aggregation": "sdv_only",
            "note": (
                "F = SDV QualityReport; KS and correlation stored as diagnostics only; "
                "no silent fallback when SDV unavailable"
            ),
        },
        "utility": "unchanged_trtr_tstr_gap",
        "privacy": "unchanged_0.5_mia_0.3_nndr_0.2_dcr",
        "objective": "J = wF*F + wU*U - wP*P (maximize); do not compare raw J across weight configs",
        "data_split": "fixed_yaml_80_20_no_val",
        "historical_multiseed_untouched": True,
    }


def rebuild_all(results_root: Path) -> dict[str, Path]:
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(root)
    all_trials = build_all_trials(root, summaries)
    best = build_best_trials(summaries)
    comparison = build_objective_weight_comparison(summaries)
    meta = build_metadata(root, summaries)

    paths = {
        "all_trials": root / "all_trials.csv",
        "best_trials": root / "best_trials.csv",
        "objective_weight_comparison": root / "objective_weight_comparison.csv",
        "objective_metadata": root / "objective_metadata.json",
    }
    all_trials.to_csv(paths["all_trials"], index=False)
    best.to_csv(paths["best_trials"], index=False)
    comparison.to_csv(paths["objective_weight_comparison"], index=False)
    paths["objective_metadata"].write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    return paths
