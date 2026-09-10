"""Persist Optuna run artifacts to a run directory."""

from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

try:
    import optuna
    from optuna.visualization import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
    )
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore


COMPLETED_MARKER = "completed.json"


def _dataset_config() -> dict:
    try:
        from datasets.loader import load_dataset_config

        return load_dataset_config() or {}
    except Exception:
        return {}


def dataset_index(dataset: str) -> int:
    """1-based dataset number from ``config/datasets.yaml`` order."""
    keys = list(_dataset_config().keys())
    if dataset in keys:
        return keys.index(dataset) + 1
    return 0


def dataset_folder(dataset: str) -> str:
    """Folder name ``{n}_{dataset}`` e.g. ``1_cancer``, ``15_real_estate``."""
    n = dataset_index(dataset)
    return f"{n}_{dataset}" if n else dataset


def resolve_task(dataset: str, task: str | None = None) -> str:
    """Map a dataset to ``classification`` or ``regression`` result folder."""
    if task:
        return "regression" if str(task).lower() == "regression" else "classification"
    cfg = _dataset_config().get(dataset) or {}
    return "regression" if str(cfg.get("task", "")).lower() == "regression" else "classification"


def run_dir(
    results_root: Path,
    dataset: str,
    generator: str,
    task: str | None = None,
) -> Path:
    """Return ``results/{classification|regression}/{n}_{dataset}/{generator}/``."""
    category = resolve_task(dataset, task)
    return Path(results_root) / category / dataset_folder(dataset) / generator


def is_completed(
    results_root: Path,
    dataset: str,
    generator: str,
    task: str | None = None,
) -> bool:
    """Resume helper — True if a successful completed marker exists."""
    marker = run_dir(results_root, dataset, generator, task=task) / COMPLETED_MARKER
    return marker.is_file()


def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def save_run_artifacts(
    study: "optuna.Study",
    out_dir: Path,
    *,
    best_params: Mapping[str, Any],
    metrics: Mapping[str, Any],
    synthetic: pd.DataFrame | None = None,
    timing: Mapping[str, Any] | None = None,
    config_snapshot: Mapping[str, Path] | None = None,
    save_plots: list[str] | None = None,
    save_study_pickle: bool = True,
) -> Path:
    """Write best params, trials, metrics, synthetic data, plots, marker."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_json(out_dir / "best_params.json", dict(best_params))
    save_json(out_dir / "metrics.json", dict(metrics))
    if timing:
        save_json(out_dir / "timing.json", dict(timing))

    # Trial history
    try:
        trials_df = study.trials_dataframe()
        trials_df.to_csv(out_dir / "trials.csv", index=False)
    except Exception:
        pd.DataFrame([{"number": t.number, "value": t.value, "state": str(t.state)}
                      for t in study.trials]).to_csv(out_dir / "trials.csv", index=False)

    if save_study_pickle:
        with (out_dir / "study.pkl").open("wb") as fh:
            pickle.dump(study, fh)

    if synthetic is not None:
        synthetic.to_csv(out_dir / "best_synthetic.csv", index=False)

    # Optional Optuna plots
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    save_plots = save_plots or []
    if optuna is not None and save_plots:
        plotters = {
            "optimization_history": plot_optimization_history,
            "param_importances": plot_param_importances,
            "parallel_coordinate": plot_parallel_coordinate,
        }
        for name in save_plots:
            fn = plotters.get(name)
            if fn is None:
                continue
            try:
                fig = fn(study)
                fig.write_html(str(plot_dir / f"{name}.html"))
                try:
                    fig.write_image(str(plot_dir / f"{name}.png"))
                except Exception:
                    pass
            except Exception:
                continue

    if config_snapshot:
        snap = out_dir / "config_snapshot"
        snap.mkdir(exist_ok=True)
        for label, src in config_snapshot.items():
            src = Path(src)
            if src.is_file():
                shutil.copy2(src, snap / f"{label}{src.suffix}")

    return out_dir


def mark_completed(
    out_dir: Path,
    *,
    dataset: str,
    generator: str,
    best_value: float | None,
    elapsed_sec: float,
) -> None:
    save_json(
        Path(out_dir) / COMPLETED_MARKER,
        {
            "dataset": dataset,
            "generator": generator,
            "best_value": best_value,
            "elapsed_sec": elapsed_sec,
            "status": "completed",
        },
    )


def mark_failed(out_dir: Path, *, dataset: str, generator: str, error: str, elapsed_sec: float) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    save_json(
        Path(out_dir) / "failed.json",
        {
            "dataset": dataset,
            "generator": generator,
            "error": error,
            "elapsed_sec": elapsed_sec,
            "status": "failed",
        },
    )
