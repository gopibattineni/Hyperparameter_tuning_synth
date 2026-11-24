#!/usr/bin/env python3
"""Import legacy seed-42 HPO runs (Standard or Multivariate TPE) into a multiseed tree.

Maps ``{package}/results/{classification,regression}/N_dataset/generator/``
(trials.csv + completed/timing) into::

    results/multiseed_YYYYMMDD/
      raw/{dataset}__{generator}__{sampler}__seed42.csv
      studies/{dataset}/{generator}/{sampler}/seed42/summary.json

Does not overwrite existing seed-42 study dirs unless ``--force``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multiseed_comparison.aggregate import rebuild_all
from multiseed_comparison.analyze_samplers import run_analysis
from multiseed_comparison.runner import study_dir, study_name
from multivariate_tpe.datasets_catalog import ALL_DATASETS, GENERATORS_NO_FOREST

SEED = 42

_PACKAGE_FOR_SAMPLER = {
    "standard_tpe": "standard_tpe",
    "multivariate_tpe": "multivariate_tpe",
    "gp": "gaussian_process",
}


def _legacy_roots(package: str) -> list[Path]:
    base = _ROOT / package / "results"
    return [base / "classification", base / "regression"]


def _dataset_folder_map(package: str) -> dict[str, Path]:
    """Map dataset key → numbered result folder (e.g. cancer → 1_cancer)."""
    mapping: dict[str, Path] = {}
    for root in _legacy_roots(package):
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if not d.is_dir() or "_" not in d.name:
                continue
            name = d.name.split("_", 1)[1]
            if name in ALL_DATASETS:
                mapping[name] = d
    return mapping


def _duration_to_seconds(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    m = re.match(
        r"(?:(\d+)\s+days?\s+)?(\d+):(\d+):(\d+(?:\.\d+)?)",
        s,
    )
    if m:
        days = int(m.group(1) or 0)
        h, mi = int(m.group(2)), int(m.group(3))
        sec = float(m.group(4))
        return days * 86400 + h * 3600 + mi * 60 + sec
    try:
        return float(s)
    except ValueError:
        return None


def _legacy_to_raw_rows(
    trials: pd.DataFrame, dataset: str, generator: str, sampler: str
) -> pd.DataFrame:
    """Convert Optuna trials.csv to multiseed raw columns."""
    rows: list[dict[str, Any]] = []
    for _, tr in trials.iterrows():
        row: dict[str, Any] = {
            "dataset": dataset,
            "generator": generator,
            "sampler": sampler,
            "seed": SEED,
            "trial_number": int(tr["number"]) if "number" in tr else None,
            "state": str(tr.get("state", "")),
            "objective": tr.get("value"),
            "fidelity": tr.get("user_attrs_fidelity"),
            "utility": tr.get("user_attrs_utility"),
            "privacy_risk": tr.get("user_attrs_privacy_risk"),
            "duration_s": _duration_to_seconds(tr.get("duration")),
            "trial_seed": None,
            "error": None,
        }
        for col in trials.columns:
            if col.startswith("params_"):
                row[f"param_{col[len('params_'):]}"] = tr[col]
        rows.append(row)
    return pd.DataFrame(rows)


def _build_summary(
    *,
    dataset: str,
    generator: str,
    sampler: str,
    trials: pd.DataFrame,
    src: Path,
) -> dict[str, Any]:
    complete = (
        trials[trials["state"] == "COMPLETE"] if "state" in trials.columns else trials
    )
    n_complete = int(len(complete))
    n_fail = int((trials["state"] == "FAIL").sum()) if "state" in trials.columns else 0
    n_pruned = (
        int((trials["state"] == "PRUNED").sum()) if "state" in trials.columns else 0
    )

    best_objective = best_number = best_f = best_u = best_p = None
    if n_complete and "value" in complete.columns:
        idx = complete["value"].astype(float).idxmax()
        best = complete.loc[idx]
        best_objective = float(best["value"])
        best_number = int(best["number"])
        best_f = best.get("user_attrs_fidelity")
        best_u = best.get("user_attrs_utility")
        best_p = best.get("user_attrs_privacy_risk")

    timing: dict[str, Any] = {}
    timing_path = src / "timing.json"
    if timing_path.is_file():
        timing = json.loads(timing_path.read_text(encoding="utf-8"))
    completed: dict[str, Any] = {}
    completed_path = src / "completed.json"
    if completed_path.is_file():
        completed = json.loads(completed_path.read_text(encoding="utf-8"))

    durations = [
        d
        for d in (_duration_to_seconds(v) for v in trials.get("duration", []))
        if d is not None
    ]
    wall = completed.get("elapsed_sec") or timing.get("elapsed_sec")
    return {
        "dataset": dataset,
        "generator": generator,
        "sampler": sampler,
        "seed": SEED,
        "study_name": study_name(dataset, generator, sampler, SEED),
        "status": "completed"
        if n_complete >= 20
        else ("partial" if n_complete else "failed"),
        "n_trials_target": 20,
        "n_complete": n_complete,
        "n_fail": n_fail,
        "n_pruned": n_pruned,
        "n_attempted": int(len(trials)),
        "best_objective": best_objective,
        "best_trial_number": best_number,
        "best_fidelity": None if best_f is None or pd.isna(best_f) else float(best_f),
        "best_utility": None if best_u is None or pd.isna(best_u) else float(best_u),
        "best_privacy_risk": None
        if best_p is None or pd.isna(best_p)
        else float(best_p),
        "study_wall_clock_s": float(wall) if wall is not None else None,
        "mean_trial_duration_s": float(sum(durations) / len(durations))
        if durations
        else None,
        "sum_trial_duration_s": float(sum(durations)) if durations else None,
        "imported_from": str(src),
        "import_note": (
            f"Legacy {sampler} (config seed=42). "
            "Sampler/generator seeding matches optuna.yaml; "
            "per-trial reseeding attrs were not recorded in the original run."
        ),
    }


def import_seed42(
    results_root: Path,
    *,
    sampler: str = "standard_tpe",
    force: bool = False,
) -> dict[str, int]:
    """Copy legacy seed-42 studies for ``sampler`` into ``results_root``."""
    if sampler not in _PACKAGE_FOR_SAMPLER:
        raise ValueError(
            f"Unsupported sampler {sampler!r}; expected one of "
            f"{sorted(_PACKAGE_FOR_SAMPLER)}"
        )
    package = _PACKAGE_FOR_SAMPLER[sampler]
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "raw").mkdir(parents=True, exist_ok=True)

    folders = _dataset_folder_map(package)
    stats = {"imported": 0, "skipped": 0, "missing": 0}
    missing: list[str] = []

    for dataset in ALL_DATASETS:
        base = folders.get(dataset)
        for generator in GENERATORS_NO_FOREST:
            if base is None:
                stats["missing"] += 1
                missing.append(f"{dataset}/{generator}")
                continue
            src = base / generator
            trials_path = src / "trials.csv"
            out = study_dir(results_root, dataset, generator, sampler, SEED)
            summary_path = out / "summary.json"
            if summary_path.exists() and not force:
                stats["skipped"] += 1
                continue
            if not trials_path.is_file():
                stats["missing"] += 1
                missing.append(f"{dataset}/{generator}")
                continue

            trials = pd.read_csv(trials_path)
            raw = _legacy_to_raw_rows(trials, dataset, generator, sampler)
            raw_path = (
                results_root
                / "raw"
                / f"{study_name(dataset, generator, sampler, SEED)}.csv"
            )
            raw.to_csv(raw_path, index=False)

            summary = _build_summary(
                dataset=dataset,
                generator=generator,
                sampler=sampler,
                trials=trials,
                src=src,
            )
            out.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(summary, indent=2, default=str)
            summary_path.write_text(payload, encoding="utf-8")
            (out / "meta.json").write_text(payload, encoding="utf-8")
            (out / "IMPORT_SOURCE.txt").write_text(
                f"{src}\n{trials_path}\n", encoding="utf-8"
            )
            stats["imported"] += 1

    if missing:
        (results_root / f"import_seed42_{sampler}_missing.txt").write_text(
            "\n".join(missing) + "\n", encoding="utf-8"
        )
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Import legacy seed-42 Standard/Multivariate TPE into a multiseed tree"
    )
    p.add_argument(
        "--results-root",
        type=str,
        default=str(_ROOT / "results" / "multiseed_20261027"),
    )
    p.add_argument(
        "--sampler",
        choices=sorted(_PACKAGE_FOR_SAMPLER),
        default="standard_tpe",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing seed-42 summaries/raw CSVs",
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Rebuild summary tables + plots after import",
    )
    args = p.parse_args(argv)

    stats = import_seed42(
        Path(args.results_root), sampler=args.sampler, force=args.force
    )
    print(
        f"Import {args.sampler} seed=42 → {args.results_root}: "
        f"imported={stats['imported']} skipped={stats['skipped']} missing={stats['missing']}"
    )
    if args.analyze:
        rebuild_all(Path(args.results_root))
        run_analysis(Path(args.results_root))
    return 0 if stats["missing"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
