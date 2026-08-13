#!/usr/bin/env python3
"""Experiment runner: 15 datasets × 8 generators Optuna HPO sweep.

For every dataset–generator pair:
  1. Run Optuna optimization (skips if resume + already completed)
  2. Save best parameters
  3. Generate tuned synthetic data
  4. Evaluate fidelity / privacy / utility
  5. Append a row to the master CSV
  6. Log wall-clock time

Progress is shown with tqdm.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Package import path — this folder only
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tqdm import tqdm

from config_utils import PACKAGE_ROOT, load_config
from datasets.loader import load_dataset_config
from generators import list_generators
from generators.registry import get_generator_class
from tuning.artifacts import is_completed
from tuning.optimizer import run_study

# Canonical 8 research generators (bootstrap_noise excluded from full sweeps)
RESEARCH_GENERATORS = [
    "gaussian_copula",
    "copulagan",
    "ctgan",
    "ctabgan",
    "tvae",
    "wgan_gp",
    "forest_diffusion",
    "tabddpm",
]

MASTER_CSV = "all_experiments.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run 15×8 Optuna HPO experiments")
    p.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset keys")
    p.add_argument("--generators", nargs="*", default=None, help="Subset of generator keys")
    p.add_argument("--n-trials", type=int, default=None, help="Override optuna.yaml n_trials")
    p.add_argument("--results-dir", type=str, default=None, help="Results root directory")
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip pairs that already have completed.json (default: on)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run even if completed.json exists",
    )
    p.add_argument(
        "--skip-unavailable",
        action="store_true",
        default=True,
        help="Skip generators whose dependencies are missing (default: on)",
    )
    p.add_argument(
        "--include-unavailable",
        action="store_true",
        help="Attempt unavailable generators (will be marked failed)",
    )
    p.add_argument("--show-trial-progress", action="store_true")
    return p.parse_args(argv)


MASTER_COLUMNS = [
    "dataset",
    "generator",
    "status",
    "best_value",
    "elapsed_sec",
    "n_trials",
    "objective",
    "fidelity",
    "utility",
    "privacy_risk",
    "ks_similarity",
    "corr_similarity",
    "mia_auc",
    "nndr",
    "dcr",
    "mean_f1_gap",
    "mean_tstr_f1",
    "mean_trtr_f1",
    "run_dir",
    "error",
]


def _normalize_row(row: dict) -> dict:
    return {c: row.get(c) for c in MASTER_COLUMNS}


def _append_master(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([_normalize_row(row)], columns=MASTER_COLUMNS)
    if path.is_file():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    resume = not args.no_resume
    skip_unavailable = not args.include_unavailable

    results_root = Path(args.results_dir) if args.results_dir else PACKAGE_ROOT / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    master_path = results_root / MASTER_CSV

    all_datasets = list(load_dataset_config().keys())
    datasets = args.datasets or all_datasets
    generators = args.generators or RESEARCH_GENERATORS

    # Validate keys
    unknown_ds = sorted(set(datasets) - set(all_datasets))
    if unknown_ds:
        raise SystemExit(f"Unknown datasets: {unknown_ds}")

    registered = set(list_generators())
    pairs: list[tuple[str, str]] = []
    skipped_unavailable: list[str] = []

    for ds in datasets:
        for gen in generators:
            if gen not in registered:
                skipped_unavailable.append(f"{ds}/{gen} (not registered)")
                continue
            cls = get_generator_class(gen)
            available = getattr(cls, "is_available", lambda: True)()
            if skip_unavailable and not available:
                skipped_unavailable.append(f"{ds}/{gen} (deps missing)")
                continue
            pairs.append((ds, gen))

    print(f"Results dir : {results_root}")
    print(f"Datasets    : {len(datasets)}")
    print(f"Generators  : {len(generators)}")
    print(f"Pairs       : {len(pairs)}  (resume={resume})")
    if skipped_unavailable:
        print(f"Skipped up-front: {len(skipped_unavailable)}")
        for s in skipped_unavailable[:12]:
            print(f"  - {s}")
        if len(skipped_unavailable) > 12:
            print(f"  … +{len(skipped_unavailable) - 12} more")

    optuna_cfg = load_config("optuna")
    if args.n_trials is not None:
        optuna_cfg["n_trials"] = args.n_trials

    summary_rows: list[dict] = []
    t_all = time.perf_counter()

    pbar = tqdm(pairs, desc="experiments", unit="pair")
    for dataset_key, generator_name in pbar:
        pbar.set_postfix(ds=dataset_key, gen=generator_name)

        if resume and is_completed(results_root, dataset_key, generator_name):
            row = {
                "dataset": dataset_key,
                "generator": generator_name,
                "status": "skipped_resume",
                "best_value": None,
                "elapsed_sec": 0.0,
                "n_trials": None,
                "objective": None,
                "fidelity": None,
                "utility": None,
                "privacy_risk": None,
                "run_dir": str(results_root / dataset_key / generator_name),
                "error": None,
            }
            # Try to enrich from existing metrics.json
            metrics_path = results_root / dataset_key / generator_name / "metrics.json"
            if metrics_path.is_file():
                import json
                m = json.loads(metrics_path.read_text())
                for k in ("objective", "fidelity", "utility", "privacy_risk", "mean_f1_gap"):
                    if k in m:
                        row[k] = m[k]
            summary_rows.append(row)
            _append_master(master_path, row)
            tqdm.write(f"[skip] {dataset_key} × {generator_name} (already completed)")
            continue

        t0 = time.perf_counter()
        tqdm.write(f"[run ] {dataset_key} × {generator_name} …")
        row = run_study(
            generator_name,
            dataset_key,
            results_root=results_root,
            n_trials=args.n_trials,
            optuna_cfg=optuna_cfg,
            show_trial_progress=args.show_trial_progress,
        )
        elapsed = time.perf_counter() - t0
        row["elapsed_sec"] = row.get("elapsed_sec") or elapsed
        summary_rows.append(row)
        _append_master(master_path, row)
        status = row.get("status")
        tqdm.write(
            f"[{status:8s}] {dataset_key} × {generator_name} "
            f"obj={row.get('objective')} time={row.get('elapsed_sec'):.1f}s"
        )

    total = time.perf_counter() - t_all
    summary_path = results_root / "experiment_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    n_ok = sum(1 for r in summary_rows if r["status"] == "completed")
    n_skip = sum(1 for r in summary_rows if r["status"] == "skipped_resume")
    n_fail = sum(1 for r in summary_rows if r["status"] == "failed")
    print("\n======== DONE ========")
    print(f"completed={n_ok}  resumed_skip={n_skip}  failed={n_fail}")
    print(f"wall_time={total:.1f}s")
    print(f"master_csv={master_path}")
    print(f"summary_csv={summary_path}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
