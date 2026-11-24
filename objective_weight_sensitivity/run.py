#!/usr/bin/env python3
"""CLI for objective-weight sensitivity pilot (Standard TPE, presets O1–O5).

Examples
--------
Smoke::

    python -m objective_weight_sensitivity.run --dry-run

Export Excel from existing CSVs (no Optuna)::

    python -m objective_weight_sensitivity.export_excel \\
      --results-root results/objective_weight_sensitivity_smoke

    python -m objective_weight_sensitivity.run --export-excel --analyze-only \\
      --results-root results/objective_weight_sensitivity_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datasets.loader import load_dataset_config
from evaluation.objective_presets import PRESET_NAMES, list_presets
from multivariate_tpe.datasets_catalog import GENERATORS_NO_FOREST
from objective_weight_sensitivity import DEFAULT_N_TRIALS, DEFAULT_SEED, N_STARTUP_TRIALS
from objective_weight_sensitivity.aggregate import rebuild_all
from objective_weight_sensitivity.plots import generate_all_plots
from objective_weight_sensitivity.runner import default_results_root, run_grid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    available = list(PRESET_NAMES)
    p = argparse.ArgumentParser(
        description=(
            "Objective-weight sensitivity pilot: independent Standard-TPE Optuna "
            "studies for presets O1–O5 (fidelity = SDV QualityReport only)."
        )
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(available),
        metavar="Oi",
        help=f"Objective presets to run (default: all of {available})",
    )
    p.add_argument("--datasets", nargs="+", default=["cancer"])
    p.add_argument("--generators", nargs="+", default=["gaussian_copula"])
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--n-startup-trials", type=int, default=N_STARTUP_TRIALS)
    p.add_argument(
        "--trial-timeout-s",
        type=float,
        default=7200.0,
        help="Per-trial hard limit in seconds (0 disables)",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Output root (default results/objective_weight_sensitivity)",
    )
    p.add_argument("--show-progress", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="2 trials × cancer × gaussian_copula × O1 only + aggregate/plots",
    )
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only rebuild CSVs/plots/Excel for --results-root (no Optuna)",
    )
    p.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip aggregate + plots + Excel after the grid",
    )
    p.add_argument(
        "--export-excel",
        action="store_true",
        help="Write objective_weight_comparison.xlsx (also done after analyze by default)",
    )
    return p.parse_args(argv)


def _export_excel(results_root: Path) -> Path:
    from objective_weight_sensitivity.export_excel import export_workbook

    path = export_workbook(results_root=results_root)
    print(f"Excel: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results_root = (
        Path(args.results_root) if args.results_root else default_results_root()
    )

    # ---- no Optuna paths ----
    if args.analyze_only:
        paths = rebuild_all(results_root)
        figs = generate_all_plots(results_root)
        print(f"Rebuilt: {paths}")
        print(f"Figures: {figs}")
        _export_excel(results_root)
        return 0

    # --export-excel alone (without dry-run): rebuild if needed, write xlsx, exit
    if args.export_excel and not args.dry_run:
        cmp_csv = results_root / "objective_weight_comparison.csv"
        if not cmp_csv.is_file():
            paths = rebuild_all(results_root)
            print(f"Rebuilt: {paths}")
            generate_all_plots(results_root)
        _export_excel(results_root)
        return 0

    # ---- Optuna grid ----
    datasets = list(args.datasets)
    generators = list(args.generators)
    configs = [c.upper() for c in args.configs]
    n_trials = int(args.n_trials)
    seed = int(args.seed)

    if args.dry_run:
        datasets = ["cancer"]
        generators = ["gaussian_copula"]
        configs = ["O1"]
        n_trials = 2
        seed = 42
        print("DRY-RUN: cancer × gaussian_copula × O1 × 2 trials × seed=42", flush=True)

    known_ds = set(load_dataset_config())
    bad_ds = [d for d in datasets if d not in known_ds]
    if bad_ds:
        raise SystemExit(f"Unknown datasets: {bad_ds}. Known e.g. {sorted(known_ds)[:8]}…")
    known_gen = set(GENERATORS_NO_FOREST)
    bad_gen = [g for g in generators if g not in known_gen]
    if bad_gen:
        raise SystemExit(f"Unknown generators: {bad_gen}. Expected subset of {sorted(known_gen)}")
    known_presets = set(list_presets())
    bad_cfg = [c for c in configs if c not in known_presets]
    if bad_cfg:
        raise SystemExit(f"Unknown configs: {bad_cfg}. Available: {sorted(known_presets)}")

    timeout = None if float(args.trial_timeout_s) <= 0 else float(args.trial_timeout_s)
    run_grid(
        datasets=datasets,
        generators=generators,
        configs=configs,
        seed=seed,
        n_trials=n_trials,
        results_root=results_root,
        n_startup_trials=int(args.n_startup_trials),
        trial_timeout_s=timeout,
        show_progress=bool(args.show_progress),
    )

    if not args.no_analyze:
        paths = rebuild_all(results_root)
        figs = generate_all_plots(results_root)
        print(f"\nResults root: {results_root}")
        print(f"Tables: {paths}")
        print(f"Figures: {figs}")
        _export_excel(results_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
