#!/usr/bin/env python3
"""CLI for multi-seed Standard TPE / Multivariate TPE / GPSampler comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multiseed_comparison import (
    ALL_SEEDS_10,
    DEFAULT_N_TRIALS,
    N_STARTUP_TRIALS,
    OBJECTIVE_PRESETS,
    SAMPLER_NAMES,
    SEEDS,
)
from multiseed_comparison.aggregate import rebuild_all
from multiseed_comparison.analyze_samplers import run_analysis
from multiseed_comparison.runner import default_results_root, run_grid
from multivariate_tpe.datasets_catalog import GENERATORS_NO_FOREST
from datasets.loader import load_dataset_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Multi-seed HPO: standard_tpe vs multivariate_tpe vs gp "
            f"(seeds={SEEDS}, n_trials={DEFAULT_N_TRIALS}, n_startup={N_STARTUP_TRIALS})"
        )
    )
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--generators", nargs="*", default=None)
    p.add_argument("--samplers", nargs="*", default=None, choices=list(SAMPLER_NAMES))
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument(
        "--all-10-seeds",
        action="store_true",
        help=f"Use the full O1 seed list {ALL_SEEDS_10}",
    )
    p.add_argument(
        "--objective-preset",
        type=str,
        default=None,
        choices=list(OBJECTIVE_PRESETS),
        help="O1–O5 weight preset (default: YAML weights = O1). Fidelity is SDV-only (KS excluded).",
    )
    p.add_argument(
        "--force-sdv-only",
        action="store_true",
        default=True,
        help="Set fidelity.aggregation=sdv_only (default). KS/corr are diagnostics only.",
    )
    p.add_argument(
        "--legacy-composite-fidelity",
        action="store_true",
        help="Replay historical fidelity.aggregation=legacy_composite (includes KS). Not used for O1–O5.",
    )
    p.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    p.add_argument("--n-startup-trials", type=int, default=N_STARTUP_TRIALS)
    p.add_argument(
        "--trial-timeout-s",
        type=float,
        default=7200.0,
        help="Per-trial hard limit in seconds (default 2h). Use 0 to disable.",
    )
    p.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Output root (default results/multiseed_YYYYMMDD)",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Reserved for future parallel studies; default 1 = sequential",
    )
    p.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES assignment for this process",
    )
    p.add_argument("--show-progress", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="2 trials × cancer × gaussian_copula × 3 samplers × 3 seeds + analysis",
    )
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only rebuild tables/plots for --results-root",
    )
    p.add_argument(
        "--no-analyze",
        action="store_true",
        help=(
            "Skip summary rebuild + plots after the grid "
            "(use when launching parallel studies that share a results root)"
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gpu is not None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Default: neural gens must use GPU (override with SYNTH_REQUIRE_CUDA=0).
    import os as _os

    _os.environ.setdefault("SYNTH_REQUIRE_CUDA", "1")

    if args.n_jobs != 1:
        print(
            f"NOTE: --n-jobs={args.n_jobs} requested but runner stays sequential "
            "so wall-clock comparisons stay fair. Ignoring parallel study launch."
        )

    results_root = (
        Path(args.results_root) if args.results_root else default_results_root()
    )

    if args.analyze_only:
        run_analysis(results_root)
        return 0

    timeout = None if args.trial_timeout_s is not None and args.trial_timeout_s <= 0 else args.trial_timeout_s

    if args.dry_run:
        print("=== DRY RUN: cancer × gaussian_copula, 2 trials, 3×3 sampler×seed ===")
        results_root = results_root.parent / (results_root.name + "_dryrun")
        # Disable hard timeout for the fast dry-run path (in-process eval).
        dry_timeout = None if timeout is None else timeout
        if args.trial_timeout_s == 7200.0:
            dry_timeout = None
        rows = run_grid(
            results_root=results_root,
            datasets=["cancer"],
            generators=["gaussian_copula"],
            samplers=list(SAMPLER_NAMES),
            seeds=list(SEEDS),
            n_trials=2,
            n_startup_trials=min(2, args.n_startup_trials),
            trial_timeout_s=dry_timeout,
            show_progress=args.show_progress,
        )
        rebuild_all(results_root)
        run_analysis(results_root)
        # Print tables
        import pandas as pd

        for name in ("summary_runs.csv", "summary_agg.csv", "pairwise.csv"):
            path = results_root / name
            print(f"\n===== {name} =====")
            if path.is_file():
                print(pd.read_csv(path).to_string(index=False))
            else:
                print("(missing)")
        print(f"\nDry-run outputs → {results_root}")
        print(f"Studies finished: {sum(1 for r in rows if r.get('status') in {'completed','skipped_resume','partial'})}/{len(rows)}")
        return 0

    datasets = args.datasets or list(load_dataset_config())
    generators = args.generators or list(GENERATORS_NO_FOREST)
    samplers = args.samplers or list(SAMPLER_NAMES)
    if args.seeds:
        seeds = list(args.seeds)
    elif args.all_10_seeds:
        seeds = list(ALL_SEEDS_10)
    else:
        seeds = list(SEEDS)

    run_grid(
        results_root=results_root,
        datasets=datasets,
        generators=generators,
        samplers=samplers,
        seeds=seeds,
        n_trials=args.n_trials,
        n_startup_trials=args.n_startup_trials,
        trial_timeout_s=timeout,
        show_progress=args.show_progress,
        objective_preset=args.objective_preset,
        force_sdv_only=not bool(args.legacy_composite_fidelity),
    )
    if args.no_analyze:
        print(f"Done (studies only, --no-analyze) → {results_root}")
        return 0
    try:
        rebuild_all(results_root)
        run_analysis(results_root)
    except Exception as exc:
        print(
            f"WARNING: studies finished but analysis failed: "
            f"{type(exc).__name__}: {exc}\n"
            f"Re-run: python multiseed_comparison/run.py --analyze-only "
            f"--results-root {results_root}"
        )
        return 2
    print(f"Done → {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
