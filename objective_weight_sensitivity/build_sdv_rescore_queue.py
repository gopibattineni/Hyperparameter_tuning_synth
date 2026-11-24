#!/usr/bin/env python3
"""Build a TSV job queue for SDV fidelity re-score (one COMPLETE trial per line)."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
from optuna.trial import TrialState

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDIES = _ROOT / "results/multiseed_20261027/studies"
DEFAULT_QUEUE = (
    _ROOT
    / "results/objective_weight_sensitivity/tables/o1_posthoc_reweight"
    / "sdv_rescore_queue.tsv"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--studies-root", type=Path, default=DEFAULT_STUDIES)
    p.add_argument("--out", type=Path, default=DEFAULT_QUEUE)
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--gpu-only", action="store_true")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for db in sorted(args.studies_root.rglob("optuna.db")):
            parts = db.parts
            try:
                j = parts.index("studies")
                dataset, generator, sampler, seeddir = parts[j + 1 : j + 5]
            except Exception:
                continue
            if args.cpu_only and generator != "gaussian_copula":
                continue
            if args.gpu_only and generator == "gaussian_copula":
                continue
            seed = int(str(seeddir).replace("seed", ""))
            name = f"{dataset}__{generator}__{sampler}__seed{seed}"
            try:
                study = optuna.load_study(
                    study_name=name, storage=f"sqlite:///{db.resolve()}"
                )
            except Exception:
                continue
            for t in study.trials:
                if t.state != TrialState.COMPLETE:
                    continue
                # dataset generator sampler seed trial_number
                f.write(
                    f"{dataset}\t{generator}\t{sampler}\t{seed}\t{int(t.number)}\n"
                )
                n += 1
    print(f"Wrote {args.out} jobs={n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
