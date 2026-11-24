#!/usr/bin/env python3
"""Export O1 multiseed Optuna trial attrs (incl. sdv/ks/corr) to one CSV.

Historical raw CSVs only stored composite ``fidelity`` (often KS+Corr because
SDV was null). This export recovers per-trial diagnostics from ``optuna.db``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import TrialState

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDIES = _ROOT / "results/multiseed_20261027/studies"
DEFAULT_OUT = (
    _ROOT
    / "results/objective_weight_sensitivity/tables/o1_posthoc_reweight"
    / "O1_trials_with_fidelity_components.csv"
)


def _study_name(dataset: str, generator: str, sampler: str, seed: int) -> str:
    return f"{dataset}__{generator}__{sampler}__seed{seed}"


def export_studies(studies_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    dbs = sorted(studies_root.rglob("optuna.db"))
    for i, db in enumerate(dbs, 1):
        parts = db.parts
        try:
            j = parts.index("studies")
            dataset, generator, sampler, seeddir = parts[j + 1 : j + 5]
        except Exception:
            continue
        seed = int(str(seeddir).replace("seed", ""))
        name = _study_name(dataset, generator, sampler, seed)
        try:
            study = optuna.load_study(
                study_name=name, storage=f"sqlite:///{db.resolve()}"
            )
        except Exception:
            continue
        for t in study.trials:
            if t.state != TrialState.COMPLETE:
                continue
            attrs = dict(t.user_attrs or {})
            row = {
                "dataset": dataset,
                "generator": generator,
                "sampler": sampler,
                "seed": seed,
                "trial_number": int(t.number),
                "state": "COMPLETE",
                "objective": t.value,
                "fidelity": attrs.get("fidelity"),
                "sdv_quality": attrs.get("sdv_quality"),
                "ks_similarity": attrs.get("ks_similarity"),
                "corr_similarity": attrs.get("corr_similarity"),
                "fidelity_aggregation": attrs.get("fidelity_aggregation"),
                "utility": attrs.get("utility"),
                "privacy_risk": attrs.get("privacy_risk"),
                "trial_seed": attrs.get("trial_seed"),
                "duration_s": attrs.get("duration_s"),
            }
            for k, v in (t.params or {}).items():
                row[f"param_{k}"] = v
            rows.append(row)
        if i % 200 == 0:
            print(f"  scanned {i}/{len(dbs)} studies → {len(rows)} complete trials", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--studies-root", type=Path, default=DEFAULT_STUDIES)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting COMPLETE trials from {args.studies_root}", flush=True)
    df = export_studies(args.studies_root)
    df.to_csv(args.out, index=False)
    n_sdv = int(df["sdv_quality"].notna().sum()) if len(df) else 0
    print(
        f"Wrote {args.out} rows={len(df)} "
        f"studies≈{df.groupby(['dataset','generator','sampler','seed']).ngroups if len(df) else 0} "
        f"sdv_nonnull={n_sdv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
