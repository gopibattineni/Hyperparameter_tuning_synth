"""Build summary_runs / summary_agg / pairwise CSVs from raw trial files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

_STATUS_RANK = {
    "completed": 0,
    "skipped_resume": 1,
    "partial": 2,
    "failed": 3,
}


def _load_all_raw(results_root: Path) -> pd.DataFrame:
    raw_dir = Path(results_root) / "raw"
    files = sorted(raw_dir.glob("*.csv")) if raw_dir.is_dir() else []
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in files]
    return pd.concat(frames, ignore_index=True)


def _dedupe_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per (dataset, generator, sampler, seed); prefer completed."""
    if df.empty:
        return df
    keys = ["dataset", "generator", "sampler", "seed"]
    missing = [k for k in keys if k not in df.columns]
    if missing:
        return df
    out = df.copy()
    out["_rank"] = out["status"].map(lambda s: _STATUS_RANK.get(str(s), 9))
    out = (
        out.sort_values("_rank")
        .drop_duplicates(subset=keys, keep="first")
        .drop(columns=["_rank"])
        .reset_index(drop=True)
    )
    return out


def build_summary_runs(results_root: Path) -> pd.DataFrame:
    """One row per (dataset, generator, sampler, seed)."""
    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    # Prefer study summary.json (has wall clock); enrich from raw if needed.
    for meta in sorted((root / "studies").rglob("summary.json")):
        s = json.loads(meta.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": s.get("dataset"),
                "generator": s.get("generator"),
                "sampler": s.get("sampler"),
                "seed": s.get("seed"),
                "status": s.get("status"),
                "best_objective": s.get("best_objective"),
                "best_fidelity": s.get("best_fidelity"),
                "best_utility": s.get("best_utility"),
                "best_privacy_risk": s.get("best_privacy_risk"),
                "best_trial_number": s.get("best_trial_number"),
                "n_complete": s.get("n_complete"),
                "n_fail": s.get("n_fail"),
                "n_pruned": s.get("n_pruned"),
                "n_attempted": s.get("n_attempted"),
                "total_time_s": s.get("sum_trial_duration_s") or s.get("study_wall_clock_s"),
                "mean_trial_time_s": s.get("mean_trial_duration_s"),
                "study_wall_clock_s": s.get("study_wall_clock_s"),
                "hostname": s.get("hostname"),
                "gpu_name": s.get("gpu_name"),
            }
        )
    if not rows:
        raw = _load_all_raw(root)
        if raw.empty:
            return pd.DataFrame()
        for keys, g in raw.groupby(
            ["dataset", "generator", "sampler", "seed"], dropna=False
        ):
            complete = g[g["state"] == "COMPLETE"]
            best_row = None
            if len(complete):
                best_row = complete.loc[complete["objective"].idxmax()]
            rows.append(
                {
                    "dataset": keys[0],
                    "generator": keys[1],
                    "sampler": keys[2],
                    "seed": keys[3],
                    "status": "completed" if len(complete) else "empty",
                    "best_objective": None if best_row is None else best_row["objective"],
                    "best_fidelity": None if best_row is None else best_row.get("fidelity"),
                    "best_utility": None if best_row is None else best_row.get("utility"),
                    "best_privacy_risk": None
                    if best_row is None
                    else best_row.get("privacy_risk"),
                    "best_trial_number": None
                    if best_row is None
                    else best_row.get("trial_number"),
                    "n_complete": int((g["state"] == "COMPLETE").sum()),
                    "n_fail": int((g["state"] == "FAIL").sum()),
                    "n_pruned": int((g["state"] == "PRUNED").sum()),
                    "n_attempted": int(len(g)),
                    "total_time_s": float(g["duration_s"].dropna().sum())
                    if "duration_s" in g
                    else None,
                    "mean_trial_time_s": float(g["duration_s"].dropna().mean())
                    if "duration_s" in g and g["duration_s"].notna().any()
                    else None,
                }
            )
    df = _dedupe_runs(pd.DataFrame(rows))
    out = root / "summary_runs.csv"
    df.to_csv(out, index=False)
    return df


def build_summary_agg(summary_runs: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    """One row per (dataset, generator, sampler): mean±std across seeds."""
    if summary_runs.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, g in summary_runs.groupby(["dataset", "generator", "sampler"]):
        vals = g["best_objective"].dropna().astype(float)
        times = g["total_time_s"].dropna().astype(float)
        rows.append(
            {
                "dataset": keys[0],
                "generator": keys[1],
                "sampler": keys[2],
                "n_seeds_finished": int(vals.shape[0]),
                "best_j_mean": float(vals.mean()) if len(vals) else None,
                "best_j_std": float(vals.std(ddof=1))
                if len(vals) > 1
                else (0.0 if len(vals) else None),
                "best_j_min": float(vals.min()) if len(vals) else None,
                "best_j_max": float(vals.max()) if len(vals) else None,
                "total_time_mean_s": float(times.mean()) if len(times) else None,
                "total_time_std_s": float(times.std(ddof=1))
                if len(times) > 1
                else (0.0 if len(times) else None),
            }
        )
    df = pd.DataFrame(rows)
    out = Path(results_root) / "summary_agg.csv"
    df.to_csv(out, index=False)
    return df


def build_pairwise(summary_runs: pd.DataFrame, results_root: Path) -> pd.DataFrame:
    """Paired ΔJ and Δtime for MT−ST, GP−ST, GP−MT across seeds."""
    if summary_runs.empty:
        return pd.DataFrame()
    pairs = [
        ("multivariate_tpe", "standard_tpe", "MT-ST"),
        ("gp", "standard_tpe", "GP-ST"),
        ("gp", "multivariate_tpe", "GP-MT"),
    ]
    rows: list[dict[str, Any]] = []
    for (ds, gen), block in summary_runs.groupby(["dataset", "generator"]):
        pivot_j = block.pivot_table(
            index="seed", columns="sampler", values="best_objective", aggfunc="first"
        )
        pivot_t = block.pivot_table(
            index="seed", columns="sampler", values="total_time_s", aggfunc="first"
        )
        for chal, base, label in pairs:
            if chal not in pivot_j.columns or base not in pivot_j.columns:
                continue
            sub = pivot_j[[chal, base]].dropna()
            if sub.empty:
                continue
            delta = sub[chal] - sub[base]
            wins = int((delta > 0).sum())
            tsub = pivot_t.reindex(sub.index)
            dt = None
            if chal in tsub.columns and base in tsub.columns:
                tt = tsub[[chal, base]].dropna()
                if len(tt):
                    dt = float((tt[chal] - tt[base]).mean())
            rows.append(
                {
                    "dataset": ds,
                    "generator": gen,
                    "pair": label,
                    "challenger": chal,
                    "baseline": base,
                    "n_seeds_compared": int(len(sub)),
                    "mean_delta_best_j": float(delta.mean()),
                    "challenger_wins": wins,
                    "mean_delta_time_s": dt,
                }
            )
    df = pd.DataFrame(rows)
    out = Path(results_root) / "pairwise.csv"
    df.to_csv(out, index=False)
    return df


def rebuild_all(results_root: Path) -> dict[str, pd.DataFrame]:
    """Rewrite summary_runs / summary_agg / pairwise under ``results_root``."""
    runs = build_summary_runs(results_root)
    agg = build_summary_agg(runs, results_root)
    pair = build_pairwise(runs, results_root)
    return {"summary_runs": runs, "summary_agg": agg, "pairwise": pair}
