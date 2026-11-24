#!/usr/bin/env python3
"""Compare standard TPE vs Multivariate TPE vs GPSampler on Cancer & Alzheimer's.

Reads:
  - standard_tpe/results/classification/{1_cancer,2_alzheimers}/<gen>/   (standard TPE baseline)
  - multivariate_tpe/results/...
  - gaussian_process/results/...

Writes comparison tables + figures under each experiment folder's ``figures/``
and a shared summary under ``sampler_comparison/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.run_experiments import RESEARCH_GENERATORS

DATASETS = ["cancer", "alzheimers"]
DATASET_FOLDER = {"cancer": "1_cancer", "alzheimers": "2_alzheimers"}
SAMPLERS = {
    "standard_tpe": _ROOT / "standard_tpe" / "results",
    "multivariate_tpe": _ROOT / "multivariate_tpe" / "results",
    "gaussian_process": _ROOT / "gaussian_process" / "results",
}
OUT = _ROOT / "sampler_comparison"
LABELS = {
    "standard_tpe": "Standard TPE",
    "multivariate_tpe": "Multivariate TPE",
    "gaussian_process": "Gaussian Process",
}
COLORS = {
    "standard_tpe": "#1f77b4",
    "multivariate_tpe": "#ff7f0e",
    "gaussian_process": "#2ca02c",
}


def _pair_dir(results_root: Path, dataset: str, generator: str) -> Path:
    return results_root / "classification" / DATASET_FOLDER[dataset] / generator


def _load_pair(results_root: Path, dataset: str, generator: str) -> dict[str, Any] | None:
    d = _pair_dir(results_root, dataset, generator)
    if not d.is_dir():
        return None
    row: dict[str, Any] = {
        "dataset": dataset,
        "generator": generator,
        "run_dir": str(d),
        "completed": (d / "completed.json").is_file(),
    }
    if (d / "completed.json").is_file():
        row.update(json.loads((d / "completed.json").read_text()))
    if (d / "metrics.json").is_file():
        row["metrics"] = json.loads((d / "metrics.json").read_text())
    if (d / "best_params.json").is_file():
        row["best_params"] = json.loads((d / "best_params.json").read_text())
    if (d / "cost_metrics.json").is_file():
        row["cost"] = json.loads((d / "cost_metrics.json").read_text())
    if (d / "timing.json").is_file():
        row["timing"] = json.loads((d / "timing.json").read_text())
    if (d / "trials.csv").is_file():
        df = pd.read_csv(d / "trials.csv")
        row["trials_df"] = df
        states = df["state"].astype(str) if "state" in df.columns else pd.Series(["COMPLETE"] * len(df))
        row["n_complete"] = int((states == "COMPLETE").sum())
        row["n_pruned"] = int((states == "PRUNED").sum())
        row["n_fail"] = int((states == "FAIL").sum())
        row["n_attempted"] = int(len(df))
        if "value" in df.columns and row["n_complete"]:
            complete = df[states == "COMPLETE"]
            row["best_value"] = float(complete["value"].max())
            row["best_trial"] = int(complete.loc[complete["value"].idxmax(), "number"])
        if "duration" in df.columns:
            secs = pd.to_timedelta(df["duration"], errors="coerce").dt.total_seconds()
            row["mean_trial_sec"] = float(secs[states == "COMPLETE"].mean())
        elapsed = None
        if row.get("cost") and row["cost"].get("elapsed_sec") is not None:
            elapsed = row["cost"]["elapsed_sec"]
        elif row.get("timing") and row["timing"].get("elapsed_sec") is not None:
            elapsed = row["timing"]["elapsed_sec"]
        elif row.get("elapsed_sec") is not None:
            elapsed = row["elapsed_sec"]
        row["elapsed_sec"] = elapsed
    return row


def _cumulative_best(df: pd.DataFrame) -> pd.DataFrame:
    states = df["state"].astype(str) if "state" in df.columns else pd.Series(["COMPLETE"] * len(df))
    complete = df[states == "COMPLETE"].sort_values("number").copy()
    if complete.empty:
        return pd.DataFrame(columns=["number", "best_so_far", "elapsed_cum"])
    complete["best_so_far"] = complete["value"].cummax()
    if "datetime_complete" in complete.columns and "datetime_start" in df.columns:
        t0 = pd.to_datetime(df["datetime_start"].iloc[0])
        complete["elapsed_cum"] = (
            pd.to_datetime(complete["datetime_complete"]) - t0
        ).dt.total_seconds()
    else:
        complete["elapsed_cum"] = complete["number"].astype(float)
    return complete[["number", "best_so_far", "elapsed_cum", "value"]]


def build_table() -> pd.DataFrame:
    rows = []
    for sampler, root in SAMPLERS.items():
        for ds in DATASETS:
            for gen in RESEARCH_GENERATORS:
                info = _load_pair(root, ds, gen)
                if info is None:
                    continue
                rows.append(
                    {
                        "sampler": sampler,
                        "sampler_label": LABELS[sampler],
                        "dataset": ds,
                        "generator": gen,
                        "best_value": info.get("best_value"),
                        "best_trial": info.get("best_trial"),
                        "elapsed_sec": info.get("elapsed_sec"),
                        "mean_trial_sec": info.get("mean_trial_sec"),
                        "n_attempted": info.get("n_attempted"),
                        "n_complete": info.get("n_complete"),
                        "n_pruned": info.get("n_pruned"),
                        "n_fail": info.get("n_fail"),
                        "completed": info.get("completed"),
                        "efficiency_obj_per_hour": (
                            (info["best_value"] / (info["elapsed_sec"] / 3600.0))
                            if info.get("best_value") is not None
                            and info.get("elapsed_sec")
                            and info["elapsed_sec"] > 0
                            else None
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_dataset(dataset: str, table: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    gens = [g for g in RESEARCH_GENERATORS if g in set(table.loc[table.dataset == dataset, "generator"])]

    # Bar: best objective by generator × sampler
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(gens))
    width = 0.25
    for i, (sampler, label) in enumerate(LABELS.items()):
        vals = []
        for g in gens:
            sub = table[(table.dataset == dataset) & (table.generator == g) & (table.sampler == sampler)]
            vals.append(float(sub.best_value.iloc[0]) if len(sub) and pd.notna(sub.best_value.iloc[0]) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=COLORS[sampler])
    ax.set_xticks(x)
    ax.set_xticklabels(gens, rotation=30, ha="right")
    ax.set_ylabel("Best objective (maximize)")
    ax.set_title(f"{dataset}: best objective by sampler")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{dataset}_best_objective_bars.png", dpi=150)
    plt.close(fig)

    # Runtime bars
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (sampler, label) in enumerate(LABELS.items()):
        vals = []
        for g in gens:
            sub = table[(table.dataset == dataset) & (table.generator == g) & (table.sampler == sampler)]
            vals.append(float(sub.elapsed_sec.iloc[0]) / 60.0 if len(sub) and pd.notna(sub.elapsed_sec.iloc[0]) else np.nan)
        ax.bar(x + (i - 1) * width, vals, width, label=label, color=COLORS[sampler])
    ax.set_xticks(x)
    ax.set_xticklabels(gens, rotation=30, ha="right")
    ax.set_ylabel("Total wall-clock (minutes)")
    ax.set_title(f"{dataset}: optimisation runtime by sampler")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{dataset}_runtime_bars.png", dpi=150)
    plt.close(fig)

    # Per-generator optimisation history + cumulative best vs time
    for gen in gens:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for sampler, label in LABELS.items():
            info = _load_pair(SAMPLERS[sampler], dataset, gen)
            if not info or "trials_df" not in info:
                continue
            df = info["trials_df"]
            states = df["state"].astype(str) if "state" in df.columns else pd.Series(["COMPLETE"] * len(df))
            complete = df[states == "COMPLETE"].sort_values("number")
            if complete.empty:
                continue
            axes[0].plot(complete["number"], complete["value"], "o-", label=label, color=COLORS[sampler], ms=3)
            cum = _cumulative_best(df)
            axes[1].plot(cum["elapsed_cum"] / 60.0, cum["best_so_far"], "-", label=label, color=COLORS[sampler])
        axes[0].set_xlabel("Trial number")
        axes[0].set_ylabel("Objective (maximize)")
        axes[0].set_title(f"{dataset} / {gen}: objective vs trial")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[1].set_xlabel("Elapsed time (minutes)")
        axes[1].set_ylabel("Cumulative best objective")
        axes[1].set_title(f"{dataset} / {gen}: best vs time")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{dataset}_{gen}_history.png", dpi=150)
        plt.close(fig)

        # Trial duration distribution
        fig, ax = plt.subplots(figsize=(7, 4))
        data, labels = [], []
        for sampler, label in LABELS.items():
            info = _load_pair(SAMPLERS[sampler], dataset, gen)
            if not info or "trials_df" not in info or "duration" not in info["trials_df"].columns:
                continue
            df = info["trials_df"]
            states = df["state"].astype(str) if "state" in df.columns else pd.Series(["COMPLETE"] * len(df))
            secs = pd.to_timedelta(df.loc[states == "COMPLETE", "duration"], errors="coerce").dt.total_seconds().dropna()
            if len(secs):
                data.append(secs.values)
                labels.append(label)
        if data:
            ax.boxplot(data, tick_labels=labels, showfliers=False)
            ax.set_ylabel("Trial duration (seconds)")
            ax.set_title(f"{dataset} / {gen}: trial duration distribution")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{dataset}_{gen}_trial_duration.png", dpi=150)
        plt.close(fig)


def write_summary(table: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "comparison_table.csv", index=False)
    try:
        table.to_excel(out_dir / "comparison_table.xlsx", index=False)
    except Exception:
        pass

    lines = [
        "# Bayesian sampler comparison (Cancer & Alzheimer's)",
        "",
        "Objective is **maximized**: `0.4×Fidelity + 0.4×Utility − 0.2×PrivacyRisk`.",
        "",
        "## Scope",
        "- Datasets: cancer, alzheimers",
        f"- Generators: {', '.join(RESEARCH_GENERATORS)}",
        "- Samplers: Standard TPE (baseline), Multivariate TPE, Gaussian Process (GPSampler)",
        "",
        "## Pruning note",
        "The shared `TrialObjective` does **not** call `trial.report()` / `should_prune()`.",
        "Configured `MedianPruner` therefore cannot prune on intermediate metrics.",
        "Generator exceptions are recorded as Optuna `PRUNED` trials via `TrialPruned`.",
        "",
        "## GPSampler compatibility",
        "Optuna 5.0 GPSampler requires `with torch.device(...):` (PyTorch ≥ 2.1).",
        "This environment uses torch 1.13.x; `gaussian_process/run.py` applies a",
        "context-manager shim so GPSampler still runs (not a sampler substitute).",
        "",
        "## Aggregate best objective (higher is better)",
        "",
    ]
    pivot = table.pivot_table(
        index=["dataset", "generator"],
        columns="sampler_label",
        values="best_value",
        aggfunc="first",
    )
    try:
        lines.append(pivot.to_markdown())
    except Exception:
        lines.append("```\n" + pivot.to_string() + "\n```")
    lines.append("")
    lines.append("## Aggregate runtime (seconds)")
    lines.append("")
    pivot_t = table.pivot_table(
        index=["dataset", "generator"],
        columns="sampler_label",
        values="elapsed_sec",
        aggfunc="first",
    )
    try:
        lines.append(pivot_t.to_markdown())
    except Exception:
        lines.append("```\n" + pivot_t.to_string() + "\n```")
    lines.append("")

    # Wins / savings
    lines.append("## Headlines")
    for ds in DATASETS:
        sub = table[table.dataset == ds]
        for gen in RESEARCH_GENERATORS:
            g = sub[sub.generator == gen]
            if g.empty or g.best_value.isna().all():
                continue
            best_row = g.loc[g.best_value.idxmax()]
            fast_row = g.dropna(subset=["elapsed_sec"]).sort_values("elapsed_sec").head(1)
            lines.append(
                f"- **{ds}/{gen}**: best objective = {best_row.best_value:.4f} "
                f"({best_row.sampler_label})"
                + (
                    f"; fastest wall-clock = {float(fast_row.elapsed_sec.iloc[0]):.1f}s "
                    f"({fast_row.sampler_label.iloc[0]})"
                    if len(fast_row)
                    else ""
                )
            )
    lines.append("")
    path = out_dir / "SUMMARY.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    table = build_table()
    if table.empty:
        print("No results found yet. Run multivariate_tpe/run.py and gaussian_process/run.py first.")
        return 1
    write_summary(table, OUT)
    fig_dir = OUT / "figures"
    for ds in DATASETS:
        plot_dataset(ds, table, fig_dir)
    # Mirror figures into experiment folders (copy, do not re-render)
    import shutil

    for exp in ("multivariate_tpe", "gaussian_process"):
        dest = _ROOT / exp / "figures"
        dest.mkdir(parents=True, exist_ok=True)
        if fig_dir.is_dir():
            for png in fig_dir.glob("*.png"):
                shutil.copy2(png, dest / png.name)
    print(f"Wrote comparison to {OUT}")
    print(table.dropna(subset=["best_value"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
