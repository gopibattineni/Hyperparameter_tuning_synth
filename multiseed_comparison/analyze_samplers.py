#!/usr/bin/env python3
"""Statistical comparison + publication plots for multi-seed sampler HPO."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multiseed_comparison import SAMPLER_NAMES
from multiseed_comparison.aggregate import rebuild_all

# Serif style suitable for LaTeX inclusion
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

SAMPLER_LABEL = {
    "standard_tpe": "Standard TPE",
    "multivariate_tpe": "Multivariate TPE",
    "gp": "GPSampler",
}
SAMPLER_ORDER = list(SAMPLER_NAMES)


def _complete_blocks(
    summary_runs: pd.DataFrame, n_seeds: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep (dataset, generator) blocks where all 3 samplers finished all seeds."""
    rows: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    if summary_runs.empty:
        return pd.DataFrame(), pd.DataFrame()
    for (ds, gen), g in summary_runs.groupby(["dataset", "generator"]):
        ok = True
        reason: list[str] = []
        for samp in SAMPLER_ORDER:
            sub = g[g["sampler"] == samp]
            vals = sub["best_objective"].dropna()
            if len(vals) < n_seeds:
                ok = False
                reason.append(f"{samp}:{len(vals)}/{n_seeds}")
        if ok:
            means = (
                g.dropna(subset=["best_objective"])
                .groupby("sampler")["best_objective"]
                .mean()
            )
            rows.append(
                {
                    "dataset": ds,
                    "generator": gen,
                    **{f"j_{s}": float(means[s]) for s in SAMPLER_ORDER if s in means},
                }
            )
        else:
            dropped.append(
                {"dataset": ds, "generator": gen, "reason": "; ".join(reason)}
            )
    return pd.DataFrame(rows), pd.DataFrame(dropped)


def friedman_wilcoxon(blocks: pd.DataFrame) -> dict[str, Any]:
    """Friedman test + pairwise Wilcoxon with Holm correction."""
    from scipy.stats import friedmanchisquare, wilcoxon
    from statsmodels.stats.multitest import multipletests

    cols = [f"j_{s}" for s in SAMPLER_ORDER]
    missing = [c for c in cols if c not in blocks.columns]
    if missing or blocks.empty or len(blocks) < 2:
        return {
            "n_blocks": int(len(blocks)),
            "friedman_stat": None,
            "friedman_p": None,
            "pairs": [],
            "note": "Need ≥2 complete blocks with all three samplers",
        }

    arrays = [blocks[c].to_numpy(dtype=float) for c in cols]
    stat, p = friedmanchisquare(*arrays)

    pair_defs = [
        ("multivariate_tpe", "standard_tpe", "MT vs ST"),
        ("gp", "standard_tpe", "GP vs ST"),
        ("gp", "multivariate_tpe", "GP vs MT"),
    ]
    raw_p: list[float] = []
    pair_rows: list[dict[str, Any]] = []
    for a, b, label in pair_defs:
        xa = blocks[f"j_{a}"].to_numpy(dtype=float)
        xb = blocks[f"j_{b}"].to_numpy(dtype=float)
        diff = xa - xb
        if np.allclose(diff, 0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = wilcoxon(xa, xb, zero_method="wilcox", alternative="two-sided")
        raw_p.append(float(w_p))
        pair_rows.append(
            {
                "pair": label,
                "mean_delta": float(diff.mean()),
                "wilcoxon_stat": float(w_stat),
                "p_raw": float(w_p),
            }
        )
    reject, p_corr, _, _ = multipletests(raw_p, method="holm")
    for i, row in enumerate(pair_rows):
        row["p_holm"] = float(p_corr[i])
        row["reject_holm_0.05"] = bool(reject[i])

    return {
        "n_blocks": int(len(blocks)),
        "friedman_stat": float(stat),
        "friedman_p": float(p),
        "pairs": pair_rows,
    }


def plot_best_j_bars(summary_runs: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Best J mean±std across seeds, faceted by dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if summary_runs.empty:
        return paths
    datasets = list(summary_runs["dataset"].dropna().unique())
    n = len(datasets)
    if n == 0:
        return paths
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    x = np.arange(len(SAMPLER_ORDER))
    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        sub = summary_runs[summary_runs["dataset"] == ds]
        means, stds = [], []
        for s in SAMPLER_ORDER:
            vals = sub.loc[sub["sampler"] == s, "best_objective"].dropna().astype(float)
            means.append(float(vals.mean()) if len(vals) else np.nan)
            stds.append(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0)
        ax.bar(x, means, yerr=stds, capsize=3, color=["#4C72B0", "#55A868", "#C44E52"])
        ax.set_xticks(x)
        ax.set_xticklabels([SAMPLER_LABEL[s] for s in SAMPLER_ORDER], rotation=15, ha="right")
        ax.set_ylabel("Best objective $J$")
        ax.set_title(ds)
        ax.grid(True, axis="y", alpha=0.3)
    for j in range(i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"best_j_by_dataset.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def _best_so_far_curve(seed_df: pd.DataFrame) -> pd.Series:
    """Running best objective indexed by unique trial number."""
    sdf = seed_df.dropna(subset=["objective", "trial_number"]).copy()
    if sdf.empty:
        return pd.Series(dtype=float)
    # One value per trial (duplicates can appear if raw CSVs were rewritten).
    sdf = (
        sdf.sort_values("trial_number")
        .groupby("trial_number", as_index=True)["objective"]
        .max()
        .sort_index()
    )
    return sdf.cummax()


def plot_convergence(raw: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Best-so-far J vs trial, mean±std across (dataset, seed); one panel per generator."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if raw.empty or "objective" not in raw.columns:
        return paths
    complete = raw[raw["state"] == "COMPLETE"].copy()
    if complete.empty:
        return paths
    generators = list(complete["generator"].dropna().unique())
    n = len(generators)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False
    )
    colors = {"standard_tpe": "#4C72B0", "multivariate_tpe": "#55A868", "gp": "#C44E52"}
    last_i = 0
    for i, gen in enumerate(generators):
        last_i = i
        ax = axes[i // ncols][i % ncols]
        gsub = complete[complete["generator"] == gen]
        for samp in SAMPLER_ORDER:
            ssub = gsub[gsub["sampler"] == samp]
            if ssub.empty:
                continue
            # One curve per (dataset, seed) so multi-dataset panels do not collide.
            group_cols = [c for c in ("dataset", "seed") if c in ssub.columns]
            if not group_cols:
                group_cols = ["seed"] if "seed" in ssub.columns else []
            curves: list[pd.Series] = []
            if group_cols:
                for _, seed_df in ssub.groupby(group_cols, dropna=False):
                    curve = _best_so_far_curve(seed_df)
                    if not curve.empty:
                        curves.append(curve)
            else:
                curve = _best_so_far_curve(ssub)
                if not curve.empty:
                    curves.append(curve)
            if not curves:
                continue
            mat = pd.concat(curves, axis=1, join="outer")
            mat = mat[~mat.index.duplicated(keep="first")].sort_index()
            mean = mat.mean(axis=1, skipna=True)
            std = mat.std(axis=1, ddof=1).fillna(0.0)
            ax.plot(
                mean.index,
                mean.values,
                label=SAMPLER_LABEL[samp],
                color=colors[samp],
            )
            ax.fill_between(
                mean.index,
                mean.values - std.values,
                mean.values + std.values,
                color=colors[samp],
                alpha=0.2,
            )
        ax.set_title(gen)
        ax.set_xlabel("Trial")
        ax.set_ylabel("Best-so-far $J$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", frameon=False)
    for j in range(last_i + 1, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"convergence_by_generator.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def plot_gain_vs_cost(pairwise: pd.DataFrame, out_dir: Path) -> list[Path]:
    """ΔJ vs Δ hours relative to Standard TPE (MT−ST and GP−ST)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if pairwise.empty:
        return paths
    sub = pairwise[pairwise["pair"].isin(["MT-ST", "GP-ST"])].copy()
    if sub.empty:
        return paths
    sub["delta_hours"] = sub["mean_delta_time_s"] / 3600.0
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for label, color in (("MT-ST", "#55A868"), ("GP-ST", "#C44E52")):
        p = sub[sub["pair"] == label]
        ax.scatter(
            p["delta_hours"],
            p["mean_delta_best_j"],
            label=label,
            color=color,
            s=40,
            alpha=0.85,
        )
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel(r"$\Delta$ hours vs Standard TPE")
    ax.set_ylabel(r"$\Delta$ best $J$ vs Standard TPE")
    ax.set_title("Objective gain vs extra compute")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = out_dir / f"gain_vs_cost.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def run_analysis(results_root: Path) -> dict[str, Any]:
    """Rebuild tables, run tests, write plots + stats_report.txt."""
    results_root = Path(results_root)
    tables = rebuild_all(results_root)
    runs = tables["summary_runs"]
    pairwise = tables["pairwise"]
    fig_dir = results_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    blocks, dropped = _complete_blocks(runs)
    stats = friedman_wilcoxon(blocks)
    if not dropped.empty:
        dropped.to_csv(results_root / "dropped_blocks.csv", index=False)
    if not blocks.empty:
        blocks.to_csv(results_root / "complete_blocks.csv", index=False)

    raw_files = list((results_root / "raw").glob("*.csv")) if (results_root / "raw").is_dir() else []
    raw = pd.concat([pd.read_csv(p) for p in raw_files], ignore_index=True) if raw_files else pd.DataFrame()

    paths = []
    paths += plot_best_j_bars(runs, fig_dir)
    paths += plot_convergence(raw, fig_dir)
    paths += plot_gain_vs_cost(pairwise, fig_dir)

    report = results_root / "stats_report.txt"
    lines = [
        f"Results root: {results_root}",
        f"Complete blocks (all 3 samplers × 3 seeds): {stats['n_blocks']}",
        f"Dropped blocks: {len(dropped)}",
    ]
    if len(dropped):
        lines.append("Dropped reasons (first 20):")
        for _, r in dropped.head(20).iterrows():
            lines.append(f"  {r['dataset']}/{r['generator']}: {r['reason']}")
    if stats.get("friedman_p") is not None:
        lines.append(
            f"Friedman χ²={stats['friedman_stat']:.4f}, p={stats['friedman_p']:.4g}"
        )
        for pr in stats["pairs"]:
            lines.append(
                f"  {pr['pair']}: mean ΔJ={pr['mean_delta']:+.4f}, "
                f"p_raw={pr['p_raw']:.4g}, p_Holm={pr['p_holm']:.4g}, "
                f"reject={pr['reject_holm_0.05']}"
            )
    else:
        lines.append(stats.get("note", "Friedman not run"))
    lines.append("Figures:")
    for p in paths:
        lines.append(f"  {p}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(report.read_text(encoding="utf-8"))
    return {"stats": stats, "dropped": dropped, "figures": paths, "tables": tables}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyze multi-seed sampler HPO results")
    p.add_argument(
        "--results-root",
        required=True,
        help="e.g. results/multiseed_20261027",
    )
    args = p.parse_args(argv)
    run_analysis(Path(args.results_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
