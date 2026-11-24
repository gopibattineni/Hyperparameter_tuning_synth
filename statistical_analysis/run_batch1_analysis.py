#!/usr/bin/env python3
"""Statistical comparison of Standard TPE vs Multivariate TPE vs GP.

Default: all six paired seeds 42, 123, 2024, 7, 99, 1337.
Sources (read-only): results/multiseed_20261027/studies/**/summary.json
Writes:
  results/multiseed_6seeds_42_123_2024_7_99_1337/   curated tables
  statistical_analysis/                             tests, figures, report

Override with --seeds / --batch-dir / --out-dir (e.g. batch-1 three seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from multivariate_tpe.datasets_catalog import (
    ALL_DATASETS,
    CLASSIFICATION_DATASETS,
    GENERATORS_NO_FOREST,
    REGRESSION_DATASETS,
)

DEFAULT_SEEDS = [42, 123, 2024, 7, 99, 1337]
SEEDS = list(DEFAULT_SEEDS)
SEED_LABELS: dict[int, str] = {s: f"seed{i}_{s}" for i, s in enumerate(SEEDS, start=1)}
SAMPLERS = ["standard_tpe", "multivariate_tpe", "gp"]
SAMPLER_LABEL = {
    "standard_tpe": "Standard TPE",
    "multivariate_tpe": "Multivariate TPE",
    "gp": "Gaussian Process",
}
STUDIES_ROOT = _ROOT / "results" / "multiseed_20261027" / "studies"
BATCH_DIR = _ROOT / "results" / "multiseed_6seeds_42_123_2024_7_99_1337"
OUT_DIR = _ROOT / "statistical_analysis"
FIG_DIR = OUT_DIR / "figures"


def _configure_run(seeds: list[int], batch_dir: Path, out_dir: Path) -> None:
    global SEEDS, SEED_LABELS, BATCH_DIR, OUT_DIR, FIG_DIR
    SEEDS = list(seeds)
    SEED_LABELS = {s: f"seed{i}_{s}" for i, s in enumerate(SEEDS, start=1)}
    BATCH_DIR = batch_dir
    OUT_DIR = out_dir
    FIG_DIR = OUT_DIR / "figures"


def _seed_list_str() -> str:
    return ", ".join(str(s) for s in SEEDS)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
    }
)


def load_study_summaries() -> pd.DataFrame:
    """Load best-J for each (dataset, generator, sampler, seed) study."""
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for sampler in SAMPLERS:
        for seed in SEEDS:
            for ds in ALL_DATASETS:
                for gen in GENERATORS_NO_FOREST:
                    path = (
                        STUDIES_ROOT
                        / ds
                        / gen
                        / sampler
                        / f"seed{seed}"
                        / "summary.json"
                    )
                    if not path.is_file():
                        missing.append(str(path))
                        continue
                    d = json.loads(path.read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "dataset": ds,
                            "generator": gen,
                            "sampler": sampler,
                            "algorithm": SAMPLER_LABEL[sampler],
                            "seed": int(seed),
                            "best_objective": d.get("best_objective"),
                            "best_fidelity": d.get("best_fidelity"),
                            "best_utility": d.get("best_utility"),
                            "best_privacy_risk": d.get("best_privacy_risk"),
                            "n_complete": d.get("n_complete"),
                            "status": d.get("status"),
                            "imported": "imported_from" in d,
                            "source_summary": str(path),
                        }
                    )
    df = pd.DataFrame(rows)
    return df, missing


def validate(df: pd.DataFrame, missing: list[str]) -> list[str]:
    """Return human-readable validation notes; raise if primary data unusable."""
    notes: list[str] = []
    if missing:
        notes.append(f"Missing summary.json files: {len(missing)}")
    expected = len(ALL_DATASETS) * len(GENERATORS_NO_FOREST) * len(SAMPLERS) * len(SEEDS)
    notes.append(f"Loaded rows: {len(df)} / expected {expected}")
    bad = df["best_objective"].isna().sum()
    if bad:
        notes.append(f"Rows with missing best_objective: {int(bad)}")
    # Completeness matrix
    for sampler in SAMPLERS:
        for seed in SEEDS:
            sub = df[(df["sampler"] == sampler) & (df["seed"] == seed)]
            n = sub["best_objective"].notna().sum()
            notes.append(
                f"  {SAMPLER_LABEL[sampler]} seed={seed}: {n}/105 studies with best J"
            )
    # Duplicates
    dup = df.duplicated(
        subset=["dataset", "generator", "sampler", "seed"], keep=False
    ).sum()
    if dup:
        notes.append(f"WARNING: duplicate keys: {int(dup)}")
    return notes


def build_seed_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, sampler, seed): mean best-J across 7 generators.

    Primary seed-level measure for a dataset = mean over generators of that
    study's best objective (fair multi-generator summary; maximisation).
    """
    complete = df.dropna(subset=["best_objective"]).copy()
    g = (
        complete.groupby(["dataset", "sampler", "algorithm", "seed"], as_index=False)
        .agg(
            best_objective=("best_objective", "mean"),
            n_generators=("best_objective", "count"),
            best_objective_max_over_gens=("best_objective", "max"),
            best_objective_std_over_gens=("best_objective", "std"),
        )
    )
    return g


def build_dataset_level(seed_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, sampler): mean/std/max across configured seeds."""
    rows: list[dict[str, Any]] = []
    for (ds, samp), g in seed_df.groupby(["dataset", "sampler"]):
        g = g.sort_values("seed")
        vals = {
            int(r.seed): float(r.best_objective) for r in g.itertuples(index=False)
        }
        arr = np.array([vals.get(s, np.nan) for s in SEEDS], dtype=float)
        row: dict[str, Any] = {
            "dataset": ds,
            "sampler": samp,
            "algorithm": SAMPLER_LABEL[samp],
            "task": (
                "classification"
                if ds in CLASSIFICATION_DATASETS
                else "regression"
            ),
            "n_seeds": int(np.isfinite(arr).sum()),
            "mean_best_j": float(np.nanmean(arr)),
            "std_best_j": float(np.nanstd(arr, ddof=1))
            if np.isfinite(arr).sum() > 1
            else 0.0,
            "max_best_j": float(np.nanmax(arr)),
            "min_best_j": float(np.nanmin(arr)),
            "cv_best_j": (
                float(np.nanstd(arr, ddof=1) / np.nanmean(arr))
                if np.nanmean(arr) not in (0, np.nan) and np.isfinite(arr).sum() > 1
                else np.nan
            ),
        }
        for s in SEEDS:
            row[SEED_LABELS[s]] = vals.get(s)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["dataset"] = pd.Categorical(out["dataset"], categories=ALL_DATASETS, ordered=True)
    out["sampler"] = pd.Categorical(out["sampler"], categories=SAMPLERS, ordered=True)
    return out.sort_values(["dataset", "sampler"]).reset_index(drop=True)


def descriptive_stats(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Overall descriptive stats + wins + average rank per algorithm."""
    # ranks within dataset (1 = best / highest mean_best_j)
    ranks = []
    for ds, g in dataset_df.groupby("dataset"):
        gg = g.copy()
        gg["rank"] = gg["mean_best_j"].rank(ascending=False, method="average")
        ranks.append(gg)
    ranked = pd.concat(ranks, ignore_index=True)

    rows = []
    for samp in SAMPLERS:
        vals = dataset_df.loc[
            dataset_df["sampler"] == samp, "mean_best_j"
        ].astype(float)
        r = ranked.loc[ranked["sampler"] == samp, "rank"].astype(float)
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        wins = int((r == 1.0).sum())  # exclusive wins only if rank exactly 1
        # count datasets where this algo has the (tied-)best mean
        best_mask = []
        for ds, g in dataset_df.groupby("dataset"):
            m = g["mean_best_j"].max()
            best_mask.append(
                float(g.loc[g["sampler"] == samp, "mean_best_j"].iloc[0]) == float(m)
            )
        n_wins_incl_ties = int(sum(best_mask))
        rows.append(
            {
                "algorithm": SAMPLER_LABEL[samp],
                "sampler": samp,
                "n_datasets": int(len(vals)),
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)),
                "median": float(vals.median()),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "iqr": float(q3 - q1),
                "q1": float(q1),
                "q3": float(q3),
                "average_rank": float(r.mean()),
                "n_wins_exclusive_rank1": wins,
                "n_wins_including_ties": n_wins_incl_ties,
            }
        )
    return pd.DataFrame(rows)


def friedman_test(dataset_df: pd.DataFrame) -> dict[str, Any]:
    """Friedman test on 15 datasets × 3 algorithms (dataset-level means)."""
    pivot = dataset_df.pivot(
        index="dataset", columns="sampler", values="mean_best_j"
    )[SAMPLERS]
    # drop incomplete blocks
    pivot = pivot.dropna()
    n = len(pivot)
    k = len(SAMPLERS)
    arrays = [pivot[s].to_numpy(dtype=float) for s in SAMPLERS]
    stat, p = stats.friedmanchisquare(*arrays)
    # Kendall's W from Friedman χ²: W = χ² / (n * (k-1))
    w = float(stat / (n * (k - 1))) if n and k > 1 else np.nan
    return {
        "n_datasets": n,
        "n_algorithms": k,
        "friedman_chi2": float(stat),
        "df": k - 1,
        "p_value": float(p),
        "kendalls_w": w,
        "alpha": 0.05,
        "significant_at_0.05": bool(p < 0.05),
        "note": (
            f"Primary unit = dataset-level mean of {len(SEEDS)} seed means "
            "(each seed mean = average best-J across 7 generators)."
        ),
    }


def pairwise_wilcoxon(dataset_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("standard_tpe", "multivariate_tpe", "Standard TPE vs Multivariate TPE"),
        ("standard_tpe", "gp", "Standard TPE vs Gaussian Process"),
        ("multivariate_tpe", "gp", "Multivariate TPE vs Gaussian Process"),
    ]
    pivot = dataset_df.pivot(
        index="dataset", columns="sampler", values="mean_best_j"
    )[SAMPLERS].dropna()
    raw_rows = []
    raw_p = []
    for a, b, label in pairs:
        xa = pivot[a].to_numpy(dtype=float)
        xb = pivot[b].to_numpy(dtype=float)
        diff = xa - xb  # positive => a better (maximisation)
        if np.allclose(diff, 0):
            w_stat, w_p = 0.0, 1.0
        else:
            # zero_method='wilcox' drops zero differences
            w_stat, w_p = stats.wilcoxon(
                xa, xb, zero_method="wilcox", alternative="two-sided"
            )
        raw_p.append(float(w_p))
        raw_rows.append(
            {
                "comparison": label,
                "algo_a": SAMPLER_LABEL[a],
                "algo_b": SAMPLER_LABEL[b],
                "mean_a": float(xa.mean()),
                "mean_b": float(xb.mean()),
                "mean_delta_a_minus_b": float(diff.mean()),
                "n_a_better": int((diff > 0).sum()),
                "n_b_better": int((diff < 0).sum()),
                "n_ties": int((diff == 0).sum()),
                "wilcoxon_stat": float(w_stat),
                "p_raw": float(w_p),
            }
        )
    reject, p_holm, _, _ = multipletests(raw_p, method="holm", alpha=0.05)
    for i, row in enumerate(raw_rows):
        row["p_holm"] = float(p_holm[i])
        row["significant_holm_0.05"] = bool(reject[i])
        d = row["mean_delta_a_minus_b"]
        if abs(d) < 1e-15:
            direction = "no difference"
        elif d > 0:
            direction = f"{row['algo_a']} higher mean (numerically)"
        else:
            direction = f"{row['algo_b']} higher mean (numerically)"
        row["direction"] = direction
    return pd.DataFrame(raw_rows)


def average_ranks_table(dataset_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, g in dataset_df.groupby("dataset"):
        gg = g.copy()
        gg["rank"] = gg["mean_best_j"].rank(ascending=False, method="average")
        for _, r in gg.iterrows():
            rows.append(
                {
                    "dataset": ds,
                    "algorithm": r["algorithm"],
                    "sampler": r["sampler"],
                    "mean_best_j": r["mean_best_j"],
                    "rank": r["rank"],
                }
            )
    per = pd.DataFrame(rows)
    avg = (
        per.groupby(["sampler", "algorithm"], as_index=False)["rank"]
        .mean()
        .rename(columns={"rank": "average_rank"})
        .sort_values("average_rank")
    )
    return per, avg


def seed_consistency(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Per algorithm: mean of per-dataset seed std / CV; plus per-dataset rows already exist."""
    rows = []
    for samp in SAMPLERS:
        sub = dataset_df[dataset_df["sampler"] == samp]
        rows.append(
            {
                "algorithm": SAMPLER_LABEL[samp],
                "sampler": samp,
                "mean_of_dataset_seed_means": float(sub["mean_best_j"].mean()),
                "mean_of_dataset_seed_stds": float(sub["std_best_j"].mean()),
                "median_of_dataset_seed_stds": float(sub["std_best_j"].median()),
                "mean_cv": float(sub["cv_best_j"].mean(skipna=True)),
                "median_cv": float(sub["cv_best_j"].median(skipna=True)),
                "n_datasets": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


# -------------------- plots --------------------


def plot_boxplot(dataset_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    data = [
        dataset_df.loc[dataset_df["sampler"] == s, "mean_best_j"].to_numpy(dtype=float)
        for s in SAMPLERS
    ]
    bp = ax.boxplot(
        data,
        tick_labels=[SAMPLER_LABEL[s] for s in SAMPLERS],
        patch_artist=True,
        showmeans=True,
    )
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel(f"Dataset-level mean best $J$ ({len(SEEDS)} seeds)")
    ax.set_title("Objective performance by optimisation algorithm")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"boxplot_algorithms.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_avg_ranks(avg_ranks: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    order = [SAMPLER_LABEL[s] for s in SAMPLERS]
    vals = [
        float(avg_ranks.loc[avg_ranks["algorithm"] == lab, "average_rank"].iloc[0])
        for lab in order
    ]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    ax.barh(order, vals, color=colors, alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel("Average rank (1 = best)")
    ax.set_title("Average rank across 15 datasets")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v + 0.02, i, f"{v:.3f}", va="center")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"average_ranks.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(dataset_df: pd.DataFrame, out: Path) -> None:
    pivot = dataset_df.pivot(
        index="dataset", columns="algorithm", values="mean_best_j"
    )
    pivot = pivot[[SAMPLER_LABEL[s] for s in SAMPLERS]]
    pivot = pivot.reindex(ALL_DATASETS)
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(f"Mean best $J$ ({len(SEEDS)} seeds) by dataset and algorithm")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean best $J$")
    # annotate
    arr = pivot.to_numpy(dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(
                j,
                i,
                f"{arr[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if arr[i, j] < np.nanmean(arr) else "black",
                fontsize=7,
            )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"heatmap_dataset_algorithm.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_dataset_winners(dataset_df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(len(ALL_DATASETS))
    width = 0.25
    colors = {"standard_tpe": "#4C72B0", "multivariate_tpe": "#55A868", "gp": "#C44E52"}
    for i, samp in enumerate(SAMPLERS):
        vals = [
            float(
                dataset_df.loc[
                    (dataset_df["dataset"] == ds) & (dataset_df["sampler"] == samp),
                    "mean_best_j",
                ].iloc[0]
            )
            for ds in ALL_DATASETS
        ]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width,
            label=SAMPLER_LABEL[samp],
            color=colors[samp],
            alpha=0.9,
        )
    # mark winners
    for xi, ds in enumerate(ALL_DATASETS):
        sub = dataset_df[dataset_df["dataset"] == ds]
        best = sub.loc[sub["mean_best_j"].idxmax()]
        ax.scatter(
            [xi + (SAMPLERS.index(best["sampler"]) - 1) * width],
            [best["mean_best_j"]],
            marker="D",
            s=28,
            color="black",
            zorder=5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_DATASETS, rotation=45, ha="right")
    ax.set_ylabel(f"Mean best $J$ ({len(SEEDS)} seeds)")
    ax.set_title("Dataset-wise algorithm comparison (* = best on dataset)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"dataset_wise_comparison.{ext}", bbox_inches="tight")
    plt.close(fig)


def plot_critical_difference(
    avg_ranks: pd.DataFrame, friedman: dict[str, Any], out: Path
) -> None:
    """Simple CD-style rank plot (Nemenyi critical difference if Friedman significant)."""
    n = friedman["n_datasets"]
    k = friedman["n_algorithms"]
    # Nemenyi CD at α=0.05
    # q_alpha for k=3, α=0.05 is approximately 2.343 (studentized range / sqrt(2))
    q_alpha = 2.343
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    order = avg_ranks.sort_values("average_rank")
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(0, 1)
    ax.axhline(0.7, color="black", lw=1)
    for _, r in order.iterrows():
        ax.plot(r["average_rank"], 0.7, "o", markersize=8)
        ax.text(
            r["average_rank"],
            0.45,
            r["algorithm"],
            ha="center",
            va="top",
            fontsize=9,
        )
        ax.text(
            r["average_rank"],
            0.82,
            f'{r["average_rank"]:.3f}',
            ha="center",
            fontsize=8,
        )
    # CD bar from best
    best = float(order["average_rank"].iloc[0])
    ax.plot([best, best + cd], [0.15, 0.15], color="#C44E52", lw=3, solid_capstyle="butt")
    ax.text(
        best + cd / 2,
        0.05,
        f"CD = {cd:.3f} (Nemenyi α=0.05)",
        ha="center",
        fontsize=8,
        color="#C44E52",
    )
    ax.set_xlabel("Average rank (lower is better)")
    ax.set_title(
        "Critical difference diagram"
        + (
            " — Friedman significant"
            if friedman["significant_at_0.05"]
            else " — Friedman not significant (CD illustrative)"
        )
    )
    ax.set_yticks([])
    ax.invert_xaxis()  # lower rank (better) on the right visually common; keep left=better
    ax.invert_xaxis()  # undo → lower rank on left
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"critical_difference.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_report(
    *,
    notes: list[str],
    desc: pd.DataFrame,
    friedman: dict[str, Any],
    pairwise: pd.DataFrame,
    avg_ranks: pd.DataFrame,
    seed_cons: pd.DataFrame,
    dataset_df: pd.DataFrame,
) -> str:
    # winners
    best_mean = desc.sort_values("mean", ascending=False).iloc[0]
    best_rank = avg_ranks.sort_values("average_rank").iloc[0]
    most_consistent = seed_cons.sort_values("mean_of_dataset_seed_stds").iloc[0]

    # GP wins
    gp_wins = []
    mt_over_st = []
    for ds in ALL_DATASETS:
        sub = dataset_df[dataset_df["dataset"] == ds].set_index("sampler")
        means = {s: float(sub.loc[s, "mean_best_j"]) for s in SAMPLERS}
        if means["gp"] > means["standard_tpe"] and means["gp"] > means["multivariate_tpe"]:
            gp_wins.append(ds)
        if means["multivariate_tpe"] > means["standard_tpe"]:
            mt_over_st.append(ds)

    n_seed = len(SEEDS)
    expected = len(ALL_DATASETS) * len(GENERATORS_NO_FOREST) * len(SAMPLERS) * n_seed
    lines = []
    lines.append(
        f"# Statistical analysis report — {n_seed} seeds ({_seed_list_str()})\n"
    )
    lines.append("## Data provenance\n")
    lines.append(
        "- Source (read-only): `results/multiseed_20261027/studies/**/summary.json`\n"
        f"- Curated export: `{BATCH_DIR.relative_to(_ROOT)}/`\n"
        "- Algorithms: Standard TPE, Multivariate TPE, Gaussian Process\n"
        f"- Seeds: {_seed_list_str()} (paired)\n"
        "- Datasets: 15 (9 classification + 6 regression)\n"
        "- Generators per dataset: 7 (forest_diffusion excluded)\n"
        f"- Expected studies: {expected} (= 15×7×3×{n_seed})\n"
        "- Objective: maximise "
        r"$J = 0.4\cdot\mathrm{Fidelity} + 0.4\cdot\mathrm{Utility} - 0.2\cdot\mathrm{Privacy\ risk}$"
        "\n"
    )
    lines.append("### Aggregation definition\n")
    lines.append(
        "For each (dataset, algorithm, seed), the seed-level score is the **mean** "
        "of the seven generator-specific study best objectives. "
        f"The dataset-level score is then the **mean of the {n_seed} seed-level scores**. "
        "Primary comparisons use this dataset-level mean (not the max across seeds).\n"
    )
    lines.append("### Validation notes\n")
    for n in notes:
        lines.append(f"- {n}\n")

    lines.append("\n## Descriptive statistics\n")
    lines.append(desc.to_markdown(index=False))
    lines.append("\n")

    lines.append("\n## Friedman test\n")
    lines.append(
        f"- χ² = {friedman['friedman_chi2']:.4f}, df = {friedman['df']}, "
        f"p = {friedman['p_value']:.6g}\n"
        f"- Kendall's W = {friedman['kendalls_w']:.4f}\n"
        f"- Significant at α=0.05: **{friedman['significant_at_0.05']}**\n"
        f"- Note: {friedman['note']}\n"
    )

    lines.append("\n## Pairwise Wilcoxon signed-rank (Holm-corrected)\n")
    lines.append(pairwise.to_markdown(index=False))
    lines.append("\n")

    lines.append("\n## Average ranks\n")
    lines.append(avg_ranks.to_markdown(index=False))
    lines.append("\n")

    lines.append("\n## Seed-level consistency (lower seed-std = more consistent)\n")
    lines.append(seed_cons.to_markdown(index=False))
    lines.append("\n")

    lines.append("\n## Research interpretation\n")
    lines.append(
        f"- Highest mean objective: **{best_mean['algorithm']}** "
        f"(mean = {best_mean['mean']:.4f}).\n"
        f"- Lowest average rank: **{best_rank['algorithm']}** "
        f"(average rank = {best_rank['average_rank']:.3f}).\n"
        f"- Friedman significant: **{friedman['significant_at_0.05']}** "
        f"(p = {friedman['p_value']:.4g}).\n"
    )
    sig_pairs = pairwise[pairwise["significant_holm_0.05"]]
    if len(sig_pairs):
        lines.append("- Significant pairwise comparisons after Holm correction:\n")
        for _, r in sig_pairs.iterrows():
            lines.append(
                f"  - {r['comparison']}: p_Holm = {r['p_holm']:.4g}; {r['direction']}\n"
            )
    else:
        lines.append(
            "- **No pairwise comparison remains significant after Holm correction.** "
            "Numerical differences should not be interpreted as statistically confirmed superiority.\n"
        )
    lines.append(
        f"- Most consistent across seeds (lowest mean per-dataset seed std): "
        f"**{most_consistent['algorithm']}** "
        f"(mean seed-std = {most_consistent['mean_of_dataset_seed_stds']:.4f}).\n"
        f"- Datasets where GP numerically beats both TPE variants: "
        f"{', '.join(gp_wins) if gp_wins else 'none'}.\n"
        f"- Datasets where Multivariate TPE numerically beats Standard TPE: "
        f"{len(mt_over_st)}/{len(ALL_DATASETS)} "
        f"({', '.join(mt_over_st)}).\n"
    )

    lines.append("\n## Limitations\n")
    lines.append(
        f"- **{n_seed}** seeds improve variability estimates relative to a 3-seed pilot, "
        "but Wilcoxon/Friedman power on 15 datasets remains modest.\n"
        "- All reported studies target 20 completed trials per (dataset, generator, sampler, seed).\n"
        "- Dataset-level scores average across seven generators; generator-specific "
        "behaviour may differ.\n"
        "- Multiple testing corrected with Holm; absence of significance is not evidence of equivalence.\n"
    )

    lines.append("\n## Results and Discussion (paper-ready draft)\n")
    lines.append(_paper_section(desc, friedman, pairwise, avg_ranks, seed_cons, gp_wins, mt_over_st, best_mean, best_rank))
    return "".join(lines)


def _paper_section(
    desc, friedman, pairwise, avg_ranks, seed_cons, gp_wins, mt_over_st, best_mean, best_rank
) -> str:
    st = desc[desc["sampler"] == "standard_tpe"].iloc[0]
    mt = desc[desc["sampler"] == "multivariate_tpe"].iloc[0]
    gp = desc[desc["sampler"] == "gp"].iloc[0]
    n_seed = len(SEEDS)
    pairwise_clause = (
        "did not identify any significant pairwise differences"
        if not pairwise["significant_holm_0.05"].any()
        else "identified significant pairwise differences as tabulated"
    )
    claim_clause = (
        "does **not** support claiming statistically significant superiority of any single optimiser over the others"
        if not pairwise["significant_holm_0.05"].any()
        else "supports pairwise claims only where Holm-adjusted tests are significant"
    )
    holm_tail = (
        "after Holm correction no pairwise contrast reaches significance, so omnibus evidence of a difference among methods should not be over-interpreted as confirming a uniquely superior optimiser"
        if not pairwise["significant_holm_0.05"].any()
        else "Holm-adjusted pairwise tests should be read jointly with the omnibus Friedman result"
    )
    return f"""
We compared three Bayesian optimisation algorithms—Standard TPE, Multivariate TPE and Gaussian Process (GP) sampling—on fifteen tabular datasets (nine classification and six regression). Each algorithm was executed with {n_seed} paired random seeds ({_seed_list_str()}). For every dataset–algorithm–seed combination, performance was summarised as the mean of the best objective values obtained across seven synthetic-data generators; dataset-level scores were then obtained by averaging the {n_seed} seed-level summaries. The composite objective maximises fidelity and downstream utility while penalising privacy risk.

Descriptively, {best_mean['algorithm']} attained the highest mean dataset-level objective ({best_mean['mean']:.4f}), followed by the remaining methods (Standard TPE: {st['mean']:.4f}; Multivariate TPE: {mt['mean']:.4f}; Gaussian Process: {gp['mean']:.4f}). Average ranks across datasets likewise favoured {best_rank['algorithm']} (average rank {best_rank['average_rank']:.3f}). These numerical patterns indicate a modest ordering among samplers but must be interpreted jointly with formal hypothesis tests.

A Friedman test on the fifteen dataset-level means did {'not ' if not friedman['significant_at_0.05'] else ''}detect a statistically significant difference among the three algorithms at α = 0.05 (χ² = {friedman['friedman_chi2']:.3f}, df = {friedman['df']}, p = {friedman['p_value']:.4g}; Kendall's W = {friedman['kendalls_w']:.3f}). Pairwise Wilcoxon signed-rank tests with Holm correction {pairwise_clause}. Accordingly, while one method may appear numerically preferable on mean objective or average rank, the present evidence {claim_clause} under the primary analysis protocol.

Seed-level consistency was summarised by the mean of per-dataset standard deviations across the {n_seed} seeds. {seed_cons.sort_values('mean_of_dataset_seed_stds').iloc[0]['algorithm']} exhibited the smallest average seed dispersion. Lower variability did not necessarily coincide with the highest mean performance, underscoring that stability and peak objective are distinct properties in this setting.

Dataset-wise inspection showed that Gaussian Process numerically outperformed both TPE variants on {len(gp_wins)} dataset(s){(': ' + ', '.join(gp_wins)) if gp_wins else ''}. Multivariate TPE exceeded Standard TPE on {len(mt_over_st)} of 15 datasets. These local wins illustrate heterogeneity across data regimes; however, {holm_tail}.

Several limitations apply. With {n_seed} seeds, variability estimates improve relative to a three-seed pilot, but power of rank-based tests on fifteen datasets remains limited. All studies target a 20-trial budget. Finally, averaging across generators emphasises overall behaviour and may mask generator-specific optimiser advantages. Generator-stratified analyses would further strengthen inferential claims.

"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Paired seeds to include (default: all six)",
    )
    p.add_argument(
        "--batch-dir",
        type=str,
        default=str(BATCH_DIR),
        help="Curated export directory",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(OUT_DIR),
        help="Statistical analysis output directory",
    )
    args = p.parse_args(argv)
    _configure_run(
        seeds=list(args.seeds),
        batch_dir=Path(args.batch_dir),
        out_dir=Path(args.out_dir),
    )

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    raw, missing = load_study_summaries()
    notes = validate(raw, missing)

    seed_df = build_seed_level(raw)
    dataset_df = build_dataset_level(seed_df)

    n_seed = len(SEEDS)
    incomplete = dataset_df[dataset_df["n_seeds"] < n_seed]
    if len(incomplete):
        notes.append(
            f"WARNING: {len(incomplete)} dataset–algorithm rows have <{n_seed} seeds"
        )
    else:
        notes.append(
            f"All 15 datasets × 3 algorithms have {n_seed} seed-level scores — OK for Friedman."
        )

    # ---- curated exports (do not touch originals) ----
    raw.to_csv(BATCH_DIR / "study_level_best_j.csv", index=False)
    seed_df.to_csv(BATCH_DIR / "seed_level_dataset_mean_over_generators.csv", index=False)
    dataset_df.to_csv(BATCH_DIR / "dataset_level_mean_std_max.csv", index=False)
    wide = dataset_df.pivot(index="dataset", columns="algorithm", values="mean_best_j")
    wide.to_csv(BATCH_DIR / "dataset_algorithm_mean_best_j_wide.csv")
    (BATCH_DIR / "README.md").write_text(
        f"# Multi-seed comparison — seeds {_seed_list_str()}\n\n"
        "Curated exports from `results/multiseed_20261027/` (originals unchanged).\n\n"
        "- `study_level_best_j.csv` — one row per (dataset, generator, sampler, seed)\n"
        "- `seed_level_dataset_mean_over_generators.csv` — mean over 7 generators\n"
        f"- `dataset_level_mean_std_max.csv` — mean/std/max across {n_seed} seeds\n"
        "- Analysis outputs: `statistical_analysis/`\n"
        "- Mean±std generator tables: `tables/`\n",
        encoding="utf-8",
    )

    # ---- stats ----
    desc = descriptive_stats(dataset_df)
    friedman = friedman_test(dataset_df)
    pairwise = pairwise_wilcoxon(dataset_df)
    per_ranks, avg_ranks = average_ranks_table(dataset_df)
    seed_cons = seed_consistency(dataset_df)

    desc.to_csv(OUT_DIR / "descriptive_statistics.csv", index=False)
    pd.DataFrame([friedman]).to_csv(OUT_DIR / "friedman_test_results.csv", index=False)
    pairwise.to_csv(OUT_DIR / "pairwise_wilcoxon_holm.csv", index=False)
    avg_ranks.to_csv(OUT_DIR / "average_ranks.csv", index=False)
    per_ranks.to_csv(OUT_DIR / "average_ranks_by_dataset.csv", index=False)
    dataset_df.to_csv(OUT_DIR / "dataset_wise_results.csv", index=False)
    seed_cons.to_csv(OUT_DIR / "seed_consistency.csv", index=False)
    seed_cols = [SEED_LABELS[s] for s in SEEDS]
    dataset_df[
        ["dataset", "algorithm", *seed_cols, "mean_best_j", "std_best_j", "cv_best_j", "max_best_j"]
    ].to_csv(OUT_DIR / "seed_consistency_by_dataset.csv", index=False)

    # ---- figures ----
    plot_boxplot(dataset_df, FIG_DIR)
    plot_avg_ranks(avg_ranks, FIG_DIR)
    plot_heatmap(dataset_df, FIG_DIR)
    plot_dataset_winners(dataset_df, FIG_DIR)
    plot_critical_difference(avg_ranks, friedman, FIG_DIR)

    report = write_report(
        notes=notes,
        desc=desc,
        friedman=friedman,
        pairwise=pairwise,
        avg_ranks=avg_ranks,
        seed_cons=seed_cons,
        dataset_df=dataset_df,
    )
    (OUT_DIR / "statistical_analysis_report.md").write_text(report, encoding="utf-8")

    # Compact console summary matching the batch-1 "Done" card
    best_mean = desc.sort_values("mean", ascending=False).iloc[0]
    most_consistent = seed_cons.sort_values("mean_of_dataset_seed_stds").iloc[0]
    print(report)
    print("\n### Done\n")
    print(
        f"Verified **{len(raw)}/{len(ALL_DATASETS)*len(GENERATORS_NO_FOREST)*len(SAMPLERS)*n_seed}** "
        f"studies (seeds **{_seed_list_str().replace(', ', ', ')}** × 3 algorithms × 15×7). "
        "Originals untouched.\n"
    )
    print(f"**Curated folder:** `{BATCH_DIR.relative_to(_ROOT)}/`")
    print(f"**Analysis:** `{OUT_DIR.relative_to(_ROOT)}/` (+ figures/ at 300 DPI)\n")
    print(
        f"### Key findings (dataset-level = mean over 7 gens, then mean of {n_seed} seeds)\n"
    )
    print("| Algorithm | Mean $J$ | Avg rank | Dataset wins |")
    print("|-----------|------------|----------|--------------|")
    for _, r in desc.sort_values("average_rank").iterrows():
        wins = int(r["n_wins_including_ties"])
        if r["algorithm"] == best_mean["algorithm"]:
            print(
                f"| **{r['algorithm']}** | **{r['mean']:.4f}** | "
                f"**{r['average_rank']:.2f}** | **{wins}/15** |"
            )
        else:
            print(
                f"| {r['algorithm']} | {r['mean']:.4f} | "
                f"{r['average_rank']:.2f} | {wins}/15 |"
            )
    print(
        f"\n**Friedman:** χ²={friedman['friedman_chi2']:.2f}, df={friedman['df']}, "
        f"**p={friedman['p_value']:.3g}** "
        f"({'significant' if friedman['significant_at_0.05'] else 'not significant'}); "
        f"Kendall’s W={friedman['kendalls_w']:.2f}"
    )
    gp_st = pairwise[
        pairwise["comparison"] == "Standard TPE vs Gaussian Process"
    ].iloc[0]
    gp_mt = pairwise[
        pairwise["comparison"] == "Multivariate TPE vs Gaussian Process"
    ].iloc[0]
    st_mt = pairwise[
        pairwise["comparison"] == "Standard TPE vs Multivariate TPE"
    ].iloc[0]
    if pairwise["significant_holm_0.05"].any():
        print(
            "**Wilcoxon + Holm:** GP beats ST "
            f"(p_Holm={gp_st['p_holm']:.3f}) and MT "
            f"(p_Holm={gp_mt['p_holm']:.3f}); ST vs MT not significant "
            f"(p_Holm={st_mt['p_holm']:.3f})"
            if gp_st["significant_holm_0.05"] and gp_mt["significant_holm_0.05"]
            else "**Wilcoxon + Holm:** significant pairwise contrasts present (see report)"
        )
    else:
        print(
            "**Wilcoxon + Holm:** **no pairwise contrast significant** "
            f"(GP vs ST p_Holm≈{gp_st['p_holm']:.3f}; GP vs MT p_Holm≈{gp_mt['p_holm']:.3f})"
        )
    print(
        f"**Most seed-stable:** {most_consistent['algorithm']} "
        f"(lowest mean seed-std = {most_consistent['mean_of_dataset_seed_stds']:.4f})"
    )
    print(f"\nBatch exports → {BATCH_DIR}")
    print(f"Analysis outputs → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
