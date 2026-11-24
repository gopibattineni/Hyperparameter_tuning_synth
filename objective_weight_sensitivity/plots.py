"""Publication-style plots for objective-weight sensitivity (compare F/U/P, not raw J)."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

METRIC_COLS = {
    "Fidelity (SDV)": "best_fidelity",
    "Utility": "best_utility",
    "Privacy risk": "best_privacy_risk",
}


def _load_comparison(results_root: Path) -> pd.DataFrame:
    path = Path(results_root) / "objective_weight_comparison.csv"
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _config_order(df: pd.DataFrame) -> list[str]:
    preferred = ["O1", "O2", "O3", "O4", "O5"]
    present = [c for c in preferred if c in set(df["config"].astype(str))]
    extras = sorted(c for c in df["config"].astype(str).unique() if c not in present)
    return present + extras


def plot_grouped_fup(df: pd.DataFrame, out: Path) -> Path | None:
    if df.empty:
        return None
    # Mean across dataset×generator for each config (pilot may be small)
    configs = _config_order(df)
    means = {
        label: [float(df.loc[df["config"] == c, col].mean()) for c in configs]
        for label, col in METRIC_COLS.items()
        if col in df.columns
    }
    if not means:
        return None
    x = np.arange(len(configs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, (label, vals) in enumerate(means.items()):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Score ([0, 1])")
    ax.set_ylim(0, 1.15)
    ax.set_title("Fidelity, utility and privacy risk by objective-weight config")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_weight_comparison(df: pd.DataFrame, out: Path) -> Path | None:
    if df.empty or not {"w_fidelity", "w_utility", "w_privacy_risk"}.issubset(df.columns):
        return None
    configs = _config_order(df)
    # One weight row per config
    rows = []
    for c in configs:
        sub = df.loc[df["config"] == c].iloc[0]
        rows.append(
            {
                "config": c,
                "wF": float(sub["w_fidelity"]),
                "wU": float(sub["w_utility"]),
                "wP": float(sub["w_privacy_risk"]),
            }
        )
    wdf = pd.DataFrame(rows)
    x = np.arange(len(configs))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (col, label) in enumerate(
        [("wF", "Weight fidelity"), ("wU", "Weight utility"), ("wP", "Weight privacy")]
    ):
        bars = ax.bar(x + (i - 1) * width, wdf[col], width, label=label)
        for bar, v in zip(bars, wdf[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 0.75)
    ax.set_title("Objective weight configurations (O1–O5)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_fidelity_vs_utility(df: pd.DataFrame, out: Path) -> Path | None:
    if df.empty or not {"best_fidelity", "best_utility", "config"}.issubset(df.columns):
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5))
    configs = _config_order(df)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(configs):
        sub = df.loc[df["config"] == c]
        ax.scatter(
            sub["best_fidelity"],
            sub["best_utility"],
            label=c,
            s=55,
            alpha=0.85,
            color=cmap(i % 10),
        )
    ax.set_xlabel("Fidelity (SDV)")
    ax.set_ylabel("Utility")
    ax.set_title("Fidelity vs utility (by weight config)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, title="Config")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_utility_vs_privacy(df: pd.DataFrame, out: Path) -> Path | None:
    if df.empty or not {"best_utility", "best_privacy_risk", "config"}.issubset(df.columns):
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5))
    configs = _config_order(df)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(configs):
        sub = df.loc[df["config"] == c]
        ax.scatter(
            sub["best_utility"],
            sub["best_privacy_risk"],
            label=c,
            s=55,
            alpha=0.85,
            color=cmap(i % 10),
        )
    ax.set_xlabel("Utility")
    ax.set_ylabel("Privacy risk (higher = worse)")
    ax.set_title("Utility vs privacy risk (by weight config)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, title="Config")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_all_plots(results_root: Path) -> list[Path]:
    root = Path(results_root)
    fig_dir = root / "figures"
    df = _load_comparison(root)
    written: list[Path] = []
    for fn, name in [
        (plot_grouped_fup, "grouped_fidelity_utility_privacy.png"),
        (plot_weight_comparison, "objective_weight_configs.png"),
        (plot_fidelity_vs_utility, "fidelity_vs_utility.png"),
        (plot_utility_vs_privacy, "utility_vs_privacy.png"),
    ]:
        path = fn(df, fig_dir / name)
        if path is not None:
            written.append(path)
    return written
