#!/usr/bin/env python3
"""Publication figures from existing Optuna regression HPO studies.

Reads only on-disk artifacts under ``results/regression/{n}_{dataset}/{generator}/``:
  - study.pkl
  - completed.json
  - trials.csv (fallback)

Does **not** rerun Optuna. Each dataset has 6 independent studies (one per
generator). Figure design:

  Fig 1  Optimization history — 6 dataset panels; COMPLETE trials for all
         generators overlaid; running-best dashed; overall best annotated.
  Fig 2  Hyperparameter importance — 6 panels; Fanova importance from the
         **best generator study** on that dataset (highest completed
         best_value). Parameters that cannot be scored are omitted.
  Fig 3  Best objective comparison — one bar per dataset (max over generators).
  Supp  Parallel-coordinate plots — best-generator study per dataset.

Outputs (PNG 300 dpi + PDF) under::

  results/regression/figures_manuscript/
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import optuna
import pandas as pd
from optuna.importance import get_param_importances
from optuna.trial import TrialState

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
REG_ROOT = ROOT / "results" / "regression"
OUT_DIR = REG_ROOT / "figures_manuscript"

DATASETS: list[tuple[str, str, str]] = [
    # (folder, key, display name)
    ("10_metro_interstate", "metro_interstate", "Metro Interstate"),
    ("11_online_shopping", "online_shopping", "Online Shopping"),
    ("12_air_quality", "air_quality", "Air Quality"),
    ("13_concrete", "concrete", "Concrete"),
    ("14_energy_efficiency", "energy_efficiency", "Energy Efficiency"),
    ("15_real_estate", "real_estate", "Real Estate"),
]

GENERATORS: list[tuple[str, str]] = [
    ("gaussian_copula", "GaussianCopula"),
    ("copulagan", "CopulaGAN"),
    ("ctgan", "CTGAN"),
    ("tvae", "TVAE"),
    ("ctabgan", "CTAB-GAN+"),
    ("wgan_gp", "WGAN-GP"),
    ("tabddpm", "TabDDPM"),
]

# Colorblind-friendly (Wong / Tol-inspired)
GEN_COLORS = {
    "gaussian_copula": "#0072B2",
    "copulagan": "#E69F00",
    "ctgan": "#009E73",
    "tvae": "#CC79A7",
    "ctabgan": "#D55E00",
    "wgan_gp": "#56B4E9",
    "tabddpm": "#999999",
}


def _apply_style() -> None:
    # Prefer Times New Roman; Liberation Serif is the metric-compatible substitute on Linux.
    preferred = [
        "Times New Roman",
        "Times",
        "Liberation Serif",
        "Nimbus Roman",
        "DejaVu Serif",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    serif = [n for n in preferred if n in available] or preferred
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": serif,
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,  # embed TrueType for editable text in PDF
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
        }
    )


def load_study(folder: str, gen_key: str) -> optuna.Study | None:
    path = REG_ROOT / folder / gen_key / "study.pkl"
    if not path.is_file():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_completed(folder: str, gen_key: str) -> dict[str, Any] | None:
    path = REG_ROOT / folder / gen_key / "completed.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def complete_trials(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    return [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]


def running_best(values: list[float], maximize: bool = True) -> list[float]:
    best: list[float] = []
    cur = -np.inf if maximize else np.inf
    for v in values:
        if maximize:
            cur = max(cur, v)
        else:
            cur = min(cur, v)
        best.append(cur)
    return best


def best_generator_for_dataset(folder: str) -> tuple[str, str, float, int] | None:
    """Return (gen_key, gen_display, best_value, best_trial_number)."""
    best: tuple[str, str, float, int] | None = None
    for gen_key, gen_disp in GENERATORS:
        study = load_study(folder, gen_key)
        if study is None:
            continue
        comps = complete_trials(study)
        if not comps:
            continue
        # Prefer completed.json when present (matches report ranking)
        meta = load_completed(folder, gen_key) or {}
        val = meta.get("best_value")
        if val is None:
            val = float(study.best_value)
        else:
            val = float(val)
        trial_no = int(study.best_trial.number)
        if best is None or val > best[2]:
            best = (gen_key, gen_disp, val, trial_no)
    return best


def safe_param_importances(study: optuna.Study) -> dict[str, float]:
    comps = complete_trials(study)
    if len(comps) < 3:
        return {}
    # Need at least one varying parameter
    keys = {k for t in comps for k in t.params}
    varying = []
    for k in keys:
        vals = {str(t.params.get(k)) for t in comps if k in t.params}
        if len(vals) > 1:
            varying.append(k)
    if not varying:
        return {}
    try:
        raw = get_param_importances(study)
    except Exception as exc:  # pragma: no cover — reported in summary
        print(f"  [warn] importance failed: {type(exc).__name__}: {exc}")
        return {}
    out = {k: float(v) for k, v in raw.items() if np.isfinite(v) and v >= 0}
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def save_fig(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  wrote {png.name} / {pdf.name}")


# ---------------------------------------------------------------------------
# Figure 1 — Optimization history
# ---------------------------------------------------------------------------

# Distinct markers help when colours overlap
GEN_MARKERS = {
    "gaussian_copula": "o",
    "copulagan": "s",
    "ctgan": "^",
    "tvae": "D",
    "ctabgan": "v",
    "wgan_gp": "P",
}


def figure_optimization_history() -> None:
    """Readable history: faint trial dots + solid running-best curves."""
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.8), sharex=True, sharey=False)
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []
    seen_gens: set[str] = set()

    for ax, (folder, _key, display) in zip(axes, DATASETS):
        overall_best_val = -np.inf
        overall_best_meta: tuple[str, str, int, float] | None = None
        y_all: list[float] = []

        for gen_key, gen_disp in GENERATORS:
            study = load_study(folder, gen_key)
            if study is None:
                continue
            comps = complete_trials(study)
            if not comps:
                continue
            comps = sorted(comps, key=lambda t: t.number)
            xs = np.asarray([t.number for t in comps], dtype=float)
            ys = np.asarray([float(t.value) for t in comps], dtype=float)
            y_all.extend(ys.tolist())
            color = GEN_COLORS[gen_key]
            marker = GEN_MARKERS[gen_key]
            rb = np.asarray(running_best(ys.tolist(), maximize=True), dtype=float)

            ax.scatter(
                xs,
                ys,
                s=18,
                c=color,
                marker=marker,
                alpha=0.35,
                linewidths=0,
                zorder=2,
            )
            (line,) = ax.plot(
                xs,
                rb,
                color=color,
                linewidth=2.2,
                solid_capstyle="round",
                zorder=3,
                label=gen_disp,
            )
            ax.scatter(
                [xs[-1]],
                [rb[-1]],
                s=32,
                c=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )

            if gen_key not in seen_gens:
                seen_gens.add(gen_key)
                legend_handles.append(line)
                legend_labels.append(gen_disp)

            local_best_i = int(np.argmax(ys))
            if ys[local_best_i] > overall_best_val:
                overall_best_val = float(ys[local_best_i])
                overall_best_meta = (
                    gen_key,
                    gen_disp,
                    int(xs[local_best_i]),
                    float(ys[local_best_i]),
                )

        # Dataset label inside axes (avoids clipping with shared legend)
        ax.text(
            0.02,
            0.98,
            display,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            bbox={
                "boxstyle": "square,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.85,
            },
            zorder=8,
        )

        if overall_best_meta is not None:
            _gkey, gdisp, tnum, bval = overall_best_meta
            ax.axhline(bval, color="0.4", linewidth=0.9, linestyle="--", zorder=1)
            ax.scatter(
                [tnum],
                [bval],
                s=110,
                facecolors="none",
                edgecolors="black",
                linewidths=1.8,
                zorder=6,
            )
            # Place callout on the side with more free space
            x_off = -55 if tnum > 12 else 10
            ha = "right" if tnum > 12 else "left"
            ax.annotate(
                f"best {bval:.3f}\n{gdisp} · trial {tnum}",
                xy=(tnum, bval),
                xytext=(x_off, -22),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="0.1",
                ha=ha,
                va="top",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#FFF8E7",
                    "edgecolor": "0.55",
                    "linewidth": 0.7,
                    "alpha": 0.95,
                },
                arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.8},
                zorder=7,
            )

        ax.set_xlim(-0.8, 19.8)
        if y_all:
            lo = min(y_all)
            hi = max(y_all)
            pad = max(0.05, 0.08 * (hi - lo + 1e-6))
            ymin = lo - pad
            if lo > 0.2:
                ymin = max(0.0, lo - 1.5 * pad)
            ax.set_ylim(ymin, hi + 1.4 * pad)

        ax.grid(True, axis="both", linestyle=":", linewidth=0.55, alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    for ax in axes[3:]:
        ax.set_xlabel("Trial number", fontsize=10)
    for ax in (axes[0], axes[3]):
        ax.set_ylabel("Objective value", fontsize=10)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=6,
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        framealpha=1.0,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=1.4,
        handlelength=2.6,
        fontsize=9,
    )
    fig.suptitle(
        "Optimization history (solid = running best; faint markers = trial values)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.12, wspace=0.22, hspace=0.28)
    save_fig(fig, "Fig1_optimization_history")


# ---------------------------------------------------------------------------
# Figure 2 — Hyperparameter importance (best generator study per dataset)
# ---------------------------------------------------------------------------

def _short_param_label(name: str) -> str:
    """Keep Optuna names, but wrap long identifiers for axis readability."""
    # Prefer natural breaks on underscores without inventing new names
    if len(name) <= 18:
        return name
    parts = name.split("_")
    if len(parts) == 1:
        return name
    mid = len(parts) // 2
    return "_".join(parts[:mid]) + "\n" + "_".join(parts[mid:])


def figure_param_importance(top_k: int = 5) -> None:
    """One panel per dataset; show top-k Fanova importances from best generator."""
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0), sharex=False)
    axes = axes.ravel()

    for ax, (folder, _key, display) in zip(axes, DATASETS):
        best = best_generator_for_dataset(folder)
        if best is None:
            ax.text(0.5, 0.5, f"{display}\n(no completed study)", ha="center", va="center")
            ax.axis("off")
            continue

        gen_key, gen_disp, best_val, best_trial = best
        study = load_study(folder, gen_key)
        assert study is not None
        imp = safe_param_importances(study)

        # Header block inside panel
        ax.text(
            0.0,
            1.08,
            display,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.0,
            1.01,
            f"{gen_disp}  ·  best {best_val:.3f} (trial {best_trial})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            color="0.35",
            clip_on=False,
        )

        if not imp:
            ax.text(
                0.5,
                0.5,
                "Importance not estimable",
                ha="center",
                va="center",
                fontsize=9,
                color="0.45",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            continue

        # Most → least, keep top_k for readability
        items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        n_total = len(items)
        shown = items[: max(1, min(top_k, len(items)))]
        # Draw least→most so most important sits at top of barh
        shown = list(reversed(shown))
        names = [k for k, _ in shown]
        vals = [v for _, v in shown]
        labels = [_short_param_label(n) for n in names]

        y = np.arange(len(names))
        base_color = GEN_COLORS.get(gen_key, "#4C72B0")
        colors = [base_color] * len(vals)
        # Emphasize the top parameter (last after reverse = highest importance)
        colors[-1] = base_color

        bars = ax.barh(
            y,
            vals,
            height=0.62,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.92,
            zorder=3,
        )
        # Darker outline on the most important bar
        bars[-1].set_alpha(1.0)
        bars[-1].set_edgecolor("0.15")
        bars[-1].set_linewidth(1.1)

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Fanova importance", fontsize=9)
        ax.set_xlim(0, 1.0)  # shared scale across panels for fair visual comparison

        for yi, v in zip(y, vals):
            # Value inside bar if wide enough, else just outside
            if v >= 0.22:
                ax.text(
                    v - 0.02,
                    yi,
                    f"{v:.2f}",
                    va="center",
                    ha="right",
                    fontsize=9,
                    fontweight="bold",
                    color="white",
                    zorder=4,
                )
            else:
                ax.text(
                    v + 0.02,
                    yi,
                    f"{v:.2f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                    fontweight="bold",
                    color="0.15",
                    zorder=4,
                )

        # Footnote if truncated
        if n_total > len(shown):
            omitted = n_total - len(shown)
            ax.text(
                0.98,
                0.02,
                f"top {len(shown)} of {n_total}  (−{omitted})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7,
                color="0.45",
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=8)

    fig.suptitle(
        "Hyperparameter importance by dataset\n"
        "(Fanova on the best-generator Optuna study; top parameters shown)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.07, wspace=0.55, hspace=0.45)
    save_fig(fig, "Fig2_hyperparameter_importance")


# ---------------------------------------------------------------------------
# Figure 3 — Best objective comparison across datasets
# ---------------------------------------------------------------------------

def figure_best_objective_comparison() -> None:
    rows = []
    for folder, key, display in DATASETS:
        best = best_generator_for_dataset(folder)
        if best is None:
            continue
        gen_key, gen_disp, best_val, best_trial = best
        rows.append(
            {
                "folder": folder,
                "key": key,
                "display": display,
                "generator": gen_disp,
                "gen_key": gen_key,
                "best_value": best_val,
                "best_trial": best_trial,
            }
        )
    if not rows:
        raise RuntimeError("No completed studies found for any regression dataset.")

    # Preserve manuscript dataset order (top → bottom = dataset 10 → 15)
    order = {d[2]: i for i, d in enumerate(DATASETS)}
    rows.sort(key=lambda r: order[r["display"]], reverse=True)  # barh: first drawn at bottom

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ys = np.arange(len(rows))
    vals = [r["best_value"] for r in rows]
    colors = [GEN_COLORS[r["gen_key"]] for r in rows]

    ax.barh(
        ys,
        vals,
        height=0.55,
        color=colors,
        edgecolor="white",
        linewidth=1.0,
        zorder=3,
    )

    ax.set_yticks(ys)
    ax.set_yticklabels([r["display"] for r in rows], fontsize=11)
    ax.set_xlabel("Best objective value", fontsize=11)
    # Leave clear space on the right so value labels never collide with the legend
    ax.set_xlim(0, max(vals) + 0.18)
    ax.set_ylim(-0.75, len(rows) - 0.25)

    for i, r in enumerate(rows):
        w = r["best_value"]
        ax.text(
            w + 0.01,
            ys[i],
            f"{w:.3f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="0.1",
            zorder=4,
        )

    mean_v = float(np.mean(vals))
    ax.axvline(mean_v, color="0.55", linestyle="--", linewidth=1.1, zorder=1)

    ax.set_title(
        "Best Optuna objective by regression dataset",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle=":", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)

    preferred = [g[0] for g in GENERATORS]
    used_keys = []
    for k in preferred:
        if any(r["gen_key"] == k for r in rows) and k not in used_keys:
            used_keys.append(k)
    handles = [
        mpl.patches.Patch(
            facecolor=GEN_COLORS[k],
            edgecolor="0.3",
            linewidth=0.6,
            label=next(r["generator"] for r in rows if r["gen_key"] == k),
        )
        for k in used_keys
    ]
    handles.append(
        mpl.lines.Line2D(
            [0],
            [0],
            color="0.55",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean ({mean_v:.3f})",
        )
    )
    # Top-right, outside the axes — never overlays bars
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        fontsize=9,
        title="Winning generator",
        title_fontsize=9,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.22, right=0.78, top=0.86, bottom=0.14)
    save_fig(fig, "Fig3_best_objective_comparison")

    rows_out = sorted(rows, key=lambda r: order[r["display"]])
    pd.DataFrame(rows_out).to_csv(OUT_DIR / "Fig3_best_objective_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Supplementary — parallel coordinates (best generator study)
# ---------------------------------------------------------------------------

def _encode_params_for_parallel(
    trials: list[optuna.trial.FrozenTrial],
) -> tuple[pd.DataFrame, list[str]]:
    """Build a numeric dataframe for parallel coordinates (categoricals → codes)."""
    keys = sorted({k for t in trials for k in t.params})
    rows = []
    for t in trials:
        row = {"trial": t.number, "objective": float(t.value)}
        for k in keys:
            row[k] = t.params.get(k, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    plot_cols = []
    for k in keys:
        if df[k].dtype == object or df[k].dtype.name == "bool":
            codes, _uniques = pd.factorize(df[k].astype(str), sort=True)
            df[k] = codes.astype(float)
            # skip constant
            if df[k].nunique(dropna=True) <= 1:
                continue
            plot_cols.append(k)
        else:
            if df[k].nunique(dropna=True) <= 1:
                continue
            df[k] = pd.to_numeric(df[k], errors="coerce")
            plot_cols.append(k)
    plot_cols.append("objective")
    return df, plot_cols


def figure_parallel_coordinates_supplementary() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.2))
    axes = axes.ravel()

    for ax, (folder, _key, display) in zip(axes, DATASETS):
        best = best_generator_for_dataset(folder)
        if best is None:
            ax.axis("off")
            continue
        gen_key, gen_disp, best_val, best_trial = best
        study = load_study(folder, gen_key)
        assert study is not None
        comps = complete_trials(study)
        if len(comps) < 3:
            ax.text(0.5, 0.5, "Too few trials", ha="center", va="center")
            ax.set_title(f"{display} — {gen_disp}", loc="left")
            continue

        df, cols = _encode_params_for_parallel(comps)
        if len(cols) < 2:
            ax.text(0.5, 0.5, "No varying params", ha="center", va="center")
            ax.set_title(f"{display} — {gen_disp}", loc="left")
            continue

        # Normalize each axis to [0, 1] for display
        norms = df[cols].copy()
        for c in cols:
            lo, hi = norms[c].min(), norms[c].max()
            if hi > lo:
                norms[c] = (norms[c] - lo) / (hi - lo)
            else:
                norms[c] = 0.5

        obj = norms["objective"].to_numpy()
        cmap = plt.get_cmap("viridis")
        x = np.arange(len(cols))
        idx_list = list(norms.index)
        for i, row in norms.iterrows():
            is_best = int(df.loc[i, "trial"]) == best_trial
            color = "crimson" if is_best else cmap(obj[idx_list.index(i)])
            lw = 2.6 if is_best else 0.75
            alpha = 1.0 if is_best else 0.35
            ax.plot(
                x,
                row[cols].to_numpy(),
                color=color,
                lw=lw,
                alpha=alpha,
                zorder=5 if is_best else 1,
                solid_capstyle="round",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=6.5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Normalized value", fontsize=8)
        ax.set_title(
            f"{display} — {gen_disp}\nbest trial {best_trial} (obj={best_val:.3f})",
            loc="left",
            fontsize=9,
            pad=4,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    # Explicit legend for line styles / colours
    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            color="crimson",
            linewidth=2.6,
            label="Best trial",
        ),
        mpl.lines.Line2D(
            [0],
            [0],
            color=plt.get_cmap("viridis")(0.65),
            linewidth=1.2,
            alpha=0.7,
            label="Other COMPLETE trials (colour ∝ objective)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
        columnspacing=2.0,
        handlelength=2.8,
    )
    fig.suptitle(
        "Supplementary: parallel coordinates (best-generator Optuna study per dataset)",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )
    fig.subplots_adjust(left=0.07, right=0.99, top=0.90, bottom=0.12, wspace=0.28, hspace=0.42)
    save_fig(fig, "Supp_parallel_coordinates")


# ---------------------------------------------------------------------------
# Optional: grouped best-by-generator heatmap-style bars (useful table companion)
# ---------------------------------------------------------------------------

def write_inventory_csv() -> None:
    """CSV inventory of every study used (reproducibility)."""
    rows = []
    for folder, key, display in DATASETS:
        for gen_key, gen_disp in GENERATORS:
            study = load_study(folder, gen_key)
            meta = load_completed(folder, gen_key) or {}
            if study is None:
                rows.append(
                    {
                        "dataset": key,
                        "display": display,
                        "generator": gen_key,
                        "generator_display": gen_disp,
                        "n_complete": 0,
                        "n_pruned": 0,
                        "n_fail": 0,
                        "best_value": None,
                        "best_trial": None,
                        "study_pkl": False,
                    }
                )
                continue
            n_c = sum(1 for t in study.trials if t.state == TrialState.COMPLETE)
            n_p = sum(1 for t in study.trials if t.state == TrialState.PRUNED)
            n_f = sum(1 for t in study.trials if t.state == TrialState.FAIL)
            best_v = meta.get("best_value")
            if best_v is None and n_c:
                best_v = float(study.best_value)
            best_t = int(study.best_trial.number) if n_c else None
            rows.append(
                {
                    "dataset": key,
                    "display": display,
                    "generator": gen_key,
                    "generator_display": gen_disp,
                    "n_complete": n_c,
                    "n_pruned": n_p,
                    "n_fail": n_f,
                    "best_value": best_v,
                    "best_trial": best_t,
                    "study_pkl": True,
                    "params": json.dumps(study.best_params if n_c else {}, default=str),
                }
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "study_inventory.csv", index=False)
    print("  wrote study_inventory.csv")


def main() -> None:
    _apply_style()
    print(f"Reading studies from {REG_ROOT}")
    print(f"Writing figures to {OUT_DIR}")
    write_inventory_csv()
    print("Fig1 …")
    figure_optimization_history()
    print("Fig2 …")
    figure_param_importance()
    print("Fig3 …")
    figure_best_objective_comparison()
    print("Supp parallel …")
    figure_parallel_coordinates_supplementary()
    print("Done.")


if __name__ == "__main__":
    main()
