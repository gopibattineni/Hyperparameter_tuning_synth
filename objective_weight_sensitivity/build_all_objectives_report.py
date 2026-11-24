#!/usr/bin/env python3
"""Build all_objectives_full_report.xlsx with O1–O5 A/B/C/D (10-seed post-hoc).

O1 sheet is the original multiseed objective (40:40:20).
O2–O5 are post-hoc reweights of the same O1 trials (ST/MT/GP × 10 seeds).
On the O2 sheet, the separate Standard-TPE seed-42 HPO (F=SDV) is appended
below section D as \"Separate O2 HPO run\" — not mixed with post-hoc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[1]
POST = ROOT / "results/objective_weight_sensitivity/tables/o1_posthoc_reweight"
MULTI = ROOT / "results/multiseed_10seeds_all"
THREE = MULTI / "tables/three_samplers_report"
SENS = ROOT / "results/objective_weight_sensitivity/tables"
OUT = ROOT / "results/objective_weight_sensitivity/all_objectives_full_report.xlsx"

ORDER_DS = [
    "cancer",
    "alzheimers",
    "adult",
    "forest_cover",
    "bank_marketing",
    "wine_quality",
    "cdc_diabetes",
    "mushroom",
    "magic_gamma",
    "metro_interstate",
    "online_shopping",
    "air_quality",
    "concrete",
    "energy_efficiency",
    "real_estate",
]
DS_LABEL = {
    "cancer": "Cancer",
    "alzheimers": "Alzheimer's",
    "adult": "Adult",
    "forest_cover": "Forest Cover",
    "bank_marketing": "Bank Marketing",
    "wine_quality": "Wine Quality",
    "cdc_diabetes": "CDC Diabetes",
    "mushroom": "Mushroom",
    "magic_gamma": "MAGIC Gamma",
    "metro_interstate": "Metro Interstate",
    "online_shopping": "Online Shopping",
    "air_quality": "Air Quality",
    "concrete": "Concrete",
    "energy_efficiency": "Energy Efficiency",
    "real_estate": "Real Estate",
}
ORDER_GEN = [
    "gaussian_copula",
    "copulagan",
    "ctgan",
    "ctabgan",
    "tvae",
    "wgan_gp",
    "tabddpm",
]
GEN_LABEL = {
    "gaussian_copula": "Gaussian Copula",
    "copulagan": "CopulaGAN",
    "ctgan": "CTGAN",
    "ctabgan": "CTAB-GAN+",
    "tvae": "TVAE",
    "wgan_gp": "WGAN-GP",
    "tabddpm": "TabDDPM",
}
SAMPLER_KEY = {
    "standard_tpe": "ST",
    "multivariate_tpe": "MT",
    "gp": "GP",
}
SAMPLER_LABEL = {
    "standard_tpe": "Standard TPE",
    "multivariate_tpe": "Multivariate TPE",
    "gp": "Gaussian Process",
}
HOURS_TOTAL = {
    "standard_tpe": 339.4,
    "multivariate_tpe": 356.2,
    "gp": 419.9,
}
WEIGHTS = {
    "O1": (0.400, 0.400, 0.200, "O1 Original 40:40:20"),
    "O2": (0.200, 0.600, 0.200, "O2 Utility-focused 20:60:20"),
    "O3": (0.600, 0.200, 0.200, "O3 Fidelity-focused 60:20:20"),
    "O4": (0.300, 0.300, 0.400, "O4 Privacy-focused 30:30:40"),
    "O5": (0.333, 0.333, 0.334, "O5 Equal ≈33.3:33.3:33.4"),
}

NAVY, BLUE, ROW, BEST, WHITE, REF, SEP = (
    "1F4E79",
    "2E75B6",
    "DDEBF7",
    "C6EFCE",
    "FFFFFF",
    "FCE4D6",
    "FFF2CC",
)
THIN = Border(
    left=Side(style="thin", color="B0B0B0"),
    right=Side(style="thin", color="B0B0B0"),
    top=Side(style="thin", color="B0B0B0"),
    bottom=Side(style="thin", color="B0B0B0"),
)


def font(bold: bool = False, color: str = "000000", size: int = 10) -> Font:
    return Font(name="Calibri", bold=bold, color=color, size=size)


def fill(h: str) -> PatternFill:
    return PatternFill("solid", fgColor=h)


def load_hours() -> dict[tuple[str, str, str], float]:
    """(sampler, dataset, generator) → sum wall hours over seeds."""
    out: dict[tuple[str, str, str], float] = {}
    files = {
        "standard_tpe": MULTI / "tables/standard_tpe_mean_std_with_wallclock.csv",
        "multivariate_tpe": MULTI / "tables/multivariate_tpe_mean_std_with_wallclock.csv",
        "gp": MULTI / "tables/gp_mean_std_with_wallclock.csv",
    }
    for samp, path in files.items():
        df = pd.read_csv(path)
        for r in df.itertuples(index=False):
            out[(samp, r.dataset, r.generator)] = float(r.sum_wall_hours)
    return out


def ctabgan_hours(hours: dict, sampler: str) -> float:
    return sum(
        v
        for (s, d, g), v in hours.items()
        if s == sampler and g == "ctgan"
    )


def build_A(preset: str, hours: dict) -> pd.DataFrame:
    summary = pd.read_csv(POST / f"{preset}_A_summary.csv")
    # Ensure mean seed-std present
    if "Mean seed-std (cells)" not in summary.columns:
        agg = pd.read_csv(POST / "generator_mean_std_by_preset.csv")
        rows = []
        for samp, label in SAMPLER_LABEL.items():
            g = agg.loc[(agg.preset == preset) & (agg.sampler == samp)]
            rows.append(
                {
                    "Sampler": label,
                    "Mean seed-std (cells)": round(float(g["std_J"].mean()), 4)
                    if len(g)
                    else np.nan,
                }
            )
        summary = summary.merge(pd.DataFrame(rows), on="Sampler", how="left")

    rows = []
    for samp, label in SAMPLER_LABEL.items():
        sub = summary.loc[summary["Sampler"] == label].iloc[0]
        tot = HOURS_TOTAL[samp]
        ctgan_h = round(ctabgan_hours(hours, samp), 1)
        rows.append(
            {
                "Sampler": label,
                "Total hours": tot,
                "Days": round(tot / 24, 1),
                "Reported total (h)": tot,
                "CTGAN hours": ctgan_h,
                "CTGAN share": f"{100 * ctgan_h / tot:.1f}%",
                "Dataset-level mean J": float(sub["Dataset-level mean J"]),
                "Avg rank": float(sub["Avg rank (1=best)"]),
                "Dataset wins": int(sub["Dataset wins"]),
                "Cells with std ≥ 0.10": int(sub["Cells with std≥0.10"]),
                "Mean seed-std (cells)": float(sub["Mean seed-std (cells)"]),
            }
        )
    return pd.DataFrame(rows)


def build_B(preset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fried = pd.read_csv(POST / f"{preset}_B_friedman.csv")
    pair = pd.read_csv(POST / f"{preset}_B_pairwise.csv")
    # Normalize column names for display
    rename = {
        "χ²": "Statistic (χ²)",
        "p": "p-value",
        "Kendall W": "Kendall's W",
        "Sig α=0.05": "Significant (α=0.05)",
        "p Holm": "p Holm",
        "p_raw": "p raw",
    }
    fried = fried.rename(columns={k: v for k, v in rename.items() if k in fried.columns})
    pair = pair.rename(columns={k: v for k, v in rename.items() if k in pair.columns})
    # Ensure p-value columns are numeric floats (not blank)
    for c in ("p raw", "p Holm", "p_raw", "p Holm"):
        if c in pair.columns:
            pair[c] = pair[c].astype(float)
    return fried, pair


def build_C(preset: str) -> pd.DataFrame:
    return pd.read_csv(POST / f"{preset}_C_dataset_mean.csv")


def build_D(preset: str, sampler: str, hours: dict) -> pd.DataFrame:
    agg = pd.read_csv(POST / "generator_mean_std_by_preset.csv")
    g = agg.loc[(agg.preset == preset) & (agg.sampler == sampler)].copy()
    rows = []
    for ds in ORDER_DS:
        row: dict = {"Dataset": DS_LABEL[ds]}
        for gen in ORDER_GEN:
            hit = g.loc[(g.dataset == ds) & (g.generator == gen)]
            lab = GEN_LABEL[gen]
            if hit.empty:
                row[f"{lab} Mean J"] = np.nan
                row[f"{lab} Std"] = np.nan
                row[f"{lab} Hours"] = hours.get((sampler, ds, gen), np.nan)
            else:
                row[f"{lab} Mean J"] = round(float(hit["mean_J"].iloc[0]), 4)
                row[f"{lab} Std"] = round(float(hit["std_J"].iloc[0]), 4)
                row[f"{lab} Hours"] = round(
                    float(hours.get((sampler, ds, gen), np.nan)), 2
                )
        rows.append(row)
    df = pd.DataFrame(rows)
    # Bottom totals row
    bottom: dict = {"Dataset": "MEAN / SUM"}
    for gen in ORDER_GEN:
        lab = GEN_LABEL[gen]
        bottom[f"{lab} Mean J"] = round(float(df[f"{lab} Mean J"].mean()), 4)
        bottom[f"{lab} Std"] = round(float(df[f"{lab} Std"].mean()), 4)
        bottom[f"{lab} Hours"] = round(float(df[f"{lab} Hours"].sum()), 2)
    df = pd.concat([df, pd.DataFrame([bottom])], ignore_index=True)
    return df


def style_header_row(ws, r: int, ncols: int, color: str = NAVY) -> None:
    for c in range(1, ncols + 1):
        cell = ws.cell(r, c)
        cell.fill = fill(color)
        cell.font = font(True, WHITE)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN


def write_block(
    ws,
    start_row: int,
    title: str,
    note: str,
    df: pd.DataFrame,
    *,
    highlight_best_j: bool = False,
    header_color: str = NAVY,
) -> int:
    """Write a titled table; return next free row."""
    ncols = max(len(df.columns), 3)
    ws.merge_cells(
        start_row=start_row, start_column=1, end_row=start_row, end_column=ncols
    )
    ws.cell(start_row, 1, title).font = font(True, NAVY, 13)
    ws.merge_cells(
        start_row=start_row + 1, start_column=1, end_row=start_row + 1, end_column=ncols
    )
    note_cell = ws.cell(start_row + 1, 1, note)
    note_cell.font = font(False, NAVY, 9)
    note_cell.fill = fill(REF)

    r = start_row + 3
    for c, h in enumerate(df.columns, 1):
        ws.cell(r, c, h)
    style_header_row(ws, r, len(df.columns), header_color)
    r += 1

    jcols = [
        c
        for c in df.columns
        if ("mean J" in c.lower() or c.endswith("Mean J")) and "Std" not in c
    ]
    for _, row in df.iterrows():
        best_cols: set[str] = set()
        if highlight_best_j and jcols:
            nums = {
                c: float(row[c])
                for c in jcols
                if pd.notna(row[c]) and str(row.get("Dataset", "")) != "MEAN / SUM"
            }
            # For summary A: best among Dataset-level mean J across samplers is per-row N/A;
            # for C: highlight best sampler mean J columns
            if "ST mean J" in df.columns:
                nums = {
                    c: float(row[c])
                    for c in ("ST mean J", "MT mean J", "GP mean J")
                    if pd.notna(row[c])
                }
                if nums:
                    mx = max(nums.values())
                    best_cols = {c for c, v in nums.items() if abs(v - mx) < 1e-12}
            elif "Dataset-level mean J" in df.columns:
                # highlight max mean J among the three sampler rows later externally
                pass
        for c, col in enumerate(df.columns, 1):
            v = row[col]
            cell = ws.cell(
                r, c, None if (isinstance(v, float) and pd.isna(v)) else v
            )
            cell.border = THIN
            cell.alignment = Alignment(
                horizontal="center" if c > 1 else "left",
                vertical="center",
                wrap_text=True,
            )
            if col in best_cols or (
                col in ("Significant", "Significant (α=0.05)") and v == "Yes"
            ):
                cell.fill = fill(BEST)
            elif col == "Best sampler":
                cell.fill = fill(BEST)
            elif str(row.get("Dataset", "")) == "MEAN / SUM":
                cell.font = font(True)
                cell.fill = fill(ROW)
            elif r % 2 == 0:
                cell.fill = fill(ROW)
            if isinstance(v, float):
                if "p" in col.lower() or "holm" in col.lower() or "raw" in col.lower():
                    cell.number_format = "0.0000"
                elif abs(v) < 10:
                    cell.number_format = "0.0000"
                else:
                    cell.number_format = "0.00"
        r += 1

    # autosize lightly
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        w = ws.column_dimensions[letter].width or 12
        for cell in col:
            if cell.value is not None:
                w = max(w, min(len(str(cell.value)) + 2, 28))
        ws.column_dimensions[letter].width = w
    return r + 2


def highlight_A_best_mean_j(ws, a_start_header_row: int, n_rows: int = 3) -> None:
    """Shade best Dataset-level mean J among the three sampler rows."""
    # header at a_start_header_row; data follows
    # find column index for Dataset-level mean J
    header = [
        ws.cell(a_start_header_row, c).value for c in range(1, ws.max_column + 1)
    ]
    try:
        jcol = header.index("Dataset-level mean J") + 1
    except ValueError:
        return
    vals = []
    for r in range(a_start_header_row + 1, a_start_header_row + 1 + n_rows):
        vals.append((r, ws.cell(r, jcol).value))
    nums = [(r, float(v)) for r, v in vals if isinstance(v, (int, float))]
    if not nums:
        return
    mx = max(v for _, v in nums)
    for r, v in nums:
        if abs(v - mx) < 1e-12:
            ws.cell(r, jcol).fill = fill(BEST)


def write_preset_sheet(
    wb: Workbook,
    preset: str,
    hours: dict,
    *,
    separate_o2: pd.DataFrame | None = None,
    separate_o2_ds: pd.DataFrame | None = None,
) -> None:
    wf, wu, wp, label = WEIGHTS[preset]
    ws = wb.create_sheet(preset)
    ws.sheet_view.showGridLines = False

    # Title
    ws.merge_cells("A1:K1")
    ws["A1"] = (
        f"{label} — Standard TPE vs Multivariate TPE vs Gaussian Process (10 seeds)"
    )
    ws["A1"].font = font(True, NAVY, 14)
    ws.merge_cells("A2:K2")
    if preset == "O1":
        note = (
            f"O1 objective: J = {wf}×Fidelity + {wu}×Utility − {wp}×Privacy risk. "
            "Original multiseed HPO (15 datasets × 7 generators × 3 samplers × 10 seeds). "
            "Blue-style reported wall-clock from study summaries; black metrics from evaluated trials."
        )
    else:
        note = (
            f"Post-hoc reweight of O1 trials under {label}: "
            f"J = {wf}×F + {wu}×U − {wp}×P. Same 3150 studies — NO new HPO. "
            "Section D Mean J / Std use reweighted best trials; Hours are O1 wall-clock "
            "(unchanged by reweight)."
        )
    ws["A2"] = note
    ws["A2"].font = font(False, NAVY, 9)
    ws["A2"].fill = fill(REF)

    # A
    A = build_A(preset, hours)
    # Remember header row = 6 after write_block from row 4
    next_r = write_block(
        ws,
        4,
        "A. Summary by sampler",
        "Dataset-level mean J = mean over 7 generators of 10-seed mean best-J. "
        "Seed-spread columns from section D cell stds. Wall-clock = O1 study hours.",
        A,
        highlight_best_j=True,
    )
    # header was at 4+3=7
    highlight_A_best_mean_j(ws, 7, n_rows=3)

    # B
    fried, pair = build_B(preset)
    next_r = write_block(
        ws,
        next_r,
        "B. Statistical tests (dataset-level)",
        "Friedman omnibus then pairwise Wilcoxon signed-rank with Holm correction (α=0.05). "
        "p raw and p Holm shown for all pairs.",
        fried,
        header_color=BLUE,
    )
    # pairwise immediately under friedman without big gap — rewrite pairwise at next_r-1
    next_r = write_block(
        ws,
        next_r,
        "B2. Pairwise Wilcoxon + Holm (p-values)",
        "Full p raw / p Holm for ST vs MT, ST vs GP, MT vs GP.",
        pair,
        header_color=BLUE,
    )

    # C
    C = build_C(preset)
    next_r = write_block(
        ws,
        next_r,
        "C. Dataset-level mean J and ranks (ST / MT / GP)",
        "Mean over 7 generators of seed-mean best J under this preset. "
        "Tie-break for Best: ST > MT > GP.",
        C,
        highlight_best_j=True,
    )

    # D for each sampler
    for samp, lab in SAMPLER_LABEL.items():
        D = build_D(preset, samp, hours)
        next_r = write_block(
            ws,
            next_r,
            f"D. Detailed results — {lab}: mean J, std and wall-clock hours over 10 seeds "
            f"(reported total {HOURS_TOTAL[samp]} h)",
            "Per cell: Mean J / Std of reweighted best-J across seeds; "
            "Hours = sum of O1 study wall-clock hours over seeds. Bottom: mean of Mean J / sum of Hours.",
            D,
        )

    if preset == "O2" and separate_o2 is not None:
        ws.cell(next_r, 1, "Separate O2 HPO run").font = font(True, NAVY, 13)
        ws.merge_cells(start_row=next_r, start_column=1, end_row=next_r, end_column=8)
        next_r += 1
        ws.merge_cells(start_row=next_r, start_column=1, end_row=next_r, end_column=8)
        cell = ws.cell(
            next_r,
            1,
            "Different experiment from the post-hoc reweight above: independent Standard-TPE "
            "HPO with O2 weights (20:60:20), seed 42 only, F = SDV QualityReport. "
            "Do not mix with the 10-seed ST/MT/GP post-hoc tables.",
        )
        cell.font = font(False, NAVY, 9)
        cell.fill = fill(SEP)
        next_r += 2
        if separate_o2_ds is not None:
            next_r = write_block(
                ws,
                next_r,
                "Separate O2 HPO — dataset-level mean J (ST, seed 42, F=SDV)",
                "From objective_weight_sensitivity O2 run. Single seed → no seed-std.",
                separate_o2_ds,
                header_color=BLUE,
            )
        next_r = write_block(
            ws,
            next_r,
            "Separate O2 HPO — detailed J and hours per generator (ST, seed 42)",
            "Source: results/objective_weight_sensitivity/tables/O2_detailed_per_generator.csv",
            separate_o2,
            header_color=BLUE,
        )


def main() -> int:
    hours = load_hours()
    best = pd.read_csv(POST / "study_best_under_each_preset.csv")
    assert len(best) == 15750, len(best)

    # Separate O2 HPO tables (seed 42 ST, F=SDV)
    sep_detail = pd.read_csv(SENS / "O2_detailed_per_generator.csv")
    sep_ds = pd.read_csv(SENS / "O2_dataset_level_mean_J.csv")
    # normalize sep_ds columns if needed
    if "Dataset" not in sep_ds.columns and "dataset" in sep_ds.columns:
        sep_ds = sep_ds.rename(columns={"dataset": "Dataset"})

    wb = Workbook()
    wb.remove(wb.active)

    cover = wb.create_sheet("00_Cover", 0)
    cover["A1"] = "FORGE — O1–O5 results (10-seed ST/MT/GP post-hoc + original O1)"
    cover["A1"].font = font(True, NAVY, 14)
    cover.merge_cells("A1:F1")
    notes = [
        "O1 sheet = Original objective 40:40:20 (multiseed HPO that was actually run).",
        "O2–O5 sheets = post-hoc reweight of the same O1 evaluated trials (NO new HPO under those weights).",
        "Each preset sheet has A (summary) / B (Friedman + pairwise p-values) / C (per-dataset) / D (per-generator × sampler).",
        "A seed-spread columns (std≥0.10, mean seed-std) are computed from section D cell stds.",
        "Wall-clock hours on O2–O5 are O1 study hours (reweight does not change runtime).",
        "O2 sheet only: below D, a separate block “Separate O2 HPO run” = ST seed-42 sensitivity HPO with F=SDV — different experiment.",
    ]
    for i, n in enumerate(notes, 3):
        cover.cell(i, 1, n)
        cover.merge_cells(start_row=i, start_column=1, end_row=i, end_column=6)
    cover.column_dimensions["A"].width = 110

    for preset in ("O1", "O2", "O3", "O4", "O5"):
        write_preset_sheet(
            wb,
            preset,
            hours,
            separate_o2=sep_detail if preset == "O2" else None,
            separate_o2_ds=sep_ds if preset == "O2" else None,
        )

    # Keep a thin overall comparison of post-hoc generator means
    gen = pd.read_csv(POST / "generator_overall_by_preset.csv")
    ws = wb.create_sheet("Overall_Generator_J")
    write_block(
        ws,
        1,
        "Mean J by generator × preset (post-hoc, mean over datasets×samplers×seeds)",
        "Comparable within a preset only for ranking generators; raw J scale differs across weight presets.",
        gen.pivot(index="generator", columns="preset", values="Mean J")
        .reindex(columns=["O1", "O2", "O3", "O4", "O5"])
        .reset_index(),
    )
    shift = pd.read_csv(POST / "best_trial_shift_vs_O1.csv")
    write_block(
        ws,
        20,
        "Best-trial shift vs O1",
        "Fraction of (dataset,generator,sampler,seed) studies whose argmax trial changes under reweight.",
        shift,
        header_color=BLUE,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
