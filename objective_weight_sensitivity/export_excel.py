#!/usr/bin/env python3
"""Export objective-weight sensitivity results to a styled Excel workbook.

Formatting matches ``objective_weight_sensitivity/templates/Sampler_results_10_seeds.xlsx``
(FORGE navy / blue / row-blue / best-green palette, section A–E layout, no gridlines).
Every objective sheet (O1–O5) uses the same section order, headers, number formats,
and chart set so all configurations share one presentation.

The template file is never overwritten; only ``objective_weight_comparison.xlsx``
is regenerated from sensitivity CSVs.

Usage::

    python -m objective_weight_sensitivity.export_excel \\
      --results-root results/objective_weight_sensitivity

    python -m objective_weight_sensitivity.run --export-excel \\
      --results-root results/objective_weight_sensitivity
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference, ScatterChart, Series
from openpyxl.chart.label import DataLabelList
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.objective_presets import PRESET_NAMES, resolve_preset
from objective_weight_sensitivity.runner import default_results_root

# Local copy of the Sampler 10-seed workbook (presentation template; never overwrite).
TEMPLATE_XLSX = Path(__file__).resolve().parent / "templates" / "Sampler_results_10_seeds.xlsx"

# FORGE / Sampler_results_10_seeds palette
NAVY = "1F4E79"
BLUE = "2E75B6"
LIGHT_BLUE = "BDD7EE"
ROW_BLUE = "DDEBF7"
REF_ORANGE = "FCE4D6"
BEST_GREEN = "C6EFCE"
WHITE = "FFFFFF"

SHEET_META = {
    "O1": ("O1_Original", "Original 40:40:20"),
    "O2": ("O2_Utility", "Utility-focused"),
    "O3": ("O3_Fidelity", "Fidelity-focused"),
    "O4": ("O4_Privacy", "Privacy-focused"),
    "O5": ("O5_Equal", "Equal weighting"),
}

THIN = Border(
    left=Side(style="thin", color="B0B0B0"),
    right=Side(style="thin", color="B0B0B0"),
    top=Side(style="thin", color="B0B0B0"),
    bottom=Side(style="thin", color="B0B0B0"),
)


def _fill(hex_rgb: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_rgb)


def _font(*, bold: bool = False, color: str = "000000", size: int = 10) -> Font:
    return Font(name="Calibri", bold=bold, color=color, size=size)


def _center() -> Alignment:
    return Alignment(horizontal="center", vertical="center", wrap_text=True)


def _left() -> Alignment:
    return Alignment(horizontal="left", vertical="center", wrap_text=True)


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    return False


def _num(v: Any, nd: int = 4) -> Any:
    """Format numeric values; leave blanks for missing (never coerce to 0)."""
    if _is_missing(v):
        return None
    try:
        return round(float(v), nd)
    except Exception:
        return v


def _style_header_row(ws, row: int, start_col: int, end_col: int, fill_hex: str = NAVY) -> None:
    for col in range(start_col, end_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = _fill(fill_hex)
        cell.font = _font(bold=True, color=WHITE, size=10)
        cell.alignment = _center()
        cell.border = THIN


def _style_data_row(ws, row: int, start_col: int, end_col: int, *, best: bool = False) -> None:
    # Match Sampler_results_10_seeds: green for best, else alternating row blue.
    if best:
        fill = _fill(BEST_GREEN)
    elif row % 2 == 0:
        fill = _fill(ROW_BLUE)
    else:
        fill = None
    for col in range(start_col, end_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.border = THIN
        cell.alignment = _center()
        if fill is not None:
            cell.fill = fill
        if isinstance(cell.value, float):
            cell.number_format = "0.0000"


def _write_section_title(ws, row: int, title: str, ncols: int = 10) -> int:
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=ncols)
    cell = ws.cell(row=row, column=1, value=title)
    cell.font = _font(bold=True, color=NAVY, size=12)
    cell.alignment = _left()
    return row + 1


def _autosize(ws, min_width: float = 10, max_width: float = 28) -> None:
    for col_cells in ws.columns:
        letter = get_column_letter(col_cells[0].column)
        length = 0
        for c in col_cells:
            if c.value is None:
                continue
            length = max(length, min(len(str(c.value)), 60))
        ws.column_dimensions[letter].width = max(min_width, min(max_width, length + 2))


def load_sensitivity_tables(results_root: Path) -> dict[str, Any]:
    root = Path(results_root)
    comparison = root / "objective_weight_comparison.csv"
    best = root / "best_trials.csv"
    trials = root / "all_trials.csv"
    meta = root / "objective_metadata.json"

    out: dict[str, Any] = {
        "comparison": pd.read_csv(comparison) if comparison.is_file() else pd.DataFrame(),
        "best": pd.read_csv(best) if best.is_file() else pd.DataFrame(),
        "trials": pd.read_csv(trials) if trials.is_file() else pd.DataFrame(),
        "metadata": {},
    }
    if meta.is_file():
        out["metadata"] = json.loads(meta.read_text(encoding="utf-8"))
    return out


def _best_trial_number(trials: pd.DataFrame, row: pd.Series) -> Any:
    if trials.empty:
        return None
    mask = (
        (trials["dataset"] == row.get("dataset"))
        & (trials["generator"] == row.get("generator"))
        & (trials["config"].astype(str) == str(row.get("config")))
        & (trials["seed"] == row.get("seed"))
        & (trials["state"] == "COMPLETE")
    )
    sub = trials.loc[mask]
    if sub.empty or sub["objective"].isna().all():
        return None
    idx = sub["objective"].astype(float).idxmax()
    return int(sub.loc[idx, "trial_number"])


def _preset_weights(config: str) -> dict[str, float]:
    try:
        cfg = resolve_preset(config, force_sdv_only=True)
        return dict(cfg["weights"])
    except Exception:
        return {"fidelity": None, "utility": None, "privacy_risk": None}


def _write_config_summary(
    ws,
    start_row: int,
    *,
    config: str,
    label: str,
    weights: dict[str, Any],
    subset: pd.DataFrame,
    metadata: dict[str, Any],
) -> int:
    r = _write_section_title(ws, start_row, "A. Objective configuration summary")
    pairs = [
        ("Configuration", f"{config} — {label}"),
        ("Fidelity weight (wF)", _num(weights.get("fidelity"), 3)),
        ("Utility weight (wU)", _num(weights.get("utility"), 3)),
        ("Privacy weight (wP)", _num(weights.get("privacy_risk"), 3)),
        ("Objective equation", "J = wF×F + wU×U − wP×P  (maximize)"),
        ("Fidelity definition", "SDV QualityReport only (KS/Corr diagnostics only)"),
        ("Utility definition", "Existing TSTR/TRTR gap formulation (unchanged)"),
        ("Privacy definition", "0.5·MIA + 0.3·NNDR_risk + 0.2·DCR_risk (unchanged)"),
        ("Sampler", "Standard TPE"),
        (
            "Datasets",
            ", ".join(sorted(subset["dataset"].astype(str).unique())) if not subset.empty else "(none yet)",
        ),
        (
            "Generators",
            ", ".join(sorted(subset["generator"].astype(str).unique())) if not subset.empty else "(none yet)",
        ),
        (
            "Seed(s)",
            ", ".join(str(int(s)) for s in sorted(subset["seed"].dropna().unique()))
            if not subset.empty and "seed" in subset.columns
            else "(none yet)",
        ),
        (
            "Trials (target / observed complete)",
            (
                f"n_complete mean={_num(subset['n_complete'].mean(), 1)}"
                if not subset.empty and "n_complete" in subset.columns
                else "(none yet)"
            ),
        ),
        (
            "Notes",
            "Do not compare raw J across weight configs as equivalent scores; compare F, U, P.",
        ),
    ]
    ws.cell(row=r, column=1, value="Field").fill = _fill(NAVY)
    ws.cell(row=r, column=1).font = _font(bold=True, color=WHITE)
    ws.cell(row=r, column=2, value="Value").fill = _fill(NAVY)
    ws.cell(row=r, column=2).font = _font(bold=True, color=WHITE)
    ws.cell(row=r, column=1).border = THIN
    ws.cell(row=r, column=2).border = THIN
    r += 1
    for i, (k, v) in enumerate(pairs):
        ws.cell(row=r, column=1, value=k).font = _font(bold=True, color=NAVY)
        ws.cell(row=r, column=1).fill = _fill(ROW_BLUE if i % 2 == 0 else WHITE)
        ws.cell(row=r, column=1).border = THIN
        cell = ws.cell(row=r, column=2, value="" if v is None else v)
        cell.fill = _fill(ROW_BLUE if i % 2 == 0 else WHITE)
        cell.border = THIN
        cell.alignment = _left()
        r += 1
    return r + 1


def _write_performance_table(ws, start_row: int, subset: pd.DataFrame) -> tuple[int, int | None, int | None]:
    """Return (next_row, data_start_row, data_end_row) for charts."""
    r = _write_section_title(ws, start_row, "B. Performance summary")
    headers = [
        "Dataset",
        "Generator",
        "Best Objective J",
        "Fidelity",
        "Utility",
        "Privacy",
        "MIA AUC",
        "NNDR",
        "DCR",
    ]
    for c, h in enumerate(headers, start=1):
        ws.cell(row=r, column=c, value=h)
    _style_header_row(ws, r, 1, len(headers), NAVY)
    r += 1
    if subset.empty:
        ws.cell(row=r, column=1, value="No results for this configuration yet.")
        ws.cell(row=r, column=1).font = _font(color=NAVY, size=10)
        return r + 2, None, None

    # Best J within this config sheet
    jcol = "best_objective"
    best_idx = None
    if jcol in subset.columns and subset[jcol].notna().any():
        best_idx = subset[jcol].astype(float).idxmax()

    data_start = r
    for idx, row in subset.sort_values(["dataset", "generator"]).iterrows():
        vals = [
            row.get("dataset"),
            row.get("generator"),
            _num(row.get("best_objective")),
            _num(row.get("best_fidelity")),
            _num(row.get("best_utility")),
            _num(row.get("best_privacy_risk")),
            _num(row.get("best_mia_auc")),
            _num(row.get("best_nndr")),
            _num(row.get("best_dcr")),
        ]
        for c, v in enumerate(vals, start=1):
            ws.cell(row=r, column=c, value=v)
        _style_data_row(
            ws, r, 1, len(headers), best=(best_idx is not None and idx == best_idx)
        )
        r += 1
    data_end = r - 1
    return r + 1, data_start, data_end


def _write_detailed_results(
    ws, start_row: int, subset: pd.DataFrame, trials: pd.DataFrame
) -> int:
    r = _write_section_title(ws, start_row, "C. Detailed optimisation results")
    headers = [
        "Dataset",
        "Generator",
        "Seed",
        "Best trial #",
        "Best J",
        "Best hyperparameters",
        "n_complete",
        "n_fail",
        "n_pruned",
        "Fidelity",
        "Utility",
        "Privacy",
        "Wall-clock (s)",
    ]
    for c, h in enumerate(headers, start=1):
        ws.cell(row=r, column=c, value=h)
    _style_header_row(ws, r, 1, len(headers), BLUE)
    r += 1
    if subset.empty:
        ws.cell(row=r, column=1, value="No detailed results available.")
        return r + 2

    for _, row in subset.sort_values(["dataset", "generator", "seed"]).iterrows():
        params = row.get("best_params")
        if _is_missing(params):
            params = None
        vals = [
            row.get("dataset"),
            row.get("generator"),
            int(row["seed"]) if not _is_missing(row.get("seed")) else None,
            _best_trial_number(trials, row),
            _num(row.get("best_objective")),
            params,
            _num(row.get("n_complete"), 0),
            _num(row.get("n_fail"), 0),
            _num(row.get("n_pruned"), 0),
            _num(row.get("best_fidelity")),
            _num(row.get("best_utility")),
            _num(row.get("best_privacy_risk")),
            _num(row.get("study_wall_clock_s"), 2),
        ]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.border = THIN
            cell.alignment = _center() if c != 6 else _left()
            if isinstance(v, float):
                cell.number_format = "0.0000" if c != 13 else "0.00"
        r += 1
    return r + 1


def _write_generator_comparison(ws, start_row: int, subset: pd.DataFrame) -> int:
    r = _write_section_title(
        ws,
        start_row,
        "D. Generator comparison (within this objective configuration)",
    )
    headers = [
        "Dataset",
        "Generator",
        "Mean J",
        "Mean Fidelity",
        "Mean Utility",
        "Mean Privacy",
        "Std J",
        "n_runs",
        "Best on dataset?",
    ]
    for c, h in enumerate(headers, start=1):
        ws.cell(row=r, column=c, value=h)
    _style_header_row(ws, r, 1, len(headers), BLUE)
    r += 1
    if subset.empty:
        ws.cell(row=r, column=1, value="No generator comparison available.")
        return r + 2

    g = (
        subset.groupby(["dataset", "generator"], dropna=False)
        .agg(
            mean_j=("best_objective", "mean"),
            mean_f=("best_fidelity", "mean"),
            mean_u=("best_utility", "mean"),
            mean_p=("best_privacy_risk", "mean"),
            std_j=("best_objective", "std"),
            n_runs=("best_objective", "count"),
        )
        .reset_index()
    )
    # Best generator per dataset by mean J
    best_by_ds: dict[str, str] = {}
    for ds, sub in g.groupby("dataset"):
        if sub["mean_j"].notna().any():
            best_by_ds[str(ds)] = str(sub.loc[sub["mean_j"].idxmax(), "generator"])

    for _, row in g.sort_values(["dataset", "mean_j"], ascending=[True, False]).iterrows():
        n_runs = int(row["n_runs"]) if not _is_missing(row["n_runs"]) else 0
        std_j = _num(row["std_j"]) if n_runs >= 2 else None  # never invent std
        is_best = best_by_ds.get(str(row["dataset"])) == str(row["generator"])
        vals = [
            row["dataset"],
            row["generator"],
            _num(row["mean_j"]),
            _num(row["mean_f"]),
            _num(row["mean_u"]),
            _num(row["mean_p"]),
            std_j,
            n_runs if n_runs else None,
            "YES" if is_best else "",
        ]
        for c, v in enumerate(vals, start=1):
            ws.cell(row=r, column=c, value=v)
        _style_data_row(ws, r, 1, len(headers), best=is_best)
        r += 1
    return r + 1


def _add_charts(
    ws,
    *,
    data_start: int | None,
    data_end: int | None,
    anchor_row: int,
) -> int:
    """Add F/U/P grouped bar, J bar, and two scatters from performance table."""
    r = _write_section_title(ws, anchor_row, "E. Visualisations")
    if data_start is None or data_end is None or data_end < data_start:
        ws.cell(row=r, column=1, value="Charts omitted — no numeric results for this configuration.")
        return r + 2

    # Hidden helper columns for chart series at far right (K+)
    # Copy generator labels + F, U, P, J for charting
    helper_col = 12  # L
    ws.cell(row=data_start - 1, column=helper_col, value="Generator")
    ws.cell(row=data_start - 1, column=helper_col + 1, value="Fidelity")
    ws.cell(row=data_start - 1, column=helper_col + 2, value="Utility")
    ws.cell(row=data_start - 1, column=helper_col + 3, value="Privacy")
    ws.cell(row=data_start - 1, column=helper_col + 4, value="J")
    for i, src_row in enumerate(range(data_start, data_end + 1)):
        rr = data_start + i
        # Generator may collide across datasets — use "dataset|generator"
        ds = ws.cell(row=src_row, column=1).value
        gen = ws.cell(row=src_row, column=2).value
        ws.cell(row=rr, column=helper_col, value=f"{ds}|{gen}")
        ws.cell(row=rr, column=helper_col + 1, value=ws.cell(row=src_row, column=4).value)
        ws.cell(row=rr, column=helper_col + 2, value=ws.cell(row=src_row, column=5).value)
        ws.cell(row=rr, column=helper_col + 3, value=ws.cell(row=src_row, column=6).value)
        ws.cell(row=rr, column=helper_col + 4, value=ws.cell(row=src_row, column=3).value)

    cats = Reference(ws, min_col=helper_col, min_row=data_start, max_row=data_end)

    # 1) Grouped F/U/P bar
    bar = BarChart()
    bar.type = "col"
    bar.grouping = "clustered"
    bar.title = "Fidelity, Utility, Privacy by generator"
    bar.y_axis.title = "Score"
    data = Reference(
        ws,
        min_col=helper_col + 1,
        max_col=helper_col + 3,
        min_row=data_start - 1,
        max_row=data_end,
    )
    bar.add_data(data, titles_from_data=True)
    bar.set_categories(cats)
    bar.shape = 4
    bar.dataLabels = DataLabelList()
    bar.dataLabels.showVal = True
    bar.width = 18
    bar.height = 10
    ws.add_chart(bar, f"A{r}")
    r += 18

    # 2) Objective J comparison
    bar2 = BarChart()
    bar2.type = "col"
    bar2.title = "Best objective J by generator (this weight config only)"
    bar2.y_axis.title = "J (config-specific scale)"
    data_j = Reference(
        ws, min_col=helper_col + 4, min_row=data_start - 1, max_row=data_end
    )
    bar2.add_data(data_j, titles_from_data=True)
    bar2.set_categories(cats)
    bar2.dataLabels = DataLabelList()
    bar2.dataLabels.showVal = True
    bar2.width = 15
    bar2.height = 9
    ws.add_chart(bar2, f"A{r}")
    r += 16

    # 3) Fidelity vs Utility scatter
    sc = ScatterChart()
    sc.title = "Fidelity vs Utility"
    sc.x_axis.title = "Fidelity"
    sc.y_axis.title = "Utility"
    xvalues = Reference(ws, min_col=helper_col + 1, min_row=data_start, max_row=data_end)
    yvalues = Reference(ws, min_col=helper_col + 2, min_row=data_start, max_row=data_end)
    series = Series(yvalues, xvalues, title="F vs U")
    sc.series.append(series)
    sc.width = 12
    sc.height = 9
    ws.add_chart(sc, f"A{r}")
    r += 16

    # 4) Utility vs Privacy scatter
    sc2 = ScatterChart()
    sc2.title = "Utility vs Privacy risk"
    sc2.x_axis.title = "Utility"
    sc2.y_axis.title = "Privacy risk"
    x2 = Reference(ws, min_col=helper_col + 2, min_row=data_start, max_row=data_end)
    y2 = Reference(ws, min_col=helper_col + 3, min_row=data_start, max_row=data_end)
    series2 = Series(y2, x2, title="U vs P")
    sc2.series.append(series2)
    sc2.width = 12
    sc2.height = 9
    ws.add_chart(sc2, f"A{r}")
    r += 16
    return r


def _build_config_sheet(
    wb: Workbook,
    *,
    config: str,
    sheet_name: str,
    label: str,
    tables: dict[str, Any],
) -> None:
    ws = wb.create_sheet(sheet_name)
    ws.sheet_view.showGridLines = False
    ws.row_dimensions[1].height = 22

    comparison: pd.DataFrame = tables["comparison"]
    best: pd.DataFrame = tables["best"]
    trials: pd.DataFrame = tables["trials"]
    metadata = tables["metadata"]

    # Prefer best_trials for detailed cols; fall back to comparison
    src = best if not best.empty else comparison
    if not src.empty and "config" in src.columns:
        subset = src.loc[src["config"].astype(str) == config].copy()
    else:
        subset = pd.DataFrame()

    weights = _preset_weights(config)
    if not subset.empty and "w_fidelity" in subset.columns:
        # Prefer recorded weights from results when present
        row0 = subset.iloc[0]
        weights = {
            "fidelity": row0.get("w_fidelity", weights.get("fidelity")),
            "utility": row0.get("w_utility", weights.get("utility")),
            "privacy_risk": row0.get("w_privacy_risk", weights.get("privacy_risk")),
        }

    # Title + note (same layout as Sampler_results_10_seeds sampler sheets)
    ws.merge_cells("A1:I1")
    title = ws.cell(row=1, column=1, value=f"Objective weight sensitivity — {config} ({label})")
    title.font = _font(bold=True, color=NAVY, size=14)
    title.alignment = _left()
    ws.merge_cells("A2:I2")
    note = ws.cell(
        row=2,
        column=1,
        value=(
            "Same format for every objective sheet (O1–O5). "
            "Raw J is weight-config-specific — compare Fidelity / Utility / Privacy across sheets."
        ),
    )
    note.font = _font(color=NAVY, size=9)
    note.fill = _fill(REF_ORANGE)

    r = 4
    r = _write_config_summary(
        ws, r, config=config, label=label, weights=weights, subset=subset, metadata=metadata
    )
    r, data_start, data_end = _write_performance_table(ws, r, subset)
    r = _write_detailed_results(ws, r, subset, trials)
    r = _write_generator_comparison(ws, r, subset)
    _add_charts(ws, data_start=data_start, data_end=data_end, anchor_row=r)
    _autosize(ws)


def _build_overall_sheet(wb: Workbook, tables: dict[str, Any]) -> None:
    ws = wb.create_sheet("Overall_Comparison", 0)
    ws.sheet_view.showGridLines = False
    ws.merge_cells("A1:H1")
    t = ws.cell(
        row=1,
        column=1,
        value="Overall comparison across objective-weight configurations (O1–O5)",
    )
    t.font = _font(bold=True, color=NAVY, size=14)

    ws.merge_cells("A2:H2")
    note = ws.cell(
        row=2,
        column=1,
        value=(
            "Note: raw Mean Objective J is config-specific and not directly comparable "
            "across rows. Compare Mean Fidelity / Utility / Privacy."
        ),
    )
    note.font = _font(color=NAVY, size=9)
    note.fill = _fill(REF_ORANGE)

    comparison: pd.DataFrame = tables["comparison"]
    headers = [
        "Configuration",
        "Fidelity Weight",
        "Utility Weight",
        "Privacy Weight",
        "Mean Fidelity",
        "Mean Utility",
        "Mean Privacy",
        "Mean Objective",
        "Best generator",
        "Best J (within config)",
        "n_studies",
    ]
    r = 4
    for c, h in enumerate(headers, start=1):
        ws.cell(row=r, column=c, value=h)
    _style_header_row(ws, r, 1, len(headers), NAVY)
    r += 1

    data_start = r
    rows_out = []
    for cfg in PRESET_NAMES:
        sheet_name, label = SHEET_META[cfg]
        w = _preset_weights(cfg)
        if comparison.empty or "config" not in comparison.columns:
            sub = pd.DataFrame()
        else:
            sub = comparison.loc[comparison["config"].astype(str) == cfg]
        mean_f = _num(sub["best_fidelity"].mean()) if not sub.empty else None
        mean_u = _num(sub["best_utility"].mean()) if not sub.empty else None
        mean_p = _num(sub["best_privacy_risk"].mean()) if not sub.empty else None
        mean_j = _num(sub["best_objective"].mean()) if not sub.empty else None
        best_gen = None
        best_j = None
        if not sub.empty and sub["best_objective"].notna().any():
            idx = sub["best_objective"].astype(float).idxmax()
            best_gen = sub.loc[idx, "generator"]
            best_j = _num(sub.loc[idx, "best_objective"])
        vals = [
            f"{cfg} — {label}",
            _num(w.get("fidelity"), 3),
            _num(w.get("utility"), 3),
            _num(w.get("privacy_risk"), 3),
            mean_f,
            mean_u,
            mean_p,
            mean_j,
            best_gen,
            best_j,
            int(len(sub)) if not sub.empty else None,
        ]
        rows_out.append(vals)
        for c, v in enumerate(vals, start=1):
            ws.cell(row=r, column=c, value=v)
        _style_data_row(ws, r, 1, len(headers), best=False)
        r += 1
    data_end = r - 1

    # Charts for overall F/U/P means
    r += 1
    r = _write_section_title(ws, r, "Weight effects on mean Fidelity / Utility / Privacy")
    if data_end >= data_start and any(
        ws.cell(row=i, column=5).value is not None for i in range(data_start, data_end + 1)
    ):
        cats = Reference(ws, min_col=1, min_row=data_start, max_row=data_end)
        bar = BarChart()
        bar.type = "col"
        bar.grouping = "clustered"
        bar.title = "Mean F / U / P by objective configuration"
        bar.y_axis.title = "Mean score"
        data = Reference(ws, min_col=5, max_col=7, min_row=4, max_row=data_end)
        bar.add_data(data, titles_from_data=True)
        bar.set_categories(cats)
        bar.dataLabels = DataLabelList()
        bar.dataLabels.showVal = True
        bar.width = 18
        bar.height = 10
        ws.add_chart(bar, f"A{r}")

        bar_w = BarChart()
        bar_w.type = "col"
        bar_w.grouping = "clustered"
        bar_w.title = "Objective weights (wF, wU, wP)"
        data_w = Reference(ws, min_col=2, max_col=4, min_row=4, max_row=data_end)
        bar_w.add_data(data_w, titles_from_data=True)
        bar_w.set_categories(cats)
        bar_w.dataLabels = DataLabelList()
        bar_w.dataLabels.showVal = True
        bar_w.width = 16
        bar_w.height = 9
        ws.add_chart(bar_w, f"J{r}")
    else:
        ws.cell(row=r, column=1, value="Overall charts omitted — insufficient results.")

    _autosize(ws, max_width=32)


def export_workbook(
    results_root: Path | None = None,
    output_path: Path | None = None,
) -> Path:
    """Build ``objective_weight_comparison.xlsx`` from existing CSV/JSON outputs."""
    root = Path(results_root) if results_root else default_results_root()
    root.mkdir(parents=True, exist_ok=True)
    out = Path(output_path) if output_path else root / "objective_weight_comparison.xlsx"

    # Template is read-only reference; never write to it.
    if not TEMPLATE_XLSX.is_file():
        raise FileNotFoundError(
            f"Missing format template {TEMPLATE_XLSX}. "
            "Expected Sampler_results_10_seeds.xlsx under objective_weight_sensitivity/templates/."
        )

    tables = load_sensitivity_tables(root)
    wb = Workbook()
    # Remove default sheet; Overall_Comparison inserted first
    default = wb.active
    wb.remove(default)

    _build_overall_sheet(wb, tables)
    for cfg in PRESET_NAMES:
        sheet_name, label = SHEET_META[cfg]
        _build_config_sheet(
            wb,
            config=cfg,
            sheet_name=sheet_name,
            label=label,
            tables=tables,
        )

    if out.resolve() == TEMPLATE_XLSX.resolve():
        raise ValueError("Refusing to overwrite the Sampler_results_10_seeds template.")

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=str,
        default=None,
        help="Sensitivity results directory (default results/objective_weight_sensitivity)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Workbook path (default <results-root>/objective_weight_comparison.xlsx)",
    )
    args = p.parse_args(argv)
    root = Path(args.results_root) if args.results_root else default_results_root()
    out = Path(args.output) if args.output else None
    path = export_workbook(results_root=root, output_path=out)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
