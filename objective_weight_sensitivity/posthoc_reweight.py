#!/usr/bin/env python3
"""Post-hoc reweight O1 trials under O2–O5 with F = SDV QualityReport (KS excluded).

Prefers ``sdv_rescore_cache.csv`` (re-scored SDV). Falls back to trial export
``sdv_quality`` when present. Never uses KS/corr (or legacy composite fidelity)
as the objective fidelity term.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from scipy import stats
from statsmodels.stats.multitest import multipletests

_ROOT = Path(__file__).resolve().parents[1]
OUT = _ROOT / "results/objective_weight_sensitivity/tables/o1_posthoc_reweight"
CACHE = OUT / "sdv_rescore_cache.csv"
EXPORT = OUT / "O1_trials_with_fidelity_components.csv"
SENS_TRIALS = _ROOT / "results/objective_weight_sensitivity/all_trials.csv"

WEIGHTS = {
    "O1": dict(fidelity=0.400, utility=0.400, privacy_risk=0.200),
    "O2": dict(fidelity=0.200, utility=0.600, privacy_risk=0.200),
    "O3": dict(fidelity=0.600, utility=0.200, privacy_risk=0.200),
    "O4": dict(fidelity=0.300, utility=0.300, privacy_risk=0.400),
    "O5": dict(fidelity=0.333, utility=0.333, privacy_risk=0.334),
}
LABELS = {
    "O1": "O1 Original 40:40:20",
    "O2": "O2 Utility-focused",
    "O3": "O3 Fidelity-focused",
    "O4": "O4 Privacy-focused",
    "O5": "O5 Equal weighting",
}
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
SAMPLER_LABEL = {
    "standard_tpe": "Standard TPE",
    "multivariate_tpe": "Multivariate TPE",
    "gp": "Gaussian Process",
}
SHORT = {"standard_tpe": "ST", "multivariate_tpe": "MT", "gp": "GP"}

NAVY, BLUE, ROW, BEST, WHITE, REF = (
    "1F4E79",
    "2E75B6",
    "DDEBF7",
    "C6EFCE",
    "FFFFFF",
    "FCE4D6",
)
THIN = Border(
    left=Side(style="thin", color="B0B0B0"),
    right=Side(style="thin", color="B0B0B0"),
    top=Side(style="thin", color="B0B0B0"),
    bottom=Side(style="thin", color="B0B0B0"),
)


def _font(bold: bool = False, color: str = "000000", size: int = 10) -> Font:
    return Font(name="Calibri", bold=bold, color=color, size=size)


def _fill(h: str) -> PatternFill:
    return PatternFill("solid", fgColor=h)


CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)


def j_of(row: pd.Series, w: dict) -> float:
    return (
        float(w["fidelity"]) * float(row["fidelity"])
        + float(w["utility"]) * float(row["utility"])
        - float(w["privacy_risk"]) * float(row["privacy_risk"])
    )


def load_sdv_trials() -> tuple[pd.DataFrame, str]:
    """Return COMPLETE trials with F = SDV only, plus a source description."""
    frames: list[pd.DataFrame] = []
    notes: list[str] = []

    if CACHE.is_file():
        c = pd.read_csv(CACHE)
        c = c.loc[c["error"].isna() | (c["error"].astype(str) == "nan")].copy()
        if "sdv_unavailable" in c.columns:
            c = c.loc[~c["sdv_unavailable"].fillna(True).astype(bool)]
        c = c.loc[c["sdv_quality"].notna() & c["utility"].notna() & c["privacy_risk"].notna()]
        if not c.empty:
            c = c.copy()
            c["fidelity"] = c["sdv_quality"].astype(float)
            c["state"] = "COMPLETE"
            keep = [
                "dataset",
                "generator",
                "sampler",
                "seed",
                "trial_number",
                "state",
                "fidelity",
                "sdv_quality",
                "utility",
                "privacy_risk",
            ]
            frames.append(c[[x for x in keep if x in c.columns]])
            notes.append(f"sdv_rescore_cache.csv ({len(c)} trials)")

    if EXPORT.is_file():
        e = pd.read_csv(EXPORT)
        e = e.loc[e["state"].eq("COMPLETE") if "state" in e.columns else slice(None)].copy()
        e = e.loc[e["sdv_quality"].notna() & e["utility"].notna() & e["privacy_risk"].notna()]
        if not e.empty:
            e = e.copy()
            e["fidelity"] = e["sdv_quality"].astype(float)
            keep = [
                "dataset",
                "generator",
                "sampler",
                "seed",
                "trial_number",
                "state",
                "fidelity",
                "sdv_quality",
                "utility",
                "privacy_risk",
            ]
            frames.append(e[[x for x in keep if x in e.columns]])
            notes.append(f"O1_trials_with_fidelity_components.csv sdv_nonnull ({len(e)} trials)")

    # Sensitivity pilot (already sdv_only) — Standard TPE, seed 42
    if SENS_TRIALS.is_file():
        s = pd.read_csv(SENS_TRIALS)
        s = s.loc[s["config"].eq("O1") & s["state"].eq("COMPLETE")].copy()
        s = s.loc[s["sdv_quality"].notna() & s["utility"].notna() & s["privacy_risk"].notna()]
        if not s.empty:
            s = s.copy()
            s["fidelity"] = s["sdv_quality"].astype(float)
            s["sampler"] = "standard_tpe"
            keep = [
                "dataset",
                "generator",
                "sampler",
                "seed",
                "trial_number",
                "state",
                "fidelity",
                "sdv_quality",
                "utility",
                "privacy_risk",
            ]
            frames.append(s[[x for x in keep if x in s.columns]])
            notes.append(f"sensitivity all_trials O1 ({len(s)} trials, ST seed42)")

    if not frames:
        raise FileNotFoundError(
            "No SDV-scored trials found. Run rescore_sdv_fidelity.py or use "
            "results/objective_weight_sensitivity/all_trials.csv"
        )

    df = pd.concat(frames, ignore_index=True)
    # Prefer rescore/export over sensitivity when keys collide
    df = df.drop_duplicates(
        subset=["dataset", "generator", "sampler", "seed", "trial_number"],
        keep="first",
    )
    df["fidelity"] = df["fidelity"].astype(float)
    df["utility"] = df["utility"].astype(float)
    df["privacy_risk"] = df["privacy_risk"].astype(float)
    source = " + ".join(notes)
    return df, source


def build_best(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(["dataset", "generator", "sampler", "seed"]):
        dataset, generator, sampler, seed = keys
        for preset, w in WEIGHTS.items():
            vals = sub.apply(lambda r: j_of(r, w), axis=1)
            idx = vals.idxmax()
            r = sub.loc[idx]
            rows.append(
                {
                    "dataset": dataset,
                    "generator": generator,
                    "sampler": sampler,
                    "seed": int(seed),
                    "n_complete": int(len(sub)),
                    "preset": preset,
                    "best_trial_number": int(r["trial_number"]),
                    "best_J": float(vals.loc[idx]),
                    "best_F": float(r["fidelity"]),
                    "best_U": float(r["utility"]),
                    "best_P": float(r["privacy_risk"]),
                }
            )
    return pd.DataFrame(rows)


def _style_header(ws, r, ncols, color=NAVY):
    for c in range(1, ncols + 1):
        cell = ws.cell(r, c)
        cell.fill = _fill(color)
        cell.font = _font(True, WHITE)
        cell.alignment = CENTER
        cell.border = THIN


def write_df(ws, title, note, df):
    ws.sheet_view.showGridLines = False
    ncols = max(len(df.columns), 3)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=ncols)
    ws.cell(1, 1, title).font = _font(True, NAVY, 14)
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=ncols)
    ws.cell(2, 1, note).font = _font(False, NAVY, 9)
    ws.cell(2, 1).fill = _fill(REF)
    r = 4
    for c, h in enumerate(df.columns, 1):
        ws.cell(r, c, h)
    _style_header(ws, r, len(df.columns))
    r += 1
    jcols = [c for c in df.columns if "mean J" in c]
    for _, row in df.iterrows():
        best_cols = set()
        if jcols:
            nums = {c: row[c] for c in jcols if pd.notna(row[c])}
            if nums:
                mx = max(nums.values())
                best_cols = {c for c, v in nums.items() if abs(v - mx) < 1e-12}
        for c, col in enumerate(df.columns, 1):
            v = row[col]
            cell = ws.cell(r, c, None if (isinstance(v, float) and pd.isna(v)) else v)
            cell.border = THIN
            cell.alignment = CENTER if c > 1 else LEFT
            if col in best_cols or (col == "Significant" and v == "Yes") or (
                col == "Best sampler"
            ):
                cell.fill = _fill(BEST)
            elif r % 2 == 0:
                cell.fill = _fill(ROW)
            if isinstance(v, float):
                cell.number_format = "0.0000" if abs(v) < 10 else "0.00"
        r += 1
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        w = 12
        for cell in col:
            if cell.value is not None:
                w = max(w, min(len(str(cell.value)) + 2, 36))
        ws.column_dimensions[letter].width = w


def run(out_dir: Path | None = None) -> Path:
    out = Path(out_dir or OUT)
    out.mkdir(parents=True, exist_ok=True)

    trials, source = load_sdv_trials()
    print(f"Loaded {len(trials)} SDV-scored trials from: {source}", flush=True)
    n_studies = trials.groupby(["dataset", "generator", "sampler", "seed"]).ngroups
    print(f"studies_with_sdv={n_studies}", flush=True)

    best = build_best(trials)
    best.to_csv(out / "study_best_under_each_preset.csv", index=False)

    agg_rows = []
    for (preset, sampler, dataset, generator), g in best.groupby(
        ["preset", "sampler", "dataset", "generator"]
    ):
        agg_rows.append(
            {
                "preset": preset,
                "sampler": sampler,
                "dataset": dataset,
                "generator": generator,
                "mean_J": float(g["best_J"].mean()),
                "std_J": float(g["best_J"].std(ddof=1)) if len(g) > 1 else 0.0,
                "mean_F": float(g["best_F"].mean()),
                "mean_U": float(g["best_U"].mean()),
                "mean_P": float(g["best_P"].mean()),
                "n_seeds": int(len(g)),
            }
        )
    agg = pd.DataFrame(agg_rows)
    agg["is_best"] = False
    for _, g in agg.groupby(["preset", "sampler", "dataset"]):
        i = g["mean_J"].idxmax()
        agg.loc[i, "is_best"] = True
    agg.to_csv(out / "generator_mean_std_by_preset.csv", index=False)

    ds_rows = []
    for (preset, sampler, dataset), g in agg.groupby(["preset", "sampler", "dataset"]):
        ds_rows.append(
            {
                "preset": preset,
                "sampler": sampler,
                "dataset": dataset,
                "mean_J": float(g["mean_J"].mean()),
                "mean_F": float(g["mean_F"].mean()),
                "mean_U": float(g["mean_U"].mean()),
                "mean_P": float(g["mean_P"].mean()),
            }
        )
    ds = pd.DataFrame(ds_rows)
    ds.to_csv(out / "dataset_level_mean_by_preset.csv", index=False)

    def rank_tiebreak(vals):
        pref = {"ST": 0, "MT": 1, "GP": 2}
        items = list(vals.items())
        items.sort(key=lambda x: (-x[1], pref[x[0]]))
        ranks = {k: i + 1 for i, (k, _) in enumerate(items)}
        return ranks, items[0][0]

    def dataset_comparison(preset):
        rows = []
        present_ds = [
            d
            for d in ORDER_DS
            if not ds.loc[(ds.preset == preset) & (ds.dataset == d)].empty
        ]
        for dataset in present_ds:
            vals = {}
            ok = True
            for samp, short in [
                ("standard_tpe", "ST"),
                ("multivariate_tpe", "MT"),
                ("gp", "GP"),
            ]:
                sub = ds.loc[
                    (ds.preset == preset)
                    & (ds.sampler == samp)
                    & (ds.dataset == dataset)
                ]
                if sub.empty:
                    ok = False
                    break
                vals[short] = float(sub["mean_J"].iloc[0])
            if not ok:
                continue
            ranks, best_s = rank_tiebreak(vals)
            rows.append(
                {
                    "Dataset": DS_LABEL[dataset],
                    "ST mean J": round(vals["ST"], 4),
                    "ST rank": ranks["ST"],
                    "MT mean J": round(vals["MT"], 4),
                    "MT rank": ranks["MT"],
                    "GP mean J": round(vals["GP"], 4),
                    "GP rank": ranks["GP"],
                    "Best sampler": {
                        "ST": "Standard TPE",
                        "MT": "Multivariate TPE",
                        "GP": "Gaussian Process",
                    }[best_s],
                }
            )
        return pd.DataFrame(rows), present_ds

    def summary_and_stats(preset):
        c, present_ds = dataset_comparison(preset)
        if c.empty or len(present_ds) < 3:
            # Sampler comparison needs ST/MT/GP; fall back to empty-safe frames
            summary = pd.DataFrame(
                columns=[
                    "Sampler",
                    "Dataset-level mean J",
                    "Avg rank (1=best)",
                    "Dataset wins",
                    "Cells with std≥0.10",
                ]
            )
            fried = pd.DataFrame(
                [
                    {
                        "preset": preset,
                        "Test": "Friedman (insufficient ST/MT/GP coverage)",
                        "Statistic (χ²)": np.nan,
                        "df": 2,
                        "p-value": np.nan,
                        "Kendall's W": np.nan,
                        "Significant (α=0.05)": "N/A",
                    }
                ]
            )
            pair = pd.DataFrame(
                columns=[
                    "Comparison",
                    "Mean Δ (A−B)",
                    "Wins (A–B)",
                    "p_raw",
                    "p Holm",
                    "Significant",
                    "Conclusion",
                ]
            )
            return summary, fried, pair, c

        rows = []
        for samp, short in [
            ("standard_tpe", "ST"),
            ("multivariate_tpe", "MT"),
            ("gp", "GP"),
        ]:
            col = f"{short} mean J"
            mean_j = float(c[col].mean())
            avg_rank = float(c[f"{short} rank"].mean())
            wins = int((c["Best sampler"] == SAMPLER_LABEL[samp]).sum())
            g = agg.loc[(agg.preset == preset) & (agg.sampler == samp)]
            high = int((g["std_J"] >= 0.10).sum())
            rows.append(
                {
                    "Sampler": SAMPLER_LABEL[samp],
                    "Dataset-level mean J": round(mean_j, 4),
                    "Avg rank (1=best)": round(avg_rank, 2),
                    "Dataset wins": wins,
                    "Cells with std≥0.10": high,
                }
            )
        summary = pd.DataFrame(rows)

        mat = []
        for samp in ["standard_tpe", "multivariate_tpe", "gp"]:
            vec = [
                float(
                    ds.loc[
                        (ds.preset == preset)
                        & (ds.sampler == samp)
                        & (ds.dataset == d),
                        "mean_J",
                    ].iloc[0]
                )
                for d in present_ds
            ]
            mat.append(vec)
        chi2, p = stats.friedmanchisquare(*mat)
        n, k = len(present_ds), 3
        W = chi2 / (n * (k - 1))
        fried = pd.DataFrame(
            [
                {
                    "preset": preset,
                    "Test": f"Friedman ({n} datasets × 3 samplers)",
                    "Statistic (χ²)": round(float(chi2), 2),
                    "df": 2,
                    "p-value": float(p),
                    "Kendall's W": round(float(W), 3),
                    "Significant (α=0.05)": "Yes" if p < 0.05 else "No",
                }
            ]
        )

        keys = ["standard_tpe", "multivariate_tpe", "gp"]
        pair_rows = []
        raw_p = []
        for (i, a), (j, b) in combinations(list(enumerate(keys)), 2):
            xa = np.array(
                [
                    float(
                        ds.loc[
                            (ds.preset == preset)
                            & (ds.sampler == a)
                            & (ds.dataset == d),
                            "mean_J",
                        ].iloc[0]
                    )
                    for d in present_ds
                ]
            )
            xb = np.array(
                [
                    float(
                        ds.loc[
                            (ds.preset == preset)
                            & (ds.sampler == b)
                            & (ds.dataset == d),
                            "mean_J",
                        ].iloc[0]
                    )
                    for d in present_ds
                ]
            )
            if np.allclose(xa, xb):
                w_stat, w_p = np.nan, 1.0
            else:
                w_stat, w_p = stats.wilcoxon(xa, xb)
            n_a = int((xa > xb).sum())
            n_b = int((xb > xa).sum())
            pair_rows.append(
                {
                    "Comparison": f"{SHORT[a]} vs {SHORT[b]}",
                    "Mean Δ (A−B)": float(np.mean(xa - xb)),
                    "Wins (A–B)": f"{n_a}–{n_b}",
                    "p_raw": float(w_p),
                }
            )
            raw_p.append(float(w_p))
        rej, p_holm, _, _ = multipletests(raw_p, method="holm", alpha=0.05)
        for i, row in enumerate(pair_rows):
            row["p Holm"] = float(p_holm[i])
            row["Significant"] = "Yes" if rej[i] else "No"
            a, b = row["Comparison"].split(" vs ")
            if rej[i]:
                row["Conclusion"] = f"{a} > {b}" if row["Mean Δ (A−B)"] > 0 else f"{b} > {a}"
            else:
                row["Conclusion"] = f"{a} ≈ {b}"
        return summary, fried, pd.DataFrame(pair_rows), c

    gen_overall = []
    for preset in WEIGHTS:
        for gen in ORDER_GEN:
            g = agg.loc[(agg.preset == preset) & (agg.generator == gen)]
            if g.empty:
                continue
            gen_overall.append(
                {
                    "preset": preset,
                    "generator": GEN_LABEL[gen],
                    "Mean J": round(float(g["mean_J"].mean()), 4),
                    "Mean F": round(float(g["mean_F"].mean()), 4),
                    "Mean U": round(float(g["mean_U"].mean()), 4),
                    "Mean P": round(float(g["mean_P"].mean()), 4),
                }
            )
    gen_overall = pd.DataFrame(gen_overall)

    shift_rows = []
    o1 = best.loc[
        best.preset == "O1",
        ["dataset", "generator", "sampler", "seed", "best_trial_number", "best_J"],
    ].rename(columns={"best_trial_number": "trial_O1", "best_J": "J_O1"})
    for preset in ["O2", "O3", "O4", "O5"]:
        other = best.loc[
            best.preset == preset,
            ["dataset", "generator", "sampler", "seed", "best_trial_number", "best_J"],
        ].rename(columns={"best_trial_number": "trial_new", "best_J": "J_new"})
        m = o1.merge(other, on=["dataset", "generator", "sampler", "seed"])
        n = len(m)
        changed = int((m["trial_O1"] != m["trial_new"]).sum()) if n else 0
        shift_rows.append(
            {
                "preset": preset,
                "studies": n,
                "best_trial_changed": changed,
                "pct_changed": round(100 * changed / n, 1) if n else 0.0,
            }
        )
    shift = pd.DataFrame(shift_rows)
    shift.to_csv(out / "best_trial_shift_vs_O1.csv", index=False)

    wb = Workbook()
    wb.remove(wb.active)
    cover = wb.create_sheet("00_Cover", 0)
    cover["A1"] = (
        "Post-hoc reweight of O1 trials under O2–O5 — F = SDV QualityReport (KS excluded)"
    )
    cover["A1"].font = _font(True, NAVY, 14)
    cover.merge_cells("A1:F1")
    notes = [
        f"Source: {source}",
        "Fidelity F = SDV QualityReport only. KS / correlation are diagnostics and do NOT enter J.",
        "J = wF·F + wU·U − wP·P. For each study, argmax trial under each preset weight.",
        "No new HPO — reuses evaluated U/P; F replaced by SDV-only scores.",
        "Studies without an SDV score are omitted (no silent KS/corr fallback).",
        f"Studies in workbook: {n_studies}. Best-trial shifts vs O1: "
        + ", ".join(f"{r.preset} {r.pct_changed}%" for r in shift.itertuples()),
    ]
    for i, n in enumerate(notes, 3):
        cover.cell(i, 1, n)
        cover.merge_cells(start_row=i, start_column=1, end_row=i, end_column=6)
    cover.column_dimensions["A"].width = 110

    write_df(
        wb.create_sheet("Trial_shift_vs_O1"),
        "How often the best trial changes when reweighting O1 points",
        "pct_changed = fraction of studies where argmax trial differs from O1 (F=SDV).",
        shift,
    )
    if not gen_overall.empty:
        write_df(
            wb.create_sheet("Generator_overall"),
            "Mean J / F / U / P by generator under each preset",
            "F = SDV QualityReport. Mean over available datasets×samplers×seeds.",
            gen_overall,
        )

    for preset in WEIGHTS:
        summary, fried, pair, ctab = summary_and_stats(preset)
        summary.to_csv(out / f"{preset}_A_summary.csv", index=False)
        fried.to_csv(out / f"{preset}_B_friedman.csv", index=False)
        pair.to_csv(out / f"{preset}_B_pairwise.csv", index=False)
        ctab.to_csv(out / f"{preset}_C_dataset_mean.csv", index=False)

        w = WEIGHTS[preset]
        write_df(
            wb.create_sheet(f"{preset}_A_Summary"),
            f"{LABELS[preset]} — A. Summary by sampler (post-hoc, F=SDV)",
            f"J={w['fidelity']}F+{w['utility']}U−{w['privacy_risk']}P with F=SDV QualityReport (KS excluded).",
            summary if not summary.empty else pd.DataFrame({"note": ["insufficient sampler coverage"]}),
        )
        ws = wb.create_sheet(f"{preset}_B_Stats")
        write_df(
            ws,
            f"{LABELS[preset]} — B. Statistical tests (dataset-level post-hoc J)",
            "Friedman + pairwise Wilcoxon with Holm correction (when ST/MT/GP all present).",
            fried,
        )
        if not pair.empty:
            r = 8
            ws.cell(r, 1, "Pairwise Wilcoxon + Holm").font = _font(True, NAVY, 12)
            r = 9
            for c, h in enumerate(pair.columns, 1):
                ws.cell(r, c, h)
            _style_header(ws, r, len(pair.columns), BLUE)
            r = 10
            for _, row in pair.iterrows():
                for c, col in enumerate(pair.columns, 1):
                    cell = ws.cell(r, c, row[col])
                    cell.border = THIN
                    if row["Significant"] == "Yes":
                        cell.fill = _fill(BEST)
                r += 1
        write_df(
            wb.create_sheet(f"{preset}_C_Dataset"),
            f"{LABELS[preset]} — C. Dataset-level mean J and ranks (ST / MT / GP)",
            "Post-hoc reweight with F=SDV; omitted datasets lack all three samplers.",
            ctab if not ctab.empty else pd.DataFrame({"note": ["insufficient sampler coverage"]}),
        )

    xlsx = out / "O1_trials_reweighted_O2_O5.xlsx"
    wb.save(xlsx)
    print(f"Wrote {xlsx}", flush=True)
    return xlsx


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
