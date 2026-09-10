#!/usr/bin/env python3
"""Write energy_efficiency HPO notation + Excel (regression / R² utility)."""

from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from datasets.loader import load_train_test  # noqa: E402

DIR = Path(__file__).resolve().parent
DATASET_KEY = "energy_efficiency"
DATASET_TITLE = "Energy Efficiency"
GENERATORS = [
    ("gaussian_copula", "GaussianCopula"),
    ("copulagan", "CopulaGAN"),
    ("ctgan", "CTGAN"),
    ("tvae", "TVAE"),
    ("ctabgan", "CTAB-GAN+"),
    ("wgan_gp", "WGAN-GP"),
]

FILL_DARK = PatternFill("solid", fgColor="1F4E79")
FILL_MAIN = PatternFill("solid", fgColor="2E75B6")
FILL_SUB = PatternFill("solid", fgColor="BDD7EE")
FILL_RANK1 = PatternFill("solid", fgColor="C6EFCE")
FONT_WHITE = Font(name="Calibri", bold=True, color="FFFFFF", size=10)
FONT_DARK = Font(name="Calibri", bold=True, color="1F4E79", size=10)
FONT_TITLE = Font(name="Calibri", bold=True, size=14, color="1F4E79")
FONT_SECTION = Font(name="Calibri", bold=True, size=12, color="1F4E79")
FONT_BODY = Font(name="Calibri", size=10)
THIN = Border(
    left=Side(style="thin"),
    right=Side(style="thin"),
    top=Side(style="thin"),
    bottom=Side(style="thin"),
)
CENTER = Alignment(horizontal="center", vertical="center", wrap_text=True)
LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)

PREFERRED_PARAM_ORDER = [
    "epochs",
    "batch_size",
    "generator_lr",
    "discriminator_lr",
    "generator_dim",
    "discriminator_dim",
    "discriminator_steps",
    "pac",
    "default_distribution",
    "embedding_dim",
    "compress_dims",
    "decompress_dims",
    "l2scale",
    "loss_factor",
    "enforce_min_max_values",
    "latent_dim",
    "hidden_dim",
    "lr",
    "n_critic",
    "lambda_gp",
    "random_dim",
    "num_channels",
    "class_dim",
]


def make_pipe(X, reg):
    X = X.copy()
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X.columns if c not in cat]
    tf = []
    if num:
        tf.append(("num", StandardScaler(), num))
    if cat:
        tf.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
        )
    return Pipeline([("pre", ColumnTransformer(tf, remainder="drop")), ("reg", reg)]), X


REGS = {
    "lr": LinearRegression(),
    "rf": RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=1
    ),
    "dt": DecisionTreeRegressor(max_depth=10, random_state=42),
}


def score_split(X_tr, y_tr, X_te, y_te):
    out = {}
    for name, reg0 in REGS.items():
        pipe, X_tr2 = make_pipe(X_tr, reg0)
        _, X_te2 = make_pipe(X_te, reg0)
        X_te2 = X_te2.reindex(columns=X_tr2.columns)
        pipe.fit(X_tr2, y_tr)
        pred = pipe.predict(X_te2)
        out[name] = float(r2_score(y_te, pred))
    return out


def r4(x):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), 4)


def main() -> None:
    missing = [k for k, _ in GENERATORS if not (DIR / k / "completed.json").is_file()]
    if missing:
        raise SystemExit(f"Incomplete generators: {missing}")

    train_df, test_df, spec = load_train_test(DATASET_KEY)
    target = spec.require_target()

    X_real = train_df.drop(columns=[target])
    y_real = pd.to_numeric(train_df[target], errors="coerce")
    X_test = test_df.drop(columns=[target])
    y_test = pd.to_numeric(test_df[target], errors="coerce")
    trtr = score_split(X_real, y_real, X_test, y_test)
    trtr_r2_mean = float(np.mean(list(trtr.values())))

    rows = []
    all_param_keys: list[str] = []
    for key, display in GENERATORS:
        d = DIR / key
        metrics = json.loads((d / "metrics.json").read_text())
        params = json.loads((d / "best_params.json").read_text())
        timing = json.loads((d / "timing.json").read_text())
        completed = json.loads((d / "completed.json").read_text())
        synth = pd.read_csv(d / "best_synthetic.csv")
        for c in train_df.columns:
            if c not in synth.columns:
                synth[c] = np.nan
        synth = synth[train_df.columns].copy()
        for c in train_df.columns:
            if pd.api.types.is_numeric_dtype(train_df[c]):
                synth[c] = pd.to_numeric(synth[c], errors="coerce")
            else:
                synth[c] = synth[c].astype(str)

        X_s = synth.drop(columns=[target])
        y_s = pd.to_numeric(synth[target], errors="coerce")
        tstr = score_split(X_s, y_s, X_test, y_test)
        tstr_r2_mean = float(np.mean(list(tstr.values())))
        r2_gap = max(trtr_r2_mean - tstr_r2_mean, 0.0)
        util_r2 = 1.0 - float(np.clip(r2_gap, 0.0, 1.0))
        F = float(metrics["fidelity"])
        P = float(metrics["privacy_risk"])
        obj = 0.4 * F + 0.4 * util_r2 - 0.2 * P
        for k in params:
            if k not in all_param_keys:
                all_param_keys.append(k)
        rows.append(
            {
                "key": key,
                "generator": display,
                "obj": obj,
                "fidelity": F,
                "utility_r2": util_r2,
                "privacy_risk": P,
                "ks": float(metrics["ks_similarity"]),
                "corr": float(metrics["corr_similarity"]),
                "tstr_r2": tstr_r2_mean,
                "trtr_r2": trtr_r2_mean,
                "r2_gap": r2_gap,
                "tstr_r2_lr": tstr["lr"],
                "tstr_r2_rf": tstr["rf"],
                "tstr_r2_dt": tstr["dt"],
                "trtr_r2_lr": trtr["lr"],
                "trtr_r2_rf": trtr["rf"],
                "trtr_r2_dt": trtr["dt"],
                "mia_auc": float(metrics["mia_auc"]),
                "nndr": float(metrics["nndr"]),
                "dcr": float(metrics.get("dcr", np.nan)),
                "utility_hpo": float(metrics["utility"]),
                "tstr_hpo": float(metrics["mean_tstr_f1"]),
                "trtr_hpo": float(metrics["mean_trtr_f1"]),
                "gap_hpo": float(metrics["mean_f1_gap"]),
                "obj_hpo": float(
                    completed.get("best_value", metrics.get("objective", np.nan))
                ),
                "time_s": float(timing.get("elapsed_sec", 0)),
                "params": params,
            }
        )

    rows.sort(key=lambda r: r["obj"], reverse=True)
    param_cols = [k for k in PREFERRED_PARAM_ORDER if k in all_param_keys]
    param_cols += [k for k in all_param_keys if k not in param_cols]

    wb = openpyxl.Workbook()
    ws0 = wb.active
    ws0.title = "Metric Guide"
    ws0.merge_cells("A1:E1")
    ws0["A1"] = f"{DATASET_TITLE} — Objective vs Sub-metrics (Regression)"
    ws0["A1"].font = FONT_TITLE
    ws0.merge_cells("A3:E3")
    ws0["A3"] = "OBJECTIVE FORMULA (maximize)"
    ws0["A3"].font = FONT_SECTION
    ws0.merge_cells("A4:E4")
    ws0["A4"] = "Objective = 0.4 × Fidelity  +  0.4 × Utility (R²)  −  0.2 × Privacy Risk"
    ws0["A4"].font = Font(name="Calibri", bold=True, size=11)

    role_headers = ["COLOR / ROLE", "Meaning", "In Objective?", "Weight", "Notes"]
    role_rows = [
        ["OBJECTIVE", "Final scalar score used for ranking", "YES — result", "—", "Sort generators by this"],
        ["MAIN — Fidelity", "Statistical similarity", "YES", "+0.4", "Higher better"],
        ["MAIN — Utility (R²)", "ML usefulness via R²", "YES", "+0.4", "Higher better"],
        ["MAIN — Privacy Risk", "Leakage / memorization risk", "YES", "−0.2", "Higher worse (subtracted)"],
        ["SUB → Fidelity", "KS Similarity, Corr Similarity", "NO", "—", "Support / explain Fidelity"],
        ["SUB → Utility", "TSTR/TRTR R², Gap, per-regressor", "NO", "—", "Support / explain Utility"],
        ["SUB → Privacy", "MIA AUC, NNDR, DCR", "NO", "—", "Support / explain Privacy Risk"],
    ]
    for c, h in enumerate(role_headers, 1):
        cell = ws0.cell(6, c, h)
        cell.fill = FILL_DARK
        cell.font = FONT_WHITE
        cell.border = THIN
    for i, row in enumerate(role_rows):
        for c, v in enumerate(row, 1):
            cell = ws0.cell(7 + i, c, v)
            cell.border = THIN
            cell.font = FONT_BODY
            if i == 0:
                cell.fill = FILL_DARK
                cell.font = FONT_WHITE
            elif i <= 3:
                cell.fill = FILL_MAIN
                cell.font = FONT_WHITE
            else:
                cell.fill = FILL_SUB
                cell.font = FONT_DARK
    for col in range(1, 6):
        ws0.column_dimensions[get_column_letter(col)].width = 28

    # Summary Ranking
    ws1 = wb.create_sheet("Summary Ranking", 1)
    ws1["A1"] = f"{DATASET_TITLE} — Generator Ranking (by Objective)"
    ws1["A1"].font = FONT_TITLE
    ws1.merge_cells("A1:G1")
    headers = [
        "Rank",
        "Generator",
        "Objective",
        "Fidelity",
        "Utility (R²)",
        "Privacy Risk",
        "Time (min)",
    ]
    for c, h in enumerate(headers, 1):
        cell = ws1.cell(3, c, h)
        cell.fill = FILL_DARK
        cell.font = FONT_WHITE
        cell.border = THIN
        cell.alignment = CENTER
    for i, r in enumerate(rows):
        vals = [
            i + 1,
            r["generator"],
            r4(r["obj"]),
            r4(r["fidelity"]),
            r4(r["utility_r2"]),
            r4(r["privacy_risk"]),
            r4(r["time_s"] / 60.0),
        ]
        for c, v in enumerate(vals, 1):
            cell = ws1.cell(4 + i, c, v)
            cell.border = THIN
            cell.alignment = CENTER if c != 2 else LEFT
            if i == 0:
                cell.fill = FILL_RANK1
    for col in range(1, 8):
        ws1.column_dimensions[get_column_letter(col)].width = 16
    ws1.column_dimensions["B"].width = 18

    # Performance Metrics
    ws2 = wb.create_sheet("Performance Metrics", 2)
    ws2["A1"] = f"{DATASET_TITLE} — Detailed Performance Metrics"
    ws2["A1"].font = FONT_TITLE
    ws2.merge_cells("A1:R1")
    perf_headers = [
        "Rank",
        "Generator",
        "Objective",
        "Fidelity",
        "Utility(R²)",
        "PrivacyRisk",
        "KS",
        "Corr",
        "TSTR R²",
        "TRTR R²",
        "R² Gap",
        "TSTR LR",
        "TSTR RF",
        "TSTR DT",
        "MIA AUC",
        "NNDR",
        "DCR",
        "HPO Obj",
    ]
    for c, h in enumerate(perf_headers, 1):
        cell = ws2.cell(3, c, h)
        cell.fill = FILL_DARK
        cell.font = FONT_WHITE
        cell.border = THIN
        cell.alignment = CENTER
    for i, r in enumerate(rows):
        vals = [
            i + 1,
            r["generator"],
            r4(r["obj"]),
            r4(r["fidelity"]),
            r4(r["utility_r2"]),
            r4(r["privacy_risk"]),
            r4(r["ks"]),
            r4(r["corr"]),
            r4(r["tstr_r2"]),
            r4(r["trtr_r2"]),
            r4(r["r2_gap"]),
            r4(r["tstr_r2_lr"]),
            r4(r["tstr_r2_rf"]),
            r4(r["tstr_r2_dt"]),
            r4(r["mia_auc"]),
            r4(r["nndr"]),
            r4(r["dcr"]),
            r4(r["obj_hpo"]),
        ]
        for c, v in enumerate(vals, 1):
            cell = ws2.cell(4 + i, c, v)
            cell.border = THIN
            cell.alignment = CENTER if c != 2 else LEFT
            if i == 0:
                cell.fill = FILL_RANK1
    for col in range(1, len(perf_headers) + 1):
        ws2.column_dimensions[get_column_letter(col)].width = 12
    ws2.column_dimensions["B"].width = 16

    # Best Hyperparameters
    ws3 = wb.create_sheet("Best Hyperparameters", 3)
    ws3["A1"] = f"{DATASET_TITLE} — Best Hyperparameters (Optuna)"
    ws3["A1"].font = FONT_TITLE
    hp_headers = ["Rank", "Generator"] + param_cols
    for c, h in enumerate(hp_headers, 1):
        cell = ws3.cell(3, c, h)
        cell.fill = FILL_DARK
        cell.font = FONT_WHITE
        cell.border = THIN
    for i, r in enumerate(rows):
        ws3.cell(4 + i, 1, i + 1).border = THIN
        ws3.cell(4 + i, 2, r["generator"]).border = THIN
        for j, pk in enumerate(param_cols):
            val = r["params"].get(pk, "")
            if isinstance(val, (list, tuple)):
                val = str(list(val))
            cell = ws3.cell(4 + i, 3 + j, val)
            cell.border = THIN
            if i == 0:
                cell.fill = FILL_RANK1
        if i == 0:
            ws3.cell(4 + i, 1).fill = FILL_RANK1
            ws3.cell(4 + i, 2).fill = FILL_RANK1
    for col in range(1, len(hp_headers) + 1):
        ws3.column_dimensions[get_column_letter(col)].width = 16

    xlsx_path = DIR / f"{DATASET_KEY}_HPO_results.xlsx"
    wb.save(xlsx_path)

    lines = [
        "=" * 80,
        f"      {DATASET_TITLE.upper()} — HYPERPARAMETER TUNING RESULTS (Notation File)",
        "=" * 80,
        f"Dataset: {DATASET_TITLE}",
        "Task: Regression (Y1)",
        "Tuning Framework: Optuna (TPE Sampler, MedianPruner)",
        "Trials per Generator: 20",
        "Sampling: max_samples=1000, seed=42 (random subsample for regression)",
        "Train/Test sizes after sampling+split: 800 / 200",
        "Preprocessing: impute (median/mode) + categorical label-encoding (train-fitted)",
        "Objective (report ranking): MAXIMIZE  0.4×Fidelity + 0.4×Utility(R²) − 0.2×PrivacyRisk",
        "",
        "SCOPE: COMPLETE with 6 generators",
        "  Included: GaussianCopula, CopulaGAN, CTGAN, TVAE, CTAB-GAN+, WGAN-GP",
        "  Excluded: ForestDiffusion, TabDDPM",
        "",
        "=" * 80,
        "                RESULTS RANKED BY R² OBJECTIVE (Best → Worst)",
        "=" * 80,
        "",
    ]
    for i, r in enumerate(rows):
        lines.append(f"#{i+1}  {r['generator']:<28s} Objective(R²): {r['obj']:.4f}")
        lines.append("    " + "─" * 49)
        lines.append(
            f"    Fidelity:      {r['fidelity']:.4f}    |  KS Sim: {r['ks']:.4f}   Corr Sim: {r['corr']:.4f}"
        )
        lines.append(
            f"    Utility(R²):   {r['utility_r2']:.4f}    |  TSTR R²: {r['tstr_r2']:.4f}   TRTR R²: {r['trtr_r2']:.4f}"
        )
        lines.append(
            f"    Privacy Risk:  {r['privacy_risk']:.4f}    |  MIA AUC: {r['mia_auc']:.4f}   NNDR: {r['nndr']:.4f}"
        )
        lines.append(f"    Time: {r['time_s']:.0f}s ({r['time_s']/60:.1f} min)")
        lines.append(
            f"    Reference HPO objective: {r['obj_hpo']:.4f}  |  Utility(HPO): {r['utility_hpo']:.4f}"
        )
        lines.append("")
        lines.append("    Best Hyperparameters:")
        for pk in param_cols:
            if pk in r["params"]:
                lines.append(f"      {pk:<20s}= {r['params'][pk]}")
        lines.append("")

    lines.extend(
        [
            "=" * 80,
            "NOTES",
            "=" * 80,
            "  Utility uses Train-on-Synthetic-Test-on-Real (TSTR) R² vs TRTR R² gap.",
            "  Regressors in report: LinearRegression, RandomForest, DecisionTree.",
            "  ForestDiffusion / TabDDPM: Not included in this 6-generator sweep.",
            f"  Generated: {date.today()} | Framework: Optuna TPE | Dataset: {DATASET_KEY} | Generators: 6",
            "",
        ]
    )
    txt_path = DIR / f"{DATASET_KEY}_HPO_notation.txt"
    txt_path.write_text("\n".join(lines))
    print(f"Wrote {xlsx_path}")
    print(f"Wrote {txt_path}")
    for i, r in enumerate(rows):
        print(
            f"  #{i+1} {r['generator']:<14s} Obj={r['obj']:.4f} "
            f"F={r['fidelity']:.4f} Ur2={r['utility_r2']:.4f} P={r['privacy_risk']:.4f}"
        )


if __name__ == "__main__":
    main()
