"""Utility evaluation — mean F1 gap (TRTR − TSTR) → ``utility`` ∈ [0, 1].

``utility = 1 − clip(mean_f1_gap, 0, 1)`` so higher is better for the Optuna
objective. Classifiers are trained on synthetic (TSTR) or real train (TRTR)
and always evaluated on the held-out real test set (no leakage).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .preprocess import align_columns, clip01


_CLASSIFIER_FACTORY = {
    "logistic_regression": lambda: LogisticRegression(max_iter=500, solver="lbfgs"),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=1
    ),
    "naive_bayes": lambda: GaussianNB(),
    "knn": lambda: KNeighborsClassifier(n_neighbors=5),
    "mlp": lambda: MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=200, random_state=42
    ),
    "svm_rbf": lambda: SVC(kernel="rbf", probability=False, random_state=42),
    "extra_trees": lambda: ExtraTreesClassifier(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=1
    ),
    "adaboost": lambda: AdaBoostClassifier(n_estimators=50, random_state=42),
    "gradient_boosting": lambda: GradientBoostingClassifier(random_state=42),
    "decision_tree": lambda: DecisionTreeClassifier(max_depth=10, random_state=42),
}

# Fast default subset for HPO trials (full list still available via YAML)
_FAST_CLASSIFIERS = [
    "logistic_regression",
    "random_forest",
    "decision_tree",
]


def _make_pipeline(X: pd.DataFrame, clf) -> Pipeline:
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            )
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("clf", clf)])


def _f1_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    clf_name: str,
) -> float:
    if clf_name not in _CLASSIFIER_FACTORY:
        raise KeyError(clf_name)
    if target not in train_df.columns or target not in test_df.columns:
        return 0.0

    X_tr = train_df.drop(columns=[target])
    y_tr = train_df[target]
    X_te = test_df.drop(columns=[target])
    y_te = test_df[target]

    # Align label space
    le = LabelEncoder()
    y_all = pd.concat([y_tr.astype(str), y_te.astype(str)], axis=0)
    le.fit(y_all)
    y_tr_e = le.transform(y_tr.astype(str))
    y_te_e = le.transform(y_te.astype(str))

    # Drop classes missing from train
    train_classes = set(y_tr_e)
    mask = np.array([c in train_classes for c in y_te_e])
    if mask.sum() < 2 or len(train_classes) < 2:
        return 0.0

    try:
        pipe = _make_pipeline(X_tr, _CLASSIFIER_FACTORY[clf_name]())
        pipe.fit(X_tr, y_tr_e)
        pred = pipe.predict(X_te.iloc[mask])
        average = "binary" if len(train_classes) == 2 else "macro"
        return float(f1_score(y_te_e[mask], pred, average=average, zero_division=0))
    except Exception:
        return 0.0


def evaluate_utility(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_test: pd.DataFrame,
    metadata: Any = None,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Return utility metrics; primary ``utility`` ∈ [0, 1] (higher better)."""
    cfg = cfg or {}
    util_cfg = cfg.get("utility", {}) if isinstance(cfg, Mapping) else {}

    target = getattr(metadata, "target", None)
    task = getattr(metadata, "task", "classification")
    if not target:
        return {"utility": 0.0, "mean_f1_gap": 1.0, "mean_tstr_f1": 0.0, "mean_trtr_f1": 0.0}

    if task == "regression":
        return _evaluate_utility_regression(real_train, synthetic, real_test, target)

    classifiers = list(util_cfg.get("classifiers") or _FAST_CLASSIFIERS)
    # Keep only known names
    classifiers = [c for c in classifiers if c in _CLASSIFIER_FACTORY]
    if not classifiers:
        classifiers = list(_FAST_CLASSIFIERS)

    real_tr, synth = align_columns(real_train, synthetic)
    _, real_te = align_columns(real_train, real_test)

    gaps = []
    tstr_scores = []
    trtr_scores = []
    for name in classifiers:
        trtr = _f1_on_split(real_tr, real_te, target, name)
        tstr = _f1_on_split(synth, real_te, target, name)
        gaps.append(trtr - tstr)
        trtr_scores.append(trtr)
        tstr_scores.append(tstr)

    mean_gap = float(np.mean(gaps)) if gaps else 1.0
    # Higher utility when the F1 gap is small (TSTR close to TRTR).
    utility = clip01(1.0 - max(mean_gap, 0.0))

    return {
        "utility": utility,
        "mean_f1_gap": float(mean_gap),
        "mean_tstr_f1": float(np.mean(tstr_scores)) if tstr_scores else 0.0,
        "mean_trtr_f1": float(np.mean(trtr_scores)) if trtr_scores else 0.0,
    }


def _evaluate_utility_regression(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_test: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    """R² gap utility for regression datasets."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score

    real_tr, synth = align_columns(real_train, synthetic)
    _, real_te = align_columns(real_train, real_test)

    def _r2(train_df: pd.DataFrame) -> float:
        X_tr = train_df.drop(columns=[target]).select_dtypes(include=[np.number])
        y_tr = pd.to_numeric(train_df[target], errors="coerce")
        X_te = real_te.drop(columns=[target]).select_dtypes(include=[np.number])
        y_te = pd.to_numeric(real_te[target], errors="coerce")
        cols = [c for c in X_tr.columns if c in X_te.columns]
        if not cols or y_tr.isna().all() or y_te.isna().all():
            return 0.0
        mask = y_te.notna()
        try:
            model = RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=42, n_jobs=1
            )
            model.fit(X_tr[cols].fillna(0.0), y_tr.fillna(y_tr.median()))
            pred = model.predict(X_te[cols].fillna(0.0))
            return float(r2_score(y_te[mask], pred[mask.to_numpy()]))
        except Exception:
            return 0.0

    trtr = _r2(real_tr)
    tstr = _r2(synth)
    gap = trtr - tstr
    utility = clip01(1.0 - max(gap, 0.0))
    return {
        "utility": utility,
        "mean_f1_gap": float(gap),  # stores R² gap for regression
        "mean_tstr_f1": float(tstr),
        "mean_trtr_f1": float(trtr),
    }
