"""Shared helpers for evaluation adapters."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clip01(x: float) -> float:
    """Clip a scalar to ``[0, 1]``."""
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return 0.0
    return float(np.clip(x, 0.0, 1.0))


def align_columns(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align synthetic columns to the real schema (order + missing cols)."""
    cols = list(real.columns)
    synth = synth.copy()
    for c in cols:
        if c not in synth.columns:
            synth[c] = np.nan
    return real[cols].copy(), synth[cols].copy()


def split_xy(
    df: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def build_feature_matrix(
    real_X: pd.DataFrame,
    *others: pd.DataFrame,
) -> tuple[np.ndarray, ...]:
    """Fit a simple preprocessor on ``real_X`` and transform all frames."""
    cat_cols = [c for c in real_X.columns if not pd.api.types.is_numeric_dtype(real_X[c])]
    num_cols = [c for c in real_X.columns if c not in cat_cols]

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
    if not transformers:
        # All-empty safeguard
        return tuple(np.zeros((len(df), 1)) for df in (real_X, *others))

    pre = ColumnTransformer(transformers, remainder="drop")
    pre.fit(real_X)
    out = [pre.transform(real_X)]
    for df in others:
        # Ensure same columns
        df = df.reindex(columns=real_X.columns)
        out.append(pre.transform(df))
    return tuple(np.asarray(a, dtype=float) for a in out)


def numeric_feature_frame(df: pd.DataFrame, exclude: Iterable[str] = ()) -> pd.DataFrame:
    """Return numeric columns only (exclude target etc.)."""
    exclude = set(exclude)
    cols = [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return df[cols].apply(pd.to_numeric, errors="coerce")
