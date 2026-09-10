"""Train-fitted preprocessing applied before generator training.

Steps (fit on ``train_df``, transform both train and test):
1. Impute missing values (median for continuous numeric, mode for discrete/categorical).
2. Label-encode categorical **and** discrete (low-cardinality integer-like) columns
   to contiguous integer codes ``0 .. K-1``.
3. Optionally standardize continuous numeric feature columns.

Class balancing and row subsampling happen earlier in ``loader.py``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

__all__ = ["fit_transform_train_test"]


def _mode_value(series: pd.Series) -> Any:
    modes = series.mode(dropna=True)
    if len(modes):
        return modes.iloc[0]
    non_null = series.dropna()
    if len(non_null):
        return non_null.iloc[0]
    return 0


def _is_integer_like(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    if pd.api.types.is_integer_dtype(series):
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    vals = non_null.to_numpy(dtype=float)
    return bool(np.allclose(vals, np.round(vals), equal_nan=False))


def _fit_impute_values(
    train_df: pd.DataFrame,
    discrete_cols: set[str],
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for col in train_df.columns:
        s = train_df[col]
        if col in discrete_cols or not pd.api.types.is_numeric_dtype(s):
            values[col] = _mode_value(s)
        else:
            med = s.median()
            values[col] = float(med) if pd.notna(med) else 0.0
    return values


def _impute_frame(df: pd.DataFrame, fill_values: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, fill in fill_values.items():
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(fill)
    return out


def _columns_to_encode(
    train_df: pd.DataFrame,
    *,
    categorical_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    target: str | None,
    task: str,
    max_int_categories: int,
) -> tuple[str, ...]:
    """Columns that must become integer category codes before generator fit.

    Includes:
      - inferred / forced categorical columns (object or low-card discrete)
      - classification target
      - any remaining non-numeric columns
      - integer-like columns with 2..max_int_categories unique values
        (safety net if inference missed them)
    """
    cat_set = set(categorical_columns)
    if target and task == "classification" and target in train_df.columns:
        cat_set.add(target)

    encode: list[str] = []
    for col in train_df.columns:
        s = train_df[col]
        if col in cat_set:
            encode.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(s):
            encode.append(col)
            continue
        if col in numeric_columns:
            # Still encode if clearly discrete and under the cardinality cap.
            nunique = int(s.nunique(dropna=True))
            if (
                max_int_categories > 0
                and _is_integer_like(s)
                and 1 < nunique <= max_int_categories
            ):
                encode.append(col)
            continue
        # Not listed as continuous numeric → encode
        encode.append(col)

    return tuple(dict.fromkeys(encode))


def fit_transform_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    categorical_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
    target: str | None,
    task: str,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply imputation, discrete/categorical encoding, and optional normalization."""
    impute = bool(cfg.get("impute", True))
    encode_categorical = bool(cfg.get("encode_categorical", True))
    normalize_numeric = bool(cfg.get("normalize_numeric", False))
    max_int_categories = int(cfg.get("max_int_categories", 20) or 0)

    train = train_df.copy()
    test = test_df.copy()

    encode_cols = _columns_to_encode(
        train,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        target=target,
        task=task,
        max_int_categories=max_int_categories,
    )
    discrete_set = set(encode_cols)

    if impute:
        fill_values = _fit_impute_values(train, discrete_set)
        train = _impute_frame(train, fill_values)
        test = _impute_frame(test, fill_values)

    if encode_categorical and encode_cols:
        for col in encode_cols:
            if col not in train.columns:
                continue
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.int64,
            )
            # Encode via string labels so mixed object/int discrete cols share one path.
            train_vals = train[[col]].astype(str)
            test_vals = test[[col]].astype(str)
            train[col] = enc.fit_transform(train_vals).ravel().astype(np.int64)
            test[col] = enc.transform(test_vals).ravel().astype(np.int64)

    if normalize_numeric:
        scale_cols = [
            c
            for c in numeric_columns
            if c in train.columns
            and c != target
            and c not in discrete_set
            and pd.api.types.is_numeric_dtype(train[c])
        ]
        if scale_cols:
            scaler = StandardScaler()
            train[scale_cols] = scaler.fit_transform(train[scale_cols])
            test[scale_cols] = scaler.transform(test[scale_cols])

    return train.reset_index(drop=True), test.reset_index(drop=True)
