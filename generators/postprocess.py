"""Snap continuous synthetic values back onto discrete / categorical levels.

Continuous generators (WGAN-GP, TabDDPM) often emit soft floats for
binary/integer targets (e.g. 0.97 instead of 1). Downstream TSTR classifiers
then see hundreds of fake classes and score ~0 utility/accuracy.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def discrete_levels_from_train(
    train_df: pd.DataFrame,
    metadata: Any = None,
) -> dict[str, list[Any]]:
    """Infer discrete columns and their allowed levels from real training data."""
    levels: dict[str, list[Any]] = {}
    cat_cols = list(getattr(metadata, "categorical_columns", ()) or [])
    target = getattr(metadata, "target", None)
    task = getattr(metadata, "task", None)

    if target and task == "classification" and target in train_df.columns and target not in cat_cols:
        cat_cols.append(target)

    for col in cat_cols:
        if col in train_df.columns:
            levels[col] = list(pd.unique(train_df[col].dropna()))

    # Low-cardinality integer columns (binary / ordinal codes)
    for col in train_df.columns:
        if col in levels:
            continue
        s = train_df[col]
        if pd.api.types.is_integer_dtype(s) and int(s.nunique(dropna=True)) <= 20:
            levels[col] = sorted(s.dropna().unique().tolist())

    return levels


def snap_series_to_levels(series: pd.Series, levels: list[Any]) -> pd.Series:
    """Map each value to the nearest allowed discrete level."""
    if not levels:
        return series

    # Prefer numeric nearest-neighbour when all train levels are numeric
    try:
        levels_f = np.asarray([float(x) for x in levels], dtype=float)
        numeric_levels = True
    except (TypeError, ValueError):
        numeric_levels = False

    if numeric_levels:
        vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        fill = float(levels_f[0])
        nan_mask = np.isnan(vals)
        if nan_mask.any():
            vals = vals.copy()
            vals[nan_mask] = fill
        idx = np.abs(vals[:, None] - levels_f[None, :]).argmin(axis=1)
        # Restore original level objects (keeps int 0/1, not float 0.0/1.0)
        snapped_objs = [levels[i] for i in idx]
        return pd.Series(snapped_objs, index=series.index)

    # Non-numeric categories: keep existing labels if valid, else mode
    level_set = {str(x) for x in levels}
    mode = levels[0]
    out = []
    for v in series:
        if pd.isna(v):
            out.append(mode)
        elif str(v) in level_set:
            match = next(x for x in levels if str(x) == str(v))
            out.append(match)
        else:
            out.append(mode)
    return pd.Series(out, index=series.index)


def snap_discrete_columns(
    synth_df: pd.DataFrame,
    levels: Mapping[str, list[Any]],
) -> pd.DataFrame:
    """Return a copy of ``synth_df`` with discrete columns snapped to train levels."""
    out = synth_df.copy()
    for col, allowed in levels.items():
        if col in out.columns and allowed:
            out[col] = snap_series_to_levels(out[col], list(allowed))
    return out


def snap_discrete_from_train(
    synth_df: pd.DataFrame,
    train_df: pd.DataFrame,
    metadata: Any = None,
) -> pd.DataFrame:
    """Convenience: infer levels from train + metadata, then snap."""
    return snap_discrete_columns(synth_df, discrete_levels_from_train(train_df, metadata))
