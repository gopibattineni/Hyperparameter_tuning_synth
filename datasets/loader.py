"""Dataset loader — YAML registry → fixed-seed train/test split.

Generators and Optuna trials must only ever call ``fit`` on ``train_df``.
The hold-out ``test_df`` is reserved for utility evaluation (TSTR / TRTR).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config_utils import CONFIG_ROOT, REPO_ROOT, load_yaml
from .schema import DatasetSpec

__all__ = [
    "DatasetSpec",
    "get_dataset_spec",
    "list_datasets",
    "load_dataset_config",
    "load_raw_dataframe",
    "load_train_test",
]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_dataset_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load the datasets YAML and return the ``datasets`` mapping."""
    path = Path(config_path) if config_path else CONFIG_ROOT / "datasets.yaml"
    raw = load_yaml(path)
    datasets = raw.get("datasets", raw)
    if not isinstance(datasets, dict):
        raise ValueError(f"Invalid datasets config in {path}")
    return datasets


def list_datasets(config_path: Path | str | None = None) -> list[str]:
    """Return sorted dataset keys from the registry."""
    return sorted(load_dataset_config(config_path))


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path.resolve()
    candidate = REPO_ROOT / path
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Dataset file not found: {path_str} (also tried {candidate})"
    )


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------

def _read_csv(path: Path, cfg: Mapping[str, Any]) -> pd.DataFrame:
    sep = cfg.get("sep", ",")
    encoding = cfg.get("encoding", "utf-8")
    return pd.read_csv(path, sep=sep, encoding=encoding)


def _read_excel(path: Path, cfg: Mapping[str, Any]) -> pd.DataFrame:
    sheet = cfg.get("sheet_name", 0)
    return pd.read_excel(path, sheet_name=sheet)


def _read_uci(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Fetch a dataset via ``ucimlrepo`` using ``uci_id`` from YAML."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "ucimlrepo is required for source: uci. pip install ucimlrepo"
        ) from exc

    uci_id = cfg.get("uci_id")
    if uci_id is None:
        raise ValueError("UCI datasets require `uci_id` in datasets.yaml")

    repo = fetch_ucirepo(id=int(uci_id))
    features = repo.data.features.copy()
    targets = repo.data.targets
    if targets is not None and len(targets.columns):
        df = pd.concat([features, targets], axis=1)
    else:
        df = features
    return df


def load_raw_dataframe(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Load a dataframe from csv / excel / uci according to YAML ``source``."""
    source = str(cfg.get("source", "csv")).lower()
    if source == "uci":
        return _read_uci(cfg)
    if "path" not in cfg:
        raise ValueError("Dataset config requires `path` unless source is uci")
    path = _resolve_path(str(cfg["path"]))
    if source == "csv":
        return _read_csv(path, cfg)
    if source in {"excel", "xlsx", "xls"}:
        return _read_excel(path, cfg)
    raise ValueError(f"Unsupported dataset source: {source!r}")


# ---------------------------------------------------------------------------
# Preprocess + schema inference
# ---------------------------------------------------------------------------

def _apply_preprocess(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    out = df.copy()

    drop_cols = list(cfg.get("drop_columns") or [])
    # Drop unnamed junk columns (common in Cancer.csv)
    if cfg.get("drop_unnamed", True):
        drop_cols.extend(c for c in out.columns if str(c).startswith("Unnamed"))
    out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    # Replace sentinel missing tokens before numeric coercion
    na_values = cfg.get("na_values")
    if na_values:
        out = out.replace(list(na_values), np.nan)

    target = cfg.get("target")
    target_map = cfg.get("target_map")
    if target and target_map and target in out.columns:
        mapped = out[target].map(target_map)
        # Keep original where key absent from map
        out[target] = mapped.where(mapped.notna(), out[target])

    # Optional per-column maps
    column_maps = cfg.get("column_maps") or {}
    for col, mapping in column_maps.items():
        if col in out.columns:
            mapped = out[col].map(mapping)
            out[col] = mapped.where(mapped.notna(), out[col])

    if cfg.get("dropna", True):
        out = out.dropna(axis=0).reset_index(drop=True)

    # Strip column name whitespace
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _infer_column_types(
    df: pd.DataFrame,
    target: str | None,
    cfg: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return (feature_columns, categorical_columns, numeric_columns)."""
    forced_cat = set(cfg.get("categorical_columns") or [])
    forced_num = set(cfg.get("numeric_columns") or [])

    feature_cols = [c for c in df.columns if c != target]
    categorical: list[str] = []
    numeric: list[str] = []

    for col in feature_cols:
        if col in forced_cat:
            categorical.append(col)
            continue
        if col in forced_num:
            numeric.append(col)
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Low-cardinality integers may still be categorical if configured
            max_card = int(cfg.get("max_int_categories", 0) or 0)
            nunique = int(df[col].nunique(dropna=True))
            if max_card > 0 and nunique <= max_card:
                categorical.append(col)
            else:
                numeric.append(col)
        else:
            categorical.append(col)

    return tuple(feature_cols), tuple(categorical), tuple(numeric)


def _build_spec(
    key: str,
    cfg: Mapping[str, Any],
    df: pd.DataFrame,
    n_train: int,
    n_test: int,
) -> DatasetSpec:
    target = cfg.get("target")
    if target is not None:
        target = str(target)
        if target not in df.columns:
            raise KeyError(
                f"Target column {target!r} not in dataframe columns: {list(df.columns)}"
            )

    features, cats, nums = _infer_column_types(df, target, cfg)
    path = str(cfg.get("path") or f"uci:{cfg.get('uci_id')}")

    return DatasetSpec(
        key=key,
        display_name=str(cfg.get("display_name", key)),
        path=path,
        task=cfg.get("task", "classification"),  # type: ignore[arg-type]
        target=target,
        test_size=float(cfg.get("test_size", 0.2)),
        seed=int(cfg.get("seed", 42)),
        source=str(cfg.get("source", "csv")),
        categorical_columns=cats,
        numeric_columns=nums,
        feature_columns=features,
        n_rows=int(len(df)),
        n_train=int(n_train),
        n_test=int(n_test),
        extras={
            k: v
            for k, v in cfg.items()
            if k
            not in {
                "display_name",
                "path",
                "task",
                "target",
                "test_size",
                "seed",
                "source",
                "categorical_columns",
                "numeric_columns",
            }
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_dataset_spec(
    dataset_key: str,
    config_path: Path | str | None = None,
    *,
    fit_preprocess: bool = True,
) -> DatasetSpec:
    """Return a :class:`DatasetSpec` (loads data once to infer schema)."""
    train_df, test_df, spec = load_train_test(
        dataset_key, config_path=config_path, return_frames=True
    )
    # frames unused when only the spec is requested
    _ = train_df, test_df, fit_preprocess
    return spec


def load_train_test(
    dataset_key: str,
    config_path: Path | str | None = None,
    *,
    return_frames: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, DatasetSpec]:
    """Load ``(train_df, test_df, DatasetSpec)`` for ``dataset_key``.

    The split is deterministic given ``seed`` / ``test_size`` in YAML.
    Classification tasks use stratified splitting when every class has ≥2 rows.
    """
    datasets = load_dataset_config(config_path)
    if dataset_key not in datasets:
        known = ", ".join(sorted(datasets))
        raise KeyError(f"Unknown dataset {dataset_key!r}. Known: {known}")

    cfg = dict(datasets[dataset_key])
    df = load_raw_dataframe(cfg)
    df = _apply_preprocess(df, cfg)

    if len(df) < 5:
        raise ValueError(
            f"Dataset {dataset_key!r} has only {len(df)} rows after preprocessing."
        )

    test_size = float(cfg.get("test_size", 0.2))
    seed = int(cfg.get("seed", 42))
    target = cfg.get("target")
    task = cfg.get("task", "classification")

    stratify = None
    if task == "classification" and target and target in df.columns:
        counts = df[target].value_counts(dropna=False)
        if counts.min() >= 2 and counts.shape[0] > 1:
            stratify = df[target]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    spec = _build_spec(dataset_key, cfg, df, n_train=len(train_df), n_test=len(test_df))

    if not return_frames:  # pragma: no cover
        return train_df, test_df, spec
    return train_df, test_df, spec
