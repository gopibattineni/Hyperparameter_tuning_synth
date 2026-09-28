"""TabDDPM train/sample helpers around the yandex-research vendor checkout."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config_utils import REPO_ROOT


def tabddpm_root() -> Path | None:
    candidates = [
        REPO_ROOT / "_vendor" / "tab-ddpm",
        Path("/home/gopi_b/SYNTH_BENCHMARK/_vendor/tab-ddpm"),
    ]
    for root in candidates:
        if (root / "tab_ddpm" / "gaussian_multinomial_diffsuion.py").is_file():
            return root
    return None


def is_tabddpm_available() -> bool:
    if tabddpm_root() is None:
        return False
    try:
        import torch  # noqa: F401
        import zero  # noqa: F401
        import rtdl  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_vendor_path() -> Path:
    root = tabddpm_root()
    if root is None:
        raise RuntimeError("TabDDPM vendor not found under SYNTH_BENCHMARK/_vendor/tab-ddpm")
    # Vendor scripts/train.py must not lose to this repo's scripts/ package.
    for extra in (root / "scripts", root):
        path = str(extra)
        if path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
    return root


def _import_vendor_train_sample():
    """Load vendor train/sample as top-level modules (not this repo's scripts/)."""
    _ensure_vendor_path()
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            mod = sys.modules.get(key)
            fname = getattr(mod, "__file__", "") or ""
            if "Hyperparameter_tuning_synth" in fname:
                del sys.modules[key]
    import importlib

    train_mod = importlib.import_module("train")
    sample_mod = importlib.import_module("sample")
    return train_mod.train, sample_mod.sample


def _label_inverse(classes: np.ndarray, values) -> np.ndarray:
    arr = np.asarray(values).astype(int)
    if arr.size == 0:
        return arr
    return classes[np.clip(arr, 0, len(classes) - 1)]


def _write_vendor_arrays(
    df: pd.DataFrame,
    *,
    target: str,
    cat_cols: list[str],
    num_cols: list[str],
    out_dir: Path,
    seed: int,
    regression: bool,
) -> dict[str, np.ndarray]:
    from sklearn.model_selection import train_test_split

    out_dir.mkdir(parents=True, exist_ok=True)
    work = df.copy()
    encoders: dict[str, np.ndarray] = {}

    feature_cats = [c for c in cat_cols if c != target]
    x_cat_parts = []
    for col in feature_cats:
        classes, codes = np.unique(work[col].astype(str), return_inverse=True)
        encoders[col] = classes
        x_cat_parts.append(codes.reshape(-1, 1))
    x_cat = np.hstack(x_cat_parts) if x_cat_parts else None
    x_num = work[num_cols].astype(float).to_numpy() if num_cols else None

    if regression:
        task_type = "regression"
        y_arr = work[target].astype(float).to_numpy()
    else:
        if target not in encoders:
            classes, y_arr = np.unique(work[target].astype(str), return_inverse=True)
            encoders[target] = classes
        else:
            y_arr = pd.Categorical(work[target].astype(str), categories=encoders[target]).codes
        task_type = "binclass" if len(np.unique(y_arr)) == 2 else "multiclass"

    n = len(y_arr)
    indices = np.arange(n)
    strat = y_arr if (not regression and len(np.unique(y_arr)) > 1 and np.min(np.bincount(y_arr)) >= 2) else None
    if n < 10:
        train_idx, val_idx, test_idx = indices, indices[:1], indices[:1]
    else:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=strat
        )
        strat_temp = y_arr[temp_idx] if strat is not None and np.min(np.bincount(y_arr[temp_idx])) >= 2 else None
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=strat_temp
        )

    for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if x_num is not None:
            np.save(out_dir / f"X_num_{split}.npy", x_num[idx])
        if x_cat is not None:
            np.save(out_dir / f"X_cat_{split}.npy", x_cat[idx])
        np.save(out_dir / f"y_{split}.npy", y_arr[idx])

    info: dict[str, Any] = {
        "name": "custom",
        "task_type": task_type,
        "n_num_features": 0 if x_num is None else int(x_num.shape[1]),
        "n_cat_features": 0 if x_cat is None else int(x_cat.shape[1]),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
    }
    if task_type != "regression":
        info["n_classes"] = int(len(np.unique(y_arr)))
    (out_dir / "info.json").write_text(json.dumps(info))
    return encoders


def _arrays_to_frame(
    template: pd.DataFrame,
    *,
    target: str,
    feature_cats: list[str],
    num_cols: list[str],
    encoders: dict[str, np.ndarray],
    x_num: np.ndarray | None,
    x_cat: np.ndarray | None,
    y: np.ndarray,
    regression: bool,
) -> pd.DataFrame:
    if x_cat is not None and x_cat.shape[1] == len(feature_cats) + 1:
        x_cat = x_cat[:, 1:]
    out = pd.DataFrame(index=range(len(y)))
    i_num = 0
    i_cat = 0
    for col in template.columns:
        if col in num_cols:
            out[col] = x_num[:, i_num] if x_num is not None else np.nan
            i_num += 1
        elif col in feature_cats:
            raw = x_cat[:, i_cat] if x_cat is not None else 0
            if col in encoders:
                raw = _label_inverse(encoders[col], raw)
            out[col] = raw
            i_cat += 1
        elif col == target:
            if regression:
                out[col] = y.astype(float)
            elif target in encoders:
                out[col] = _label_inverse(encoders[target], y)
            else:
                out[col] = y
    return out[template.columns]


class TabDDPMBackend:
    """Fit once, sample many times from the vendor checkpoint."""

    def __init__(self, params: dict[str, Any]) -> None:
        self.params = dict(params)
        self._tmp: tempfile.TemporaryDirectory[str] | None = None
        self._data_dir: Path | None = None
        self._model_dir: Path | None = None
        self._encoders: dict[str, np.ndarray] = {}
        self._columns: list[str] = []
        self._target: str = ""
        self._feature_cats: list[str] = []
        self._num_cols: list[str] = []
        self._regression = False
        self._model_params: dict[str, Any] = {}
        self._t_dict: dict[str, Any] = {}
        self._device = None

    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "TabDDPMBackend":
        import torch
        tabddpm_train, _ = _import_vendor_train_sample()
        self._columns = list(train_df.columns)
        self._target = str(getattr(metadata, "target", None) or "")
        if not self._target or self._target not in train_df.columns:
            raise RuntimeError("TabDDPM requires a target column")

        cat_cols = list(getattr(metadata, "categorical_columns", ()) or [])
        for col in train_df.columns:
            if not pd.api.types.is_numeric_dtype(train_df[col]) and col not in cat_cols:
                cat_cols.append(col)
        task = getattr(metadata, "task", "classification")
        self._regression = str(task).lower() == "regression"
        if not self._regression and self._target not in cat_cols:
            cat_cols.append(self._target)
        self._feature_cats = [c for c in cat_cols if c != self._target]
        self._num_cols = [
            c for c in train_df.columns
            if c != self._target and c not in self._feature_cats
        ]

        n = len(train_df)
        n_layers = int(self.params.get("n_layers", 4))
        d_width = int(self.params.get("d_layers", 256))
        d_layers = [d_width] * n_layers
        num_classes = 0 if self._regression else int(train_df[self._target].nunique())
        # Classification: condition on y (is_y_cond=True).
        # Regression: vendor sample.py always treats X_num[:,0] as y when
        # num_classes==0, so y must be concatenated into X_num (is_y_cond=False).
        self._is_y_cond = not self._regression
        self._model_params = {
            "num_classes": num_classes,
            "is_y_cond": self._is_y_cond,
            "rtdl_params": {
                "d_layers": d_layers,
                "dropout": float(self.params.get("dropout", 0.0)),
            },
        }
        self._t_dict = {
            "seed": int(self.params.get("seed", 42)),
            # Quantile when any numeric is present. Regression always concatenates
            # y into X_num (is_y_cond=False), so enable even if all features are cat.
            "normalization": "quantile" if (self._num_cols or self._regression) else None,
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": "one-hot",
            "y_policy": "default",
        }
        device_name = self.params.get("device")
        if not device_name:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_name)

        self._tmp = tempfile.TemporaryDirectory(prefix="tabddpm_hpo_")
        tmp = Path(self._tmp.name)
        self._data_dir = tmp / "data"
        self._model_dir = tmp / "model"
        self._data_dir.mkdir()
        self._model_dir.mkdir()
        self._encoders = _write_vendor_arrays(
            train_df,
            target=self._target,
            cat_cols=self._feature_cats,
            num_cols=self._num_cols,
            out_dir=self._data_dir,
            seed=int(self.params.get("seed", 42)),
            regression=self._regression,
        )
        if not self._regression and self._target in train_df.columns and self._target not in self._encoders:
            classes, _ = np.unique(train_df[self._target].astype(str), return_inverse=True)
            self._encoders[self._target] = classes

        batch_size = int(self.params.get("batch_size", 1024))
        batch_size = max(32, min(batch_size, n))
        tabddpm_train(
            parent_dir=str(self._model_dir),
            real_data_path=str(self._data_dir),
            steps=int(self.params.get("steps", 10000)),
            lr=float(self.params.get("lr", 0.001)),
            weight_decay=float(self.params.get("weight_decay", 1e-4)),
            batch_size=batch_size,
            model_type="mlp",
            model_params=dict(self._model_params),
            num_timesteps=int(self.params.get("num_timesteps", 1000)),
            gaussian_loss_type=str(self.params.get("gaussian_loss_type", "mse")),
            scheduler=str(self.params.get("scheduler", "cosine")),
            T_dict=self._t_dict,
            device=self._device,
            seed=int(self.params.get("seed", 42)),
        )
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        import torch
        _, tabddpm_sample = _import_vendor_train_sample()
        if self._model_dir is None or self._data_dir is None:
            raise RuntimeError("Call fit() before sample().")
        seed0 = int(self.params.get("seed", 42) if seed is None else seed)
        n = int(n)
        # Pass raw numeric feature count only. Vendor sample.py adds +1 for
        # regression when is_y_cond=False (y prepended before quantile).
        n_num_arg = len(self._num_cols)
        tabddpm_sample(
            parent_dir=str(self._model_dir),
            real_data_path=str(self._data_dir),
            batch_size=min(2000, max(n, 1)),
            num_samples=n,
            model_type="mlp",
            model_params=dict(self._model_params),
            model_path=str(self._model_dir / "model.pt"),
            num_timesteps=int(self.params.get("num_timesteps", 1000)),
            gaussian_loss_type=str(self.params.get("gaussian_loss_type", "mse")),
            scheduler=str(self.params.get("scheduler", "cosine")),
            T_dict=self._t_dict,
            num_numerical_features=n_num_arg,
            device=self._device or torch.device("cpu"),
            seed=seed0,
        )
        x_num_path = self._model_dir / "X_num_train.npy"
        x_cat_path = self._model_dir / "X_cat_train.npy"
        y_path = self._model_dir / "y_train.npy"
        x_num = np.load(x_num_path, allow_pickle=True) if x_num_path.is_file() else None
        x_cat = np.load(x_cat_path, allow_pickle=True) if x_cat_path.is_file() else None
        y = np.load(y_path, allow_pickle=True) if y_path.is_file() else None
        if y is None:
            y = np.zeros(n)
        # Guard: align x_num width to expected feature count
        if x_num is not None and x_num.shape[1] > len(self._num_cols):
            x_num = x_num[:, : len(self._num_cols)]
        if x_num is not None and x_num.shape[1] < len(self._num_cols):
            pad = np.zeros((x_num.shape[0], len(self._num_cols) - x_num.shape[1]))
            x_num = np.hstack([x_num, pad])
        template = pd.DataFrame(columns=self._columns)
        out = _arrays_to_frame(
            template,
            target=self._target,
            feature_cats=self._feature_cats,
            num_cols=self._num_cols,
            encoders=self._encoders,
            x_num=x_num,
            x_cat=x_cat,
            y=y,
            regression=self._regression,
        )
        if len(out) < n:
            reps = int(np.ceil(n / max(len(out), 1)))
            out = pd.concat([out] * reps, ignore_index=True)
        return out.iloc[:n].reset_index(drop=True)

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
            self._tmp = None
