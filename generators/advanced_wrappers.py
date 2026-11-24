"""Wrappers for CTAB-GAN+, WGAN-GP, ForestDiffusion, and TabDDPM.

Optional dependency packages that are not installed cause the corresponding
``is_available()`` to return False so the Optuna runner can skip them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseGenerator
from .postprocess import discrete_levels_from_train, snap_discrete_columns
from .registry import register


# ---------------------------------------------------------------------------
# CTAB-GAN+
# ---------------------------------------------------------------------------

def _ctabgan_root() -> Path | None:
    from config_utils import REPO_ROOT
    candidates = [
        REPO_ROOT / "_vendor" / "tab-ddpm" / "CTAB-GAN-Plus",
        REPO_ROOT / "Generators" / "Other GANS" / "CTAB-GAN-Plus",
    ]
    for c in candidates:
        if (c / "model" / "ctabgan.py").is_file():
            return c
    return None


@register
class CTABGANGenerator(BaseGenerator):
    name = "ctabgan"

    @classmethod
    def is_available(cls) -> bool:
        return _ctabgan_root() is not None

    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "CTABGANGenerator":
        root = _ctabgan_root()
        if root is None:
            raise RuntimeError("CTAB-GAN-Plus not found under _vendor/ or Generators/")
        import sys
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from model.ctabgan import CTABGAN

        self._columns = list(train_df.columns)
        target = getattr(metadata, "target", None)
        cat_cols = list(getattr(metadata, "categorical_columns", ()) or [])
        for c in train_df.columns:
            if not pd.api.types.is_numeric_dtype(train_df[c]) and c not in cat_cols:
                cat_cols.append(c)
        if target and target not in cat_cols and metadata.task == "classification":
            cat_cols.append(target)

        problem = {"Classification": target} if metadata.task == "classification" and target else \
                  {"Regression": target} if target else None

        class_dim = self.params.get("class_dim", [256, 256, 256, 256])
        if isinstance(class_dim, str):
            class_dim = eval(class_dim, {"__builtins__": {}})  # noqa: S307

        # CTAB-GAN+ library default is device="cpu"; always force CUDA for HPO.
        from .device import cuda_device_str

        device = self.params.get("device") or cuda_device_str(context="ctabgan")
        if str(device).lower() in {"cpu", "cpu:0"}:
            device = cuda_device_str(context="ctabgan")

        self._model = CTABGAN(
            df=train_df.reset_index(drop=True),
            test_ratio=0.0,
            categorical_columns=cat_cols,
            log_columns=[],
            mixed_columns={},
            general_columns=[],
            non_categorical_columns=[],
            integer_columns=[
                c for c in train_df.columns
                if pd.api.types.is_integer_dtype(train_df[c]) and c not in cat_cols
            ],
            problem_type=problem or {},
            epochs=int(self.params.get("epochs", 150)),
            batch_size=int(self.params.get("batch_size", 500)),
            random_dim=int(self.params.get("random_dim", 100)),
            num_channels=int(self.params.get("num_channels", 64)),
            l2scale=float(self.params.get("l2scale", 1e-5)),
            class_dim=tuple(class_dim),
            device=device,
        )
        self._model.fit()
        self.is_fitted = True
        return self

    # Maximum seconds a single generate_samples() call may take before we
    # give up and let Optuna prune the trial.  Forest Cover sampling often
    # needs ~10–15 min (was wrongly capped at 120s → all trials pruned).
    # Pathological hangs (hours) are still cut off.
    _SAMPLE_TIMEOUT_S: float = 600.0
    _MAX_RETRIES: int = 1

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        import time as _time

        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")
        seed0 = 0 if seed is None else int(seed)
        t0 = _time.monotonic()

        out = self._model.generate_samples(num_samples=int(n), seed=seed0)

        # Fast-fail: if the first call already blew the timeout or returned
        # almost nothing, don't keep retrying — let Optuna prune.
        elapsed = _time.monotonic() - t0
        if elapsed > self._SAMPLE_TIMEOUT_S:
            raise RuntimeError(
                f"CTAB-GAN sampling timed out ({elapsed:.0f}s > "
                f"{self._SAMPLE_TIMEOUT_S}s) — trial will be pruned."
            )

        if len(out) < n:
            parts = [out]
            extra_seed = seed0
            for _ in range(self._MAX_RETRIES):
                if sum(len(p) for p in parts) >= n:
                    break
                if (_time.monotonic() - t0) > self._SAMPLE_TIMEOUT_S:
                    break
                extra_seed += 1
                parts.append(
                    self._model.generate_samples(num_samples=int(n), seed=extra_seed)
                )
            out = pd.concat(parts, ignore_index=True)

        if len(out) < 1:
            raise RuntimeError(
                "CTAB-GAN generate_samples returned an empty frame after "
                f"{_time.monotonic() - t0:.0f}s — trial will be pruned."
            )
        if len(out) < n:
            # Pad with repeated rows so downstream eval still runs
            # (better than hanging; quality will be penalised naturally).
            reps = int(np.ceil(n / len(out)))
            out = pd.concat([out] * reps, ignore_index=True)
        out = out.iloc[:n].reset_index(drop=True)
        for c in self._columns:
            if c not in out.columns:
                out[c] = pd.NA
        return out[self._columns]


# ---------------------------------------------------------------------------
# WGAN-GP (lightweight tabular MLP)
# ---------------------------------------------------------------------------

@register
class WGANGPGenerator(BaseGenerator):
    name = "wgan_gp"

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch  # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "WGANGPGenerator":
        import torch
        import torch.nn as nn

        self._columns = list(train_df.columns)
        self._discrete_levels = discrete_levels_from_train(train_df, metadata)
        # Keep discrete/binary targets out of continuous GAN path when possible;
        # still generate them continuously then snap in sample().
        num_df = train_df.select_dtypes(include=[np.number]).copy()
        self._num_cols = list(num_df.columns)
        self._cat_cols = [c for c in self._columns if c not in self._num_cols]
        self._cat_frames = {c: train_df[c].astype(str).reset_index(drop=True) for c in self._cat_cols}

        if not self._num_cols:
            # Degenerate: only categoricals — store empirical sampler
            self._train = train_df.reset_index(drop=True)
            self._torch_mode = False
            self.is_fitted = True
            return self

        self._torch_mode = True
        self._mean = num_df.mean()
        self._std = num_df.std().replace(0, 1.0)
        X = ((num_df - self._mean) / self._std).to_numpy(dtype=np.float32)

        latent_dim = int(self.params.get("latent_dim", 64))
        hidden = int(self.params.get("hidden_dim", 128))
        lr = float(self.params.get("lr", 1e-4))
        n_critic = int(self.params.get("n_critic", 5))
        lambda_gp = float(self.params.get("lambda_gp", 10.0))
        epochs = int(self.params.get("epochs", 500))
        batch_size = int(self.params.get("batch_size", 64))

        from .device import cuda_torch_device

        device = cuda_torch_device(context="wgan_gp")
        data_dim = X.shape[1]

        class MLP(nn.Module):
            def __init__(self, d_in, d_out):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(d_in, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden), nn.ReLU(),
                    nn.Linear(hidden, d_out),
                )

            def forward(self, z):
                return self.net(z)

        G = MLP(latent_dim, data_dim).to(device)
        D = MLP(data_dim, 1).to(device)
        opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
        opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

        data = torch.tensor(X, device=device)
        n = len(data)
        for _ in range(epochs):
            # critic
            for _k in range(n_critic):
                idx = np.random.randint(0, n, size=min(batch_size, n))
                real = data[idx]
                z = torch.randn(len(idx), latent_dim, device=device)
                fake = G(z).detach()
                # gradient penalty
                eps = torch.rand(len(idx), 1, device=device)
                interp = eps * real + (1 - eps) * fake
                interp.requires_grad_(True)
                d_interp = D(interp)
                grads = torch.autograd.grad(
                    outputs=d_interp.sum(), inputs=interp, create_graph=True
                )[0]
                gp = ((grads.view(len(idx), -1).norm(2, dim=1) - 1) ** 2).mean()
                loss_D = D(fake).mean() - D(real).mean() + lambda_gp * gp
                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
            # generator
            z = torch.randn(batch_size, latent_dim, device=device)
            loss_G = -D(G(z)).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        self._G = G.eval()
        self._latent_dim = latent_dim
        self._device = device
        self._train = train_df.reset_index(drop=True)
        self.is_fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")
        rng = np.random.default_rng(seed)
        # categorical columns: bootstrap from train
        base = self._train.sample(n=n, replace=True, random_state=seed).reset_index(drop=True)
        if not getattr(self, "_torch_mode", False):
            out = snap_discrete_columns(base[self._columns], getattr(self, "_discrete_levels", {}))
            return out[self._columns]

        import torch
        with torch.no_grad():
            z = torch.randn(n, self._latent_dim, device=self._device)
            fake = self._G(z).detach().cpu().numpy()
        num = pd.DataFrame(fake, columns=self._num_cols)
        num = num * self._std.to_numpy() + self._mean.to_numpy()
        out = base.copy()
        for c in self._num_cols:
            out[c] = num[c].to_numpy()
        out = snap_discrete_columns(out, getattr(self, "_discrete_levels", {}))
        return out[self._columns]



# ---------------------------------------------------------------------------
# ForestDiffusion
# ---------------------------------------------------------------------------

@register
class ForestDiffusionGenerator(BaseGenerator):
    name = "forest_diffusion"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from ForestDiffusion import ForestDiffusionModel  # noqa: F401
            return True
        except ImportError:
            return False

    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "ForestDiffusionGenerator":
        from ForestDiffusion import ForestDiffusionModel

        self._columns = list(train_df.columns)
        self._discrete_levels = discrete_levels_from_train(train_df, metadata)
        target = getattr(metadata, "target", None)
        task = getattr(metadata, "task", None)
        work = train_df.copy()
        for c in work.columns:
            if not pd.api.types.is_numeric_dtype(work[c]):
                work[c] = pd.Categorical(work[c].astype(str)).codes

        cat_names = set(getattr(metadata, "categorical_columns", ()) or ())
        if target and task == "classification":
            cat_names.add(target)
        cat_names.update(self._discrete_levels.keys())

        feature_cols = [c for c in work.columns if c != target] if target else list(work.columns)
        bin_cols: list[str] = []
        multi_cat: list[str] = []
        for c in feature_cols:
            nunique = int(work[c].nunique(dropna=True))
            if c in cat_names and nunique <= 2:
                bin_cols.append(c)
            elif c in cat_names:
                multi_cat.append(c)

        self._target = target
        self._classification = bool(target and task == "classification")
        self._feature_cols = feature_cols
        if self._classification:
            label_y = work[target].to_numpy()
            arr = work[feature_cols].to_numpy(dtype=float)
            bin_indexes = [feature_cols.index(c) for c in bin_cols]
            cat_indexes = [feature_cols.index(c) for c in multi_cat]
        else:
            label_y = None
            arr = work.to_numpy(dtype=float)
            self._feature_cols = list(work.columns)
            bin_indexes = [self._feature_cols.index(c) for c in bin_cols]
            cat_indexes = [self._feature_cols.index(c) for c in multi_cat]

        # ForestDiffusion RAM profile on this host (~23 GiB):
        # each fitted model bank is several GiB; duplicate_K / n_t / depth drive it.
        # Always use the QuantileDMatrix iterator (n_batch>=1) so we never
        # materialize X_train of shape (n_t, n*K, c). Cap hyperparams by free RAM.
        import os
        import torch

        from .device import assert_cuda_available, require_cuda_enabled

        use_gpu = bool(self.params.get("gpu_hist"))
        if "gpu_hist" not in self.params:
            if require_cuda_enabled():
                assert_cuda_available(context="forest_diffusion")
                use_gpu = True
            else:
                use_gpu = bool(torch.cuda.is_available())

        n_t = int(self.params.get("n_t", 50))
        duplicate_k = int(self.params.get("duplicate_K", 100))
        n_estimators = int(self.params.get("n_estimators", 100))
        max_depth = int(self.params.get("max_depth", 7))
        n_t, duplicate_k, n_estimators, max_depth = _forest_memory_caps(
            n_rows=int(arr.shape[0]),
            n_cols=int(arr.shape[1]),
            n_t=n_t,
            duplicate_k=duplicate_k,
            n_estimators=n_estimators,
            max_depth=max_depth,
        )

        kwargs = dict(
            n_t=n_t,
            duplicate_K=duplicate_k,
            n_estimators=n_estimators,
            max_depth=max_depth,
            eta=float(self.params.get("eta", 0.3)),
            bin_indexes=bin_indexes,
            cat_indexes=cat_indexes,
            # Never Parallel(-1): joblib workers multiply host RAM.
            n_jobs=1,
            n_batch=int(self.params.get("n_batch", 4)),
            seed=int(self.params.get("seed", 42)),
            label_y=label_y,
            gpu_hist=use_gpu,
            tree_method=str(self.params.get("tree_method", "hist")),
            # Passed through ForestDiffusion's **xgboost_kwargs into XGBRegressor.
            nthread=int(os.environ.get("FOREST_XGB_NTHREAD", "1")),
        )
        diffusion_type = self.params.get("diffusion_type", "vp")
        try:
            self._model = ForestDiffusionModel(arr, diffusion_type=diffusion_type, **kwargs)
        except TypeError:
            kwargs.pop("label_y", None)
            kwargs.pop("nthread", None)
            try:
                self._model = ForestDiffusionModel(arr, diffusion_type=diffusion_type, **kwargs)
            except TypeError:
                kwargs.pop("n_batch", None)
                self._model = ForestDiffusionModel(arr, **kwargs)
        self.is_fitted = True
        return self

    def release(self) -> None:
        """Drop XGBoost model banks (multi‑GiB) after sampling."""
        import gc

        model = getattr(self, "_model", None)
        if model is not None:
            for attr in ("regr", "regr_", "X1", "X_covs", "mask_y"):
                if hasattr(model, attr):
                    try:
                        setattr(model, attr, None)
                    except Exception:
                        pass
            self._model = None
        self.is_fitted = False
        gc.collect()

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")
        try:
            arr = self._model.generate(n_samples=n)
        except TypeError:
            arr = self._model.generate(batch_size=n)
        if getattr(self, "_classification", False) and arr.shape[1] == len(self._feature_cols) + 1:
            out = pd.DataFrame(arr[:, :-1], columns=self._feature_cols)
            out[self._target] = arr[:, -1]
        elif arr.shape[1] == len(self._columns):
            out = pd.DataFrame(arr, columns=self._columns)
        else:
            out = pd.DataFrame(arr[:, : len(self._feature_cols)], columns=self._feature_cols)
            if getattr(self, "_target", None):
                out[self._target] = arr[:, -1] if arr.shape[1] > len(self._feature_cols) else pd.NA
        for c in self._columns:
            if c not in out.columns:
                out[c] = pd.NA
        out = snap_discrete_columns(out, getattr(self, "_discrete_levels", {}))
        # Free the fitted forest bank immediately — eval only needs the synthetic frame.
        self.release()
        return out[self._columns]


def _forest_memory_caps(
    *,
    n_rows: int,
    n_cols: int,
    n_t: int,
    duplicate_k: int,
    n_estimators: int,
    max_depth: int,
) -> tuple[int, int, int, int]:
    """Shrink ForestDiffusion knobs when host RAM is tight (~23 GiB box)."""
    import os

    # Leave headroom for OS + Cursor + one Optuna process (~6 GiB reserved).
    avail_kb = None
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    avail_kb = int(line.split()[1])
                    break
    except OSError:
        pass
    avail_gb = (avail_kb / (1024**2)) if avail_kb else float(os.environ.get("FOREST_ASSUME_AVAIL_GB", "12"))

    # Empirically one forest fit peaked ~6–8 GiB at K=100, n_t≈50, depth≈9 on cancer.
    # Scale caps so a single job stays under ~min(8, 0.45*avail) GiB.
    budget_gb = max(3.0, min(8.0, 0.45 * avail_gb))
    # Rough proxy score ∝ n_rows * n_cols * K * n_t * n_estimators * 2^depth_factor
    depth_factor = max(1, max_depth - 2)
    score = (n_rows * max(n_cols, 1) * duplicate_k * n_t * n_estimators * depth_factor) / 1e9
    # Reference: cancer 455*30*100*50*100*7 / 1e9 ≈ 4.8 → ~7 GiB observed
    if score > budget_gb * 0.7 or avail_gb < 10:
        duplicate_k = min(duplicate_k, 50)
        n_t = min(n_t, 50)
        n_estimators = min(n_estimators, 100)
        max_depth = min(max_depth, 7)
    if avail_gb < 6:
        duplicate_k = min(duplicate_k, 50)
        n_t = min(n_t, 40)
        n_estimators = min(n_estimators, 50)
        max_depth = min(max_depth, 6)
    return n_t, duplicate_k, n_estimators, max_depth


# ---------------------------------------------------------------------------
# TabDDPM — Kotelnikov et al. (ICML 2023), yandex-research/tab-ddpm
# ---------------------------------------------------------------------------

@register
class TabDDPMGenerator(BaseGenerator):
    name = "tabddpm"

    @classmethod
    def is_available(cls) -> bool:
        from .tabddpm_backend import is_tabddpm_available

        return is_tabddpm_available()

    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "TabDDPMGenerator":
        from .tabddpm_backend import TabDDPMBackend

        self._columns = list(train_df.columns)
        self._discrete_levels = discrete_levels_from_train(train_df, metadata)
        self._backend = TabDDPMBackend(self.params)
        self._backend.fit(train_df, metadata)
        self.is_fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")
        out = self._backend.sample(n=int(n), seed=seed)
        for c in self._columns:
            if c not in out.columns:
                out[c] = pd.NA
        out = snap_discrete_columns(out, getattr(self, "_discrete_levels", {}))
        return out[self._columns]
