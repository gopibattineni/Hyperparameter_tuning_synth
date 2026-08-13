"""Wrappers for CTAB-GAN+, WGAN-GP, ForestDiffusion, TabDDPM.

Heavy backends are optional; ``is_available()`` gates the experiment runner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseGenerator
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
        # Heuristic: object columns are categorical
        for c in train_df.columns:
            if not pd.api.types.is_numeric_dtype(train_df[c]) and c not in cat_cols:
                cat_cols.append(c)
        if target and target not in cat_cols and metadata.task == "classification":
            cat_cols.append(target)

        # CTABGAN expects a CSV path in original API — write a temp frame
        import tempfile
        self._tmpdir = tempfile.TemporaryDirectory()
        csv_path = Path(self._tmpdir.name) / "train.csv"
        train_df.to_csv(csv_path, index=False)

        problem = {"Classification": target} if metadata.task == "classification" and target else \
                  {"Regression": target} if target else None

        kwargs = {
            "raw_csv_path": str(csv_path),
            "test_ratio": 0.0,
            "categorical_columns": cat_cols,
            "log_columns": [],
            "mixed_columns": {},
            "general_columns": [],
            "non_categorical_columns": [],
            "integer_columns": [
                c for c in train_df.columns
                if pd.api.types.is_integer_dtype(train_df[c])
            ],
            "problem_type": problem or {"Classification": cat_cols[-1]},
            "epochs": int(self.params.get("epochs", 150)),
            "batch_size": int(self.params.get("batch_size", 500)),
            "random_dim": int(self.params.get("random_dim", 100)),
            "num_channels": int(self.params.get("num_channels", 64)),
            "l2scale": float(self.params.get("l2scale", 1e-5)),
        }
        class_dim = self.params.get("class_dim", [256, 256, 256, 256])
        kwargs["class_dim"] = tuple(class_dim)

        self._model = CTABGAN(**kwargs)
        self._model.fit()
        self.is_fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")
        out = self._model.generate_samples()
        if len(out) < n:
            # top-up by repeated generation
            parts = [out]
            while sum(len(p) for p in parts) < n:
                parts.append(self._model.generate_samples())
            out = pd.concat(parts, ignore_index=True)
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        self._G = G.cpu().eval()
        self._latent_dim = latent_dim
        self._device = torch.device("cpu")
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
            return base[self._columns]

        import torch
        with torch.no_grad():
            z = torch.randn(n, self._latent_dim)
            fake = self._G(z).numpy()
        num = pd.DataFrame(fake, columns=self._num_cols)
        num = num * self._std.to_numpy() + self._mean.to_numpy()
        out = base.copy()
        for c in self._num_cols:
            out[c] = num[c].to_numpy()
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
        target = getattr(metadata, "target", None)
        X = train_df.copy()
        y = None
        if target and target in X.columns and metadata.task == "classification":
            y = X[target].to_numpy()
            # ForestDiffusion often wants numeric X; label-encode objects
        for c in X.columns:
            if not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = pd.Categorical(X[c].astype(str)).codes

        self._X_cols = list(X.columns)
        arr = X.to_numpy(dtype=float)
        cat_idx = [
            i for i, c in enumerate(self._X_cols)
            if c in getattr(metadata, "categorical_columns", ())
        ]
        kwargs = dict(
            n_t=int(self.params.get("n_t", 50)),
            duplicate_K=int(self.params.get("duplicate_K", 100)),
            cat_indexes=cat_idx,
            n_jobs=1,
        )
        # Optional XGB hypers
        if "max_depth" in self.params or "n_estimators" in self.params or "eta" in self.params:
            kwargs["max_depth"] = int(self.params.get("max_depth", 7))
            # API varies across versions — try common names
        diffusion_type = self.params.get("diffusion_type", "vp")
        try:
            self._model = ForestDiffusionModel(arr, diffusion_type=diffusion_type, **kwargs)
        except TypeError:
            self._model = ForestDiffusionModel(arr, **kwargs)
        self.is_fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")
        try:
            arr = self._model.generate(n_samples=n)
        except TypeError:
            arr = self._model.generate(batch_size=n)
        out = pd.DataFrame(arr, columns=self._X_cols)
        for c in self._columns:
            if c not in out.columns:
                out[c] = pd.NA
        return out[self._columns]


# ---------------------------------------------------------------------------
# TabDDPM — availability gated (full training pipeline is heavy)
# ---------------------------------------------------------------------------

@register
class TabDDPMGenerator(BaseGenerator):
    name = "tabddpm"

    @classmethod
    def is_available(cls) -> bool:
        # Gated until the vendor training adapter is finished (Step 7).
        return False

    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "TabDDPMGenerator":
        # Full TabDDPM training requires the vendor scripts / configs.
        # Provide a clear error until the dedicated adapter is finalized.
        raise NotImplementedError(
            "TabDDPM wrapper training is not fully wired yet. "
            "Use vendor scripts under _vendor/tab-ddpm or wait for Step 7 completion."
        )

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        raise NotImplementedError("TabDDPM sample requires a fitted model.")
