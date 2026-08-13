"""Bootstrap / noise generator for smoke-testing the Optuna objective.

Not a research baseline — used only to verify TrialObjective wiring without
requiring SDV / diffusion dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import BaseGenerator
from .registry import register


@register
class BootstrapNoiseGenerator(BaseGenerator):
    """Resample training rows with mild numeric noise."""

    name = "bootstrap_noise"

    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "BootstrapNoiseGenerator":
        self._train = train_df.copy()
        self._metadata = metadata
        self.is_fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(self._train), size=n, replace=True)
        out = self._train.iloc[idx].reset_index(drop=True).copy()
        noise = float(self.params.get("noise_std", 0.01))
        for col in out.columns:
            if pd.api.types.is_numeric_dtype(out[col]):
                scale = out[col].std(ddof=0) or 1.0
                out[col] = out[col].astype(float) + rng.normal(0.0, noise * scale, size=n)
        return out
