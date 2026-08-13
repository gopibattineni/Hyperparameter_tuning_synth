"""SDV synthesizer wrappers: GaussianCopula, CopulaGAN, CTGAN, TVAE."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from .base import BaseGenerator
from .registry import register


def _sdv_available() -> bool:
    try:
        import sdv  # noqa: F401
        return True
    except ImportError:
        return False


def _build_metadata(df: pd.DataFrame):
    from sdv.metadata import SingleTableMetadata

    meta = SingleTableMetadata()
    meta.detect_from_dataframe(df)
    return meta


class _SDVBase(BaseGenerator):
    """Shared fit/sample logic for SDV single-table synthesizers."""

    _synthesizer_cls_path: str = ""

    @classmethod
    def is_available(cls) -> bool:
        return _sdv_available()

    def _make_synthesizer(self, metadata, params: Mapping[str, Any]):
        raise NotImplementedError

    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "_SDVBase":
        if not self.is_available():
            raise RuntimeError("sdv is not installed. pip install sdv")
        self._columns = list(train_df.columns)
        meta = _build_metadata(train_df)
        # Filter params to known synthesizer kwargs
        params = dict(self.params)
        self._synth = self._make_synthesizer(meta, params)
        self._synth.fit(train_df)
        self.is_fitted = True
        return self

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Call fit() before sample().")
        if seed is not None:
            try:
                out = self._synth.sample(num_rows=n, batch_size=n)
            except TypeError:
                out = self._synth.sample(num_rows=n)
        else:
            out = self._synth.sample(num_rows=n)
        # Align columns
        for c in self._columns:
            if c not in out.columns:
                out[c] = pd.NA
        return out[self._columns].reset_index(drop=True)


@register
class GaussianCopulaGenerator(_SDVBase):
    name = "gaussian_copula"

    def _make_synthesizer(self, metadata, params: Mapping[str, Any]):
        from sdv.single_table import GaussianCopulaSynthesizer

        kwargs = {}
        if "default_distribution" in params:
            kwargs["default_distribution"] = params["default_distribution"]
        if "enforce_min_max_values" in params:
            kwargs["enforce_min_max_values"] = bool(params["enforce_min_max_values"])
        return GaussianCopulaSynthesizer(metadata, **kwargs)


@register
class CTGANGenerator(_SDVBase):
    name = "ctgan"

    def _make_synthesizer(self, metadata, params: Mapping[str, Any]):
        from sdv.single_table import CTGANSynthesizer

        keys = [
            "epochs", "batch_size", "generator_lr", "discriminator_lr",
            "generator_dim", "discriminator_dim", "discriminator_steps", "pac",
        ]
        kwargs = {k: params[k] for k in keys if k in params}
        # Ensure list dims
        for d in ("generator_dim", "discriminator_dim"):
            if d in kwargs and isinstance(kwargs[d], tuple):
                kwargs[d] = list(kwargs[d])
        return CTGANSynthesizer(metadata, **kwargs)


@register
class CopulaGANGenerator(_SDVBase):
    name = "copulagan"

    def _make_synthesizer(self, metadata, params: Mapping[str, Any]):
        from sdv.single_table import CopulaGANSynthesizer

        keys = [
            "epochs", "batch_size", "generator_lr", "discriminator_lr",
            "generator_dim", "discriminator_dim", "discriminator_steps", "pac",
            "default_distribution",
        ]
        kwargs = {k: params[k] for k in keys if k in params}
        for d in ("generator_dim", "discriminator_dim"):
            if d in kwargs and isinstance(kwargs[d], tuple):
                kwargs[d] = list(kwargs[d])
        return CopulaGANSynthesizer(metadata, **kwargs)


@register
class TVAEGenerator(_SDVBase):
    name = "tvae"

    def _make_synthesizer(self, metadata, params: Mapping[str, Any]):
        from sdv.single_table import TVAESynthesizer

        keys = [
            "epochs", "batch_size", "embedding_dim", "compress_dims",
            "decompress_dims", "l2scale", "loss_factor",
        ]
        kwargs = {k: params[k] for k in keys if k in params}
        for d in ("compress_dims", "decompress_dims"):
            if d in kwargs and isinstance(kwargs[d], tuple):
                kwargs[d] = list(kwargs[d])
        return TVAESynthesizer(metadata, **kwargs)
