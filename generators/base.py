"""Common interface for all synthetic tabular data generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd

# Optuna is optional at import time so Step-1 scaffolding works before
# full tuning deps are installed in every environment.
try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore


class BaseGenerator(ABC):
    """Abstract base class every generator must implement.

    Contract
    --------
    * ``fit`` trains only on the real training split (never sees the hold-out test).
    * ``sample`` returns a synthetic dataframe with the same columns as training data.
    * ``get_search_space`` maps an Optuna trial + YAML config → concrete hyperparameters.
    * ``from_params`` builds an unfitted instance from a parameter dict.
    """

    #: Canonical registry key, e.g. ``"ctgan"``.
    name: str = "base"

    @classmethod
    def is_available(cls) -> bool:
        """Return False when optional dependencies are missing."""
        return True

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = dict(params or {})
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, metadata: Any) -> "BaseGenerator":
        """Fit the generator on real training data.

        Parameters
        ----------
        train_df:
            Real training dataframe (features + target if present).
        metadata:
            Dataset specification (column types, target name, task type).
            Typed as ``DatasetSpec`` once datasets.schema is implemented.
        """

    @abstractmethod
    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        """Draw ``n`` synthetic rows after ``fit``."""

    @classmethod
    def get_search_space(cls, trial: "optuna.Trial", cfg: Mapping[str, Any]) -> dict[str, Any]:
        """Sample hyperparameters from YAML ``search_space`` via Optuna.

        Default implementation is shared by all eight generators: merge
        ``defaults`` with values suggested from ``search_space``. Subclasses
        may override if they need custom conditional spaces.
        """
        from tuning.search_space import defaults_from_yaml, suggest_from_yaml

        params = defaults_from_yaml(cfg)
        params.update(suggest_from_yaml(trial, cfg.get("search_space") or {}))
        return params

    @classmethod
    def from_params(cls, params: Mapping[str, Any]) -> "BaseGenerator":
        """Construct an unfitted generator from a parameter dictionary."""
        return cls(params=params)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_params(self) -> dict[str, Any]:
        """Return a copy of the current hyperparameters."""
        return dict(self.params)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r}, fitted={self.is_fitted})"
