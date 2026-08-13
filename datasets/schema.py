"""Dataset specification for the HPO framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class DatasetSpec:
    """Immutable description of a tabular dataset used by generators and evaluators.

    Column-type tuples are filled by the loader after preprocessing so generators
    receive a consistent schema without hard-coding dataset quirks.
    """

    key: str
    display_name: str
    path: str
    task: Literal["classification", "regression"]
    target: str | None
    test_size: float = 0.2
    seed: int = 42
    source: str = "csv"
    categorical_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    n_rows: int = 0
    n_train: int = 0
    n_test: int = 0
    extras: dict[str, Any] = field(default_factory=dict)

    def require_target(self) -> str:
        """Return the target column name or raise if missing."""
        if not self.target:
            raise ValueError(f"Dataset {self.key!r} has no target column configured.")
        return self.target
