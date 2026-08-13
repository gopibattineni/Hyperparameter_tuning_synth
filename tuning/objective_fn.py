"""Reusable Optuna trial objective for every registered generator.

Workflow per trial
------------------
1. Sample hyperparameters from the generator YAML search space.
2. ``fit`` on real training data only.
3. ``sample`` ``len(train_df)`` synthetic rows.
4. Evaluate fidelity / privacy / utility.
5. Normalize metrics to ``[0, 1]``.
6. Return ``0.4·F + 0.4·U − 0.2·PrivacyRisk``.
"""

from __future__ import annotations

from typing import Any, Mapping, Type

import pandas as pd

try:
    import optuna
except ImportError as exc:  # pragma: no cover
    raise ImportError("optuna is required for TrialObjective") from exc

from config_utils import load_config, load_generator_config
from datasets.schema import DatasetSpec
from evaluation.fidelity import evaluate_fidelity
from evaluation.objective import compute_objective
from evaluation.privacy import evaluate_privacy
from evaluation.utility import evaluate_utility
from generators.base import BaseGenerator
from .search_space import defaults_from_yaml, suggest_from_yaml


class TrialObjective:
    """Callable passed to ``study.optimize`` — generator-agnostic.

    Parameters
    ----------
    generator_cls:
        Any ``BaseGenerator`` subclass (CTGAN, TabDDPM, …).
    train_df, test_df:
        Fixed real splits from :func:`datasets.loader.load_train_test`.
    metadata:
        :class:`DatasetSpec` for the dataset.
    generator_cfg:
        Parsed generator YAML. Loaded automatically from ``generator_cls.name``
        when omitted.
    objective_cfg:
        Parsed ``objective.yaml``. Loaded automatically when omitted.
    sample_n:
        Synthetic rows to draw. Defaults to ``len(train_df)``.
    """

    def __init__(
        self,
        generator_cls: Type[BaseGenerator],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metadata: DatasetSpec,
        generator_cfg: Mapping[str, Any] | None = None,
        objective_cfg: Mapping[str, Any] | None = None,
        sample_n: int | None = None,
        seed: int = 42,
    ) -> None:
        self.generator_cls = generator_cls
        self.train_df = train_df
        self.test_df = test_df
        self.metadata = metadata
        self.generator_cfg = dict(
            generator_cfg
            if generator_cfg is not None
            else load_generator_config(generator_cls.name)
        )
        self.objective_cfg = dict(
            objective_cfg if objective_cfg is not None else load_config("objective")
        )
        self.sample_n = int(sample_n if sample_n is not None else len(train_df))
        self.seed = int(seed)
        self.last_metrics_: dict[str, Any] | None = None
        self.last_params_: dict[str, Any] | None = None
        self.last_synthetic_: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
        """Merge YAML defaults with Optuna-sampled search-space values."""
        # Prefer generator class hook (all eight gens share this contract).
        try:
            return self.generator_cls.get_search_space(trial, self.generator_cfg)
        except NotImplementedError:
            pass
        params = defaults_from_yaml(self.generator_cfg)
        params.update(
            suggest_from_yaml(trial, self.generator_cfg.get("search_space") or {})
        )
        return params

    def __call__(self, trial: optuna.Trial) -> float:
        """Run one Optuna trial and return the scalar objective (maximize)."""
        params = self._sample_params(trial)
        self.last_params_ = dict(params)

        generator = self.generator_cls.from_params(params)
        try:
            generator.fit(self.train_df, self.metadata)
            synthetic = generator.sample(n=self.sample_n, seed=self.seed + trial.number)
        except Exception as exc:
            # Failed trials are pruned so Optuna continues exploring.
            trial.set_user_attr("error", f"{type(exc).__name__}: {exc}")
            raise optuna.TrialPruned(f"generator failed: {exc}") from exc

        # Ensure column schema matches training data
        for col in self.train_df.columns:
            if col not in synthetic.columns:
                synthetic[col] = pd.NA
        synthetic = synthetic[self.train_df.columns]

        fidelity = evaluate_fidelity(
            self.train_df, synthetic, self.metadata, self.objective_cfg
        )
        privacy = evaluate_privacy(
            self.train_df, synthetic, self.metadata, self.objective_cfg
        )
        utility = evaluate_utility(
            self.train_df, synthetic, self.test_df, self.metadata, self.objective_cfg
        )
        combined = compute_objective(fidelity, privacy, utility, self.objective_cfg)

        # Log raw + normalized metrics on the trial for later analysis
        for key, value in {**fidelity, **privacy, **utility, **combined}.items():
            try:
                trial.set_user_attr(key, float(value) if value == value else None)
            except Exception:
                trial.set_user_attr(key, str(value))

        self.last_metrics_ = {
            "fidelity": fidelity,
            "privacy": privacy,
            "utility": utility,
            "combined": combined,
        }
        self.last_synthetic_ = synthetic

        return float(combined["objective"])
