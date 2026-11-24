"""Trial objective wrapper: seeding, timing, F/U/P attrs, soft timeout."""

from __future__ import annotations

import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from typing import Any, Mapping, Type

import optuna
import pandas as pd

from datasets.schema import DatasetSpec
from evaluation.fidelity import evaluate_fidelity, sdv_required_but_missing
from evaluation.objective import compute_objective
from evaluation.privacy import evaluate_privacy
from evaluation.utility import evaluate_utility
from generators.base import BaseGenerator
from multiseed_comparison.seeding import set_global_seed, trial_seed
from tuning.objective_fn import TrialObjective
from tuning.search_space import defaults_from_yaml, suggest_from_yaml


@dataclass
class _EvalResult:
    ok: bool
    objective: float | None = None
    fidelity: dict[str, Any] | None = None
    privacy: dict[str, Any] | None = None
    utility: dict[str, Any] | None = None
    combined: dict[str, Any] | None = None
    error: str | None = None
    traceback: str | None = None


def _fit_and_evaluate(
    *,
    generator_name: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: DatasetSpec,
    objective_cfg: Mapping[str, Any],
    sample_n: int,
    seed: int,
) -> _EvalResult:
    """Fit generator, sample, and score F/U/P → J (shared by in-process / worker)."""
    try:
        set_global_seed(seed)
        from generators.registry import get_generator_class

        cls = get_generator_class(generator_name)
        generator = cls.from_params(params)
        try:
            generator.fit(train_df, metadata)
            synthetic = generator.sample(n=sample_n, seed=seed)
        finally:
            if hasattr(generator, "release"):
                try:
                    generator.release()
                except Exception:
                    pass

        for col in train_df.columns:
            if col not in synthetic.columns:
                synthetic[col] = pd.NA
        synthetic = synthetic[train_df.columns]

        fidelity = evaluate_fidelity(train_df, synthetic, metadata, objective_cfg)
        if sdv_required_but_missing(fidelity, objective_cfg):
            return _EvalResult(
                ok=False,
                fidelity=dict(fidelity),
                error=(
                    "FidelityUnavailableError: SDV QualityReport required "
                    "(fidelity.aggregation=sdv_only / require_sdv) but unavailable; "
                    "no silent KS/corr fallback"
                ),
            )
        privacy = evaluate_privacy(train_df, synthetic, metadata, objective_cfg)
        utility = evaluate_utility(
            train_df, synthetic, test_df, metadata, objective_cfg
        )
        combined = compute_objective(fidelity, privacy, utility, objective_cfg)
        return _EvalResult(
            ok=True,
            objective=float(combined["objective"]),
            fidelity=dict(fidelity),
            privacy=dict(privacy),
            utility=dict(utility),
            combined=dict(combined),
        )
    except Exception as exc:
        return _EvalResult(
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )


def _fit_eval_worker(
    q: mp.Queue,
    generator_name: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: DatasetSpec,
    objective_cfg: dict[str, Any],
    sample_n: int,
    seed: int,
) -> None:
    """Child process: fit generator, sample, evaluate (killable on timeout)."""
    q.put(
        _fit_and_evaluate(
            generator_name=generator_name,
            params=params,
            train_df=train_df,
            test_df=test_df,
            metadata=metadata,
            objective_cfg=objective_cfg,
            sample_n=sample_n,
            seed=seed,
        )
    )


def _run_fit_eval(
    *,
    generator_name: str,
    params: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: DatasetSpec,
    objective_cfg: Mapping[str, Any],
    sample_n: int,
    seed: int,
    timeout_s: float | None,
) -> _EvalResult:
    """Run fit/eval in a subprocess when ``timeout_s`` is set; else in-process."""
    if timeout_s is None or timeout_s <= 0:
        return _fit_and_evaluate(
            generator_name=generator_name,
            params=params,
            train_df=train_df,
            test_df=test_df,
            metadata=metadata,
            objective_cfg=objective_cfg,
            sample_n=sample_n,
            seed=seed,
        )

    ctx = mp.get_context("spawn")
    q: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_fit_eval_worker,
        args=(
            q,
            generator_name,
            params,
            train_df,
            test_df,
            metadata,
            dict(objective_cfg),
            sample_n,
            seed,
        ),
    )
    proc.start()
    proc.join(timeout=float(timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
        return _EvalResult(
            ok=False,
            error=f"TimeoutError: trial exceeded {timeout_s:.0f}s",
        )
    if not q.empty():
        return q.get()
    return _EvalResult(ok=False, error="RuntimeError: worker exited without result")


class TimedTrialObjective:
    """Optuna objective that reuses YAML search spaces and existing metrics.

    Does **not** change objective weights. Adds:
      - per-trial reseeding from ``(study_seed, trial.number)``
      - ``duration_s``, fidelity/utility/privacy user attrs
      - optional hard timeout via subprocess kill
    """

    def __init__(
        self,
        *,
        generator_cls: Type[BaseGenerator],
        generator_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        metadata: DatasetSpec,
        generator_cfg: Mapping[str, Any],
        objective_cfg: Mapping[str, Any],
        study_seed: int,
        trial_timeout_s: float | None = None,
        sample_n: int | None = None,
    ) -> None:
        self.generator_cls = generator_cls
        self.generator_name = generator_name
        self.train_df = train_df
        self.test_df = test_df
        self.metadata = metadata
        self.generator_cfg = dict(generator_cfg)
        self.objective_cfg = dict(objective_cfg)
        self.study_seed = int(study_seed)
        self.trial_timeout_s = trial_timeout_s
        self.sample_n = int(sample_n if sample_n is not None else len(train_df))
        # Keep a reference so search-space hooks match TrialObjective.
        self._base = TrialObjective(
            generator_cls=generator_cls,
            train_df=train_df,
            test_df=test_df,
            metadata=metadata,
            generator_cfg=generator_cfg,
            objective_cfg=objective_cfg,
            sample_n=self.sample_n,
            seed=self.study_seed,
        )

    def _sample_params(self, trial: optuna.Trial) -> dict[str, Any]:
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
        t0 = time.perf_counter()
        params = self._sample_params(trial)
        seed = trial_seed(self.study_seed, trial.number)
        trial.set_user_attr("trial_seed", seed)
        trial.set_user_attr("study_seed", self.study_seed)

        result = _run_fit_eval(
            generator_name=self.generator_name,
            params=params,
            train_df=self.train_df,
            test_df=self.test_df,
            metadata=self.metadata,
            objective_cfg=self.objective_cfg,
            sample_n=self.sample_n,
            seed=seed,
            timeout_s=self.trial_timeout_s,
        )
        duration = time.perf_counter() - t0
        trial.set_user_attr("duration_s", float(duration))

        if not result.ok:
            trial.set_user_attr("error", result.error or "unknown")
            if result.traceback:
                trial.set_user_attr("traceback", result.traceback[:2000])
            raise optuna.TrialPruned(result.error or "trial failed")

        # Store F / U / P (and related) as user attrs — same keys as TrialObjective.
        blob = {
            **(result.fidelity or {}),
            **(result.privacy or {}),
            **(result.utility or {}),
            **(result.combined or {}),
        }
        for key, value in blob.items():
            try:
                trial.set_user_attr(key, float(value) if value == value else None)
            except Exception:
                trial.set_user_attr(key, str(value))

        assert result.objective is not None
        return float(result.objective)
