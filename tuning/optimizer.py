"""Run a single Optuna study for one generator × dataset pair."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Type

import pandas as pd

try:
    import optuna
    from optuna.samplers import RandomSampler, TPESampler
    from optuna.pruners import MedianPruner, NopPruner
except ImportError as exc:  # pragma: no cover
    raise ImportError("optuna is required") from exc

from config_utils import CONFIG_ROOT, load_config, load_generator_config
from datasets.loader import load_train_test
from datasets.schema import DatasetSpec
from evaluation.fidelity import evaluate_fidelity
from evaluation.objective import compute_objective
from evaluation.privacy import evaluate_privacy
from evaluation.utility import evaluate_utility
from generators.base import BaseGenerator
from generators.registry import get_generator_class
from .artifacts import mark_completed, mark_failed, run_dir, save_run_artifacts
from .objective_fn import TrialObjective


def _build_sampler(cfg: Mapping[str, Any]):
    name = str((cfg.get("sampler") or {}).get("name", "TPESampler"))
    seed = int((cfg.get("sampler") or {}).get("seed", cfg.get("seed", 42)))
    if name == "RandomSampler":
        return RandomSampler(seed=seed)
    return TPESampler(seed=seed)


def _build_pruner(cfg: Mapping[str, Any]):
    p = cfg.get("pruner") or {}
    name = str(p.get("name", "MedianPruner"))
    if name == "NopPruner":
        return NopPruner()
    return MedianPruner(
        n_startup_trials=int(p.get("n_startup_trials", 5)),
        n_warmup_steps=int(p.get("n_warmup_steps", 0)),
    )


def run_study(
    generator_name: str,
    dataset_key: str,
    *,
    results_root: Path | str | None = None,
    n_trials: int | None = None,
    timeout_seconds: float | None = None,
    optuna_cfg: Mapping[str, Any] | None = None,
    objective_cfg: Mapping[str, Any] | None = None,
    generator_cls: Type[BaseGenerator] | None = None,
    show_trial_progress: bool = False,
) -> dict[str, Any]:
    """Optimize one pair and persist artifacts.

    Returns a summary dict suitable for appending to the master results CSV.
    """
    t0 = time.perf_counter()
    optuna_cfg = dict(optuna_cfg or load_config("optuna"))
    objective_cfg = dict(objective_cfg or load_config("objective"))
    gen_cfg = load_generator_config(generator_name)

    cls = generator_cls or get_generator_class(generator_name)
    if hasattr(cls, "is_available") and not cls.is_available():
        raise RuntimeError(f"Generator {generator_name!r} dependencies are not available")

    train_df, test_df, metadata = load_train_test(dataset_key)

    from config_utils import PACKAGE_ROOT

    root = Path(results_root) if results_root else PACKAGE_ROOT / "results"
    out = run_dir(root, dataset_key, generator_name)
    out.mkdir(parents=True, exist_ok=True)

    n_trials = int(n_trials if n_trials is not None else optuna_cfg.get("n_trials", 50))
    timeout = timeout_seconds if timeout_seconds is not None else optuna_cfg.get("timeout_seconds")

    objective = TrialObjective(
        generator_cls=cls,
        train_df=train_df,
        test_df=test_df,
        metadata=metadata,
        generator_cfg=gen_cfg,
        objective_cfg=objective_cfg,
        seed=int(optuna_cfg.get("seed", 42)),
    )

    study = optuna.create_study(
        direction=str(objective_cfg.get("direction", "maximize")),
        sampler=_build_sampler(optuna_cfg),
        pruner=_build_pruner(optuna_cfg),
        study_name=f"{dataset_key}__{generator_name}",
    )

    try:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=None if timeout in (None, "null") else float(timeout),
            show_progress_bar=show_trial_progress,
        )

        # Retrain best params and export final synthetic + metrics
        best_params = dict(study.best_params)
        # Merge with defaults for params not in search space
        from .search_space import defaults_from_yaml

        full_params = defaults_from_yaml(gen_cfg)
        full_params.update(best_params)

        best_gen = cls.from_params(full_params)
        best_gen.fit(train_df, metadata)
        synthetic = best_gen.sample(n=len(train_df), seed=int(optuna_cfg.get("seed", 42)))
        for col in train_df.columns:
            if col not in synthetic.columns:
                synthetic[col] = pd.NA
        synthetic = synthetic[train_df.columns]

        fidelity = evaluate_fidelity(train_df, synthetic, metadata, objective_cfg)
        privacy = evaluate_privacy(train_df, synthetic, metadata, objective_cfg)
        utility = evaluate_utility(train_df, synthetic, test_df, metadata, objective_cfg)
        combined = compute_objective(fidelity, privacy, utility, objective_cfg)

        elapsed = time.perf_counter() - t0
        timing = {"elapsed_sec": elapsed, "n_trials": len(study.trials), "n_complete": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        )}

        art_cfg = optuna_cfg.get("artifact") or {}
        save_run_artifacts(
            study,
            out,
            best_params=full_params,
            metrics={**fidelity, **privacy, **utility, **combined},
            synthetic=synthetic,
            timing=timing,
            config_snapshot={
                "optuna": CONFIG_ROOT / "optuna.yaml",
                "objective": CONFIG_ROOT / "objective.yaml",
                "generator": CONFIG_ROOT / "generators" / f"{generator_name}.yaml",
            }
            if art_cfg.get("snapshot_configs", True)
            else None,
            save_plots=list(art_cfg.get("save_plots") or []),
            save_study_pickle=bool(art_cfg.get("save_study_pickle", True)),
        )
        mark_completed(
            out,
            dataset=dataset_key,
            generator=generator_name,
            best_value=float(study.best_value) if study.best_trial else None,
            elapsed_sec=elapsed,
        )

        return {
            "dataset": dataset_key,
            "generator": generator_name,
            "status": "completed",
            "best_value": float(study.best_value) if study.best_trial else None,
            "elapsed_sec": elapsed,
            "n_trials": len(study.trials),
            **{k: combined[k] for k in ("objective", "fidelity", "utility", "privacy_risk")},
            **{k: fidelity.get(k) for k in ("ks_similarity", "corr_similarity")},
            **{k: privacy.get(k) for k in ("mia_auc", "nndr", "dcr")},
            **{k: utility.get(k) for k in ("mean_f1_gap", "mean_tstr_f1", "mean_trtr_f1")},
            "run_dir": str(out),
            "error": None,
        }
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        mark_failed(out, dataset=dataset_key, generator=generator_name, error=str(exc), elapsed_sec=elapsed)
        return {
            "dataset": dataset_key,
            "generator": generator_name,
            "status": "failed",
            "best_value": None,
            "elapsed_sec": elapsed,
            "n_trials": 0,
            "objective": None,
            "fidelity": None,
            "utility": None,
            "privacy_risk": None,
            "run_dir": str(out),
            "error": f"{type(exc).__name__}: {exc}",
        }
