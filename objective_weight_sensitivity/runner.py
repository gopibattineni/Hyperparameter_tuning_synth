"""Run independent Standard-TPE Optuna studies per objective-weight preset."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any, Sequence

import optuna
from optuna.trial import TrialState

from config_utils import PACKAGE_ROOT, load_generator_config
from datasets.loader import load_train_test
from evaluation.objective_presets import resolve_preset
from generators.registry import get_generator_class
from multiseed_comparison.objective_wrap import TimedTrialObjective
from multiseed_comparison.samplers import make_sampler
from multiseed_comparison.seeding import set_global_seed
from objective_weight_sensitivity import N_STARTUP_TRIALS, SAMPLER_NAME

# Ensure generator registry is populated.
import generators  # noqa: F401


def default_results_root() -> Path:
    return PACKAGE_ROOT / "results" / "objective_weight_sensitivity"


def study_dir(
    results_root: Path,
    dataset: str,
    generator: str,
    config: str,
    seed: int,
) -> Path:
    return (
        Path(results_root)
        / dataset
        / generator
        / str(config).upper()
        / f"seed{int(seed)}"
    )


def study_name(dataset: str, generator: str, config: str, seed: int) -> str:
    return f"{dataset}__{generator}__{config.upper()}__seed{int(seed)}__{SAMPLER_NAME}"


def _n_complete(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)


def _host_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "sampler": SAMPLER_NAME,
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    return meta


def run_one_study(
    *,
    dataset: str,
    generator: str,
    config: str,
    seed: int,
    n_trials: int,
    results_root: Path,
    n_startup_trials: int = N_STARTUP_TRIALS,
    trial_timeout_s: float | None = 7200.0,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Independent Optuna study for one (dataset, generator, weight preset, seed)."""
    preset = str(config).upper()
    objective_cfg = resolve_preset(preset, force_sdv_only=True)
    weights = dict(objective_cfg["weights"])

    out = study_dir(results_root, dataset, generator, preset, seed)
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / "optuna.db"
    summary_path = out / "summary.json"
    meta_path = out / "meta.json"

    # Resume skip when already complete
    if summary_path.is_file():
        try:
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            prior = {}
        prior_n = prior.get("n_complete")
        prior_status = str(prior.get("status") or "")
        if (
            prior_n is not None
            and int(prior_n) >= int(n_trials)
            and prior_status in {"completed", "skipped_resume"}
            and prior.get("best_objective") is not None
        ):
            prior = dict(prior)
            prior["status"] = "skipped_resume"
            meta_path.write_text(json.dumps(prior, indent=2, default=str), encoding="utf-8")
            return prior

    set_global_seed(seed)
    train_df, test_df, metadata = load_train_test(dataset)
    gen_cfg = load_generator_config(generator)
    gen_cls = get_generator_class(generator)

    name = study_name(dataset, generator, preset, seed)
    storage = f"sqlite:///{db_path.resolve()}"
    sampler = make_sampler(SAMPLER_NAME, seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
    )

    host = _host_meta()
    meta: dict[str, Any] = {
        "dataset": dataset,
        "generator": generator,
        "config": preset,
        "preset": preset,
        "sampler": SAMPLER_NAME,
        "seed": int(seed),
        "n_trials_target": int(n_trials),
        "n_startup_trials": int(n_startup_trials),
        "study_name": name,
        "weights": weights,
        "fidelity_aggregation": (objective_cfg.get("fidelity") or {}).get("aggregation"),
        "require_sdv": (objective_cfg.get("fidelity") or {}).get("require_sdv"),
        "direction": "maximize",
        "objective_formula": "J = wF*F + wU*U - wP*P",
        "utility_formulation": "unchanged_trtr_tstr_gap",
        "privacy_formulation": "unchanged_0.5_mia_0.3_nndr_0.2_dcr",
        "data_split": "fixed_yaml_80_20_no_val",
        "run_dir": str(out),
        **host,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    n_done = _n_complete(study)
    n_remaining = max(0, int(n_trials) - n_done)
    summary: dict[str, Any] = {
        **meta,
        "n_complete_before": n_done,
    }

    if n_done >= int(n_trials):
        summary["status"] = "skipped_resume"
        summary["n_complete"] = n_done
        if n_done:
            best = study.best_trial
            summary["best_objective"] = (
                float(best.value) if best.value is not None else None
            )
            summary["best_params"] = dict(best.params)
            summary["best_fidelity"] = best.user_attrs.get("fidelity")
            summary["best_utility"] = best.user_attrs.get("utility")
            summary["best_privacy_risk"] = best.user_attrs.get("privacy_risk")
            summary["best_mia_auc"] = best.user_attrs.get("mia_auc")
            summary["best_nndr"] = best.user_attrs.get("nndr")
            summary["best_dcr"] = best.user_attrs.get("dcr")
            summary["best_sdv_quality"] = best.user_attrs.get("sdv_quality")
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        return summary

    objective = TimedTrialObjective(
        generator_cls=gen_cls,
        generator_name=generator,
        train_df=train_df,
        test_df=test_df,
        metadata=metadata,
        generator_cfg=gen_cfg,
        objective_cfg=objective_cfg,
        study_seed=seed,
        trial_timeout_s=trial_timeout_s,
    )

    t0 = time.perf_counter()
    study.optimize(
        objective,
        n_trials=n_remaining,
        show_progress_bar=show_progress,
        catch=(Exception,),
    )
    wall = time.perf_counter() - t0

    n_complete = _n_complete(study)
    n_fail = sum(1 for t in study.trials if t.state == TrialState.FAIL)
    n_pruned = sum(1 for t in study.trials if t.state == TrialState.PRUNED)

    best_value = None
    best_params = None
    best_attrs: dict[str, Any] = {}
    if n_complete:
        best = study.best_trial
        best_value = float(best.value) if best.value is not None else None
        best_params = dict(best.params)
        best_attrs = dict(best.user_attrs or {})

    summary.update(
        {
            "status": "completed" if n_complete >= int(n_trials) else "partial",
            "n_complete": n_complete,
            "n_fail": n_fail,
            "n_pruned": n_pruned,
            "n_attempted": len(study.trials),
            "best_objective": best_value,
            "best_params": best_params,
            "best_fidelity": best_attrs.get("fidelity"),
            "best_utility": best_attrs.get("utility"),
            "best_privacy_risk": best_attrs.get("privacy_risk"),
            "best_mia_auc": best_attrs.get("mia_auc"),
            "best_nndr": best_attrs.get("nndr"),
            "best_dcr": best_attrs.get("dcr"),
            "best_sdv_quality": best_attrs.get("sdv_quality"),
            "best_ks_similarity": best_attrs.get("ks_similarity"),
            "best_corr_similarity": best_attrs.get("corr_similarity"),
            "study_wall_clock_s": float(wall),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    meta_path.write_text(json.dumps({**meta, **summary}, indent=2, default=str), encoding="utf-8")
    return summary


def run_grid(
    *,
    datasets: Sequence[str],
    generators: Sequence[str],
    configs: Sequence[str],
    seed: int,
    n_trials: int,
    results_root: Path | None = None,
    n_startup_trials: int = N_STARTUP_TRIALS,
    trial_timeout_s: float | None = 7200.0,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    root = Path(results_root) if results_root else default_results_root()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    jobs = [
        (ds, gen, cfg)
        for ds in datasets
        for gen in generators
        for cfg in configs
    ]
    total = len(jobs)
    for i, (ds, gen, cfg) in enumerate(jobs, start=1):
        print(
            f"[{i}/{total}] {ds} × {gen} × {cfg} × seed={seed} "
            f"(n_trials={n_trials}, sampler={SAMPLER_NAME})",
            flush=True,
        )
        try:
            summary = run_one_study(
                dataset=ds,
                generator=gen,
                config=cfg,
                seed=seed,
                n_trials=n_trials,
                results_root=root,
                n_startup_trials=n_startup_trials,
                trial_timeout_s=trial_timeout_s,
                show_progress=show_progress,
            )
            print(
                f"  → status={summary.get('status')} "
                f"best={summary.get('best_objective')} "
                f"complete={summary.get('n_complete')}",
                flush=True,
            )
        except Exception as exc:
            summary = {
                "dataset": ds,
                "generator": gen,
                "config": str(cfg).upper(),
                "seed": int(seed),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            fail_dir = study_dir(root, ds, gen, cfg, seed)
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            print(f"  FAILED: {summary['error']}", flush=True)
        rows.append(summary)
    return rows
