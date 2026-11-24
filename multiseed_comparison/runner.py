"""Run multi-seed Optuna studies and write raw trial CSVs."""

from __future__ import annotations

import json
import os
import socket
import time
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import optuna
import pandas as pd
from optuna.trial import TrialState

from config_utils import load_config, load_generator_config
from datasets.loader import load_dataset_config, load_train_test
from generators.registry import get_generator_class
from multiseed_comparison import (
    DEFAULT_N_TRIALS,
    N_STARTUP_TRIALS,
    SAMPLER_NAMES,
    SEEDS,
)
from evaluation.objective_presets import resolve_preset
from multiseed_comparison.objective_wrap import TimedTrialObjective
from multiseed_comparison.samplers import ensure_gp_torch_compat, make_sampler
from multiseed_comparison.seeding import set_global_seed
from multivariate_tpe.datasets_catalog import GENERATORS_NO_FOREST


def default_results_root(base: Path | None = None) -> Path:
    """``results/multiseed_YYYYMMDD`` under the package root (or ``base``)."""
    from config_utils import PACKAGE_ROOT

    root = Path(base) if base else PACKAGE_ROOT / "results"
    return root / f"multiseed_{date.today().strftime('%Y%m%d')}"


def study_name(dataset: str, generator: str, sampler: str, seed: int) -> str:
    return f"{dataset}__{generator}__{sampler}__seed{seed}"


def study_dir(results_root: Path, dataset: str, generator: str, sampler: str, seed: int) -> Path:
    return (
        Path(results_root)
        / "studies"
        / dataset
        / generator
        / sampler
        / f"seed{seed}"
    )


def _n_complete(study: optuna.Study) -> int:
    return sum(1 for t in study.trials if t.state == TrialState.COMPLETE)


def _host_meta() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        meta["torch_version"] = torch.__version__
        meta["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["gpu_count_visible"] = torch.cuda.device_count()
    except Exception:
        pass
    return meta


def _trial_row(
    trial: optuna.FrozenTrial,
    *,
    dataset: str,
    generator: str,
    sampler: str,
    seed: int,
) -> dict[str, Any]:
    attrs = dict(trial.user_attrs or {})
    row: dict[str, Any] = {
        "dataset": dataset,
        "generator": generator,
        "sampler": sampler,
        "seed": seed,
        "trial_number": trial.number,
        "state": trial.state.name,
        "objective": trial.value if trial.state == TrialState.COMPLETE else None,
        "fidelity": attrs.get("fidelity"),
        "sdv_quality": attrs.get("sdv_quality"),
        "ks_similarity": attrs.get("ks_similarity"),
        "corr_similarity": attrs.get("corr_similarity"),
        "fidelity_aggregation": attrs.get("fidelity_aggregation"),
        "utility": attrs.get("utility"),
        "privacy_risk": attrs.get("privacy_risk"),
        "duration_s": attrs.get("duration_s"),
        "trial_seed": attrs.get("trial_seed"),
        "error": attrs.get("error"),
    }
    for k, v in (trial.params or {}).items():
        row[f"param_{k}"] = v
    return row


def write_raw_trials_csv(
    study: optuna.Study,
    path: Path,
    *,
    dataset: str,
    generator: str,
    sampler: str,
    seed: int,
) -> Path:
    rows = [
        _trial_row(t, dataset=dataset, generator=generator, sampler=sampler, seed=seed)
        for t in study.trials
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_one_study(
    *,
    dataset: str,
    generator: str,
    sampler_name: str,
    seed: int,
    results_root: Path,
    n_trials: int = DEFAULT_N_TRIALS,
    n_startup_trials: int = N_STARTUP_TRIALS,
    trial_timeout_s: float | None = 7200.0,
    show_progress: bool = False,
    objective_preset: str | None = None,
    force_sdv_only: bool = True,
) -> dict[str, Any]:
    """Run or resume one study; write raw CSV + meta.json under the study dir.

    ``objective_preset`` (O1–O5) swaps weights. Fidelity defaults to
    ``sdv_only`` (KS is diagnostic only). Pass ``force_sdv_only=False`` only
    to replay legacy_composite.
    """
    out = study_dir(results_root, dataset, generator, sampler_name, seed)
    out.mkdir(parents=True, exist_ok=True)
    raw_csv = results_root / "raw" / f"{study_name(dataset, generator, sampler_name, seed)}.csv"
    db_path = out / "optuna.db"
    storage = f"sqlite:///{db_path.resolve()}"
    name = study_name(dataset, generator, sampler_name, seed)

    meta_path = out / "meta.json"
    summary_path = out / "summary.json"
    host = _host_meta()

    if sampler_name in {"gp", "gpsampler", "gaussian_process"}:
        note = ensure_gp_torch_compat()
        if note:
            host["gp_torch_compat"] = note

    # Honor imported/legacy summaries that already meet the trial budget
    # (they may have raw CSV + summary.json but no local optuna.db).
    if summary_path.is_file():
        try:
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            prior = {}
        prior_n = prior.get("n_complete")
        prior_status = str(prior.get("status") or "")
        # Resume only if the stored preset matches (when a preset is requested).
        prior_preset = str(prior.get("objective_preset") or "").upper() or None
        want_preset = str(objective_preset).upper() if objective_preset else None
        preset_ok = want_preset is None or prior_preset in {None, want_preset}
        if (
            preset_ok
            and prior_n is not None
            and int(prior_n) >= int(n_trials)
            and prior_status in {"completed", "skipped_resume"}
            and prior.get("best_objective") is not None
        ):
            prior = dict(prior)
            prior["status"] = "skipped_resume"
            prior["n_complete_before"] = int(prior_n)
            prior["run_dir"] = str(out)
            prior.update({k: v for k, v in host.items() if k not in prior})
            meta_path.write_text(
                json.dumps(prior, indent=2, default=str), encoding="utf-8"
            )
            return prior

    set_global_seed(seed)
    if objective_preset:
        objective_cfg = resolve_preset(
            objective_preset, force_sdv_only=bool(force_sdv_only)
        )
    else:
        objective_cfg = load_config("objective")
    gen_cfg = load_generator_config(generator)
    cls = get_generator_class(generator)
    if hasattr(cls, "is_available") and not cls.is_available():
        raise RuntimeError(f"Generator {generator!r} dependencies unavailable")

    # Neural generators: refuse silent CPU fallback (gaussian_copula is CPU-only).
    from generators.device import CPU_ONLY_GENERATORS, assert_cuda_available

    if generator not in CPU_ONLY_GENERATORS:
        assert_cuda_available(context=f"{dataset}×{generator}")

    # Fixed YAML train/test split (constraint: do not change data splits).
    train_df, test_df, metadata = load_train_test(dataset)

    sampler = make_sampler(sampler_name, seed, n_startup_trials=n_startup_trials)
    study = optuna.create_study(
        study_name=name,
        storage=storage,
        load_if_exists=True,
        direction=str(objective_cfg.get("direction", "maximize")),
        sampler=sampler,
    )

    n_done = _n_complete(study)
    weights = dict(objective_cfg.get("weights") or {})
    summary: dict[str, Any] = {
        "dataset": dataset,
        "generator": generator,
        "sampler": sampler_name,
        "seed": seed,
        "study_name": name,
        "n_trials_target": int(n_trials),
        "n_complete_before": n_done,
        "run_dir": str(out),
        "objective_preset": (
            str(objective_preset).upper()
            if objective_preset
            else str(objective_cfg.get("preset") or "DEFAULT")
        ),
        "w_fidelity": weights.get("fidelity"),
        "w_utility": weights.get("utility"),
        "w_privacy_risk": weights.get("privacy_risk"),
        "fidelity_aggregation": (objective_cfg.get("fidelity") or {}).get(
            "aggregation"
        ),
        **host,
    }

    if n_done >= int(n_trials):
        summary["status"] = "skipped_resume"
        summary["n_complete"] = n_done
        # Mirror completed fields so aggregation can use this study.
        if n_done:
            best = study.best_trial
            summary["best_objective"] = (
                float(best.value) if best.value is not None else None
            )
            summary["best_trial_number"] = int(best.number)
            summary["best_fidelity"] = best.user_attrs.get("fidelity")
            summary["best_utility"] = best.user_attrs.get("utility")
            summary["best_privacy_risk"] = best.user_attrs.get("privacy_risk")
        write_raw_trials_csv(
            study,
            raw_csv,
            dataset=dataset,
            generator=generator,
            sampler=sampler_name,
            seed=seed,
        )
        payload = json.dumps(summary, indent=2, default=str)
        meta_path.write_text(payload, encoding="utf-8")
        summary_path.write_text(payload, encoding="utf-8")
        return summary

    objective = TimedTrialObjective(
        generator_cls=cls,
        generator_name=generator,
        train_df=train_df,
        test_df=test_df,
        metadata=metadata,
        generator_cfg=gen_cfg,
        objective_cfg=objective_cfg,
        study_seed=seed,
        trial_timeout_s=trial_timeout_s,
    )

    n_remaining = int(n_trials) - n_done
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
    best_number = None
    best_f = best_u = best_p = None
    if n_complete:
        best = study.best_trial
        best_value = float(best.value) if best.value is not None else None
        best_number = int(best.number)
        best_f = best.user_attrs.get("fidelity")
        best_u = best.user_attrs.get("utility")
        best_p = best.user_attrs.get("privacy_risk")

    durations = [
        float(t.user_attrs["duration_s"])
        for t in study.trials
        if t.user_attrs.get("duration_s") is not None
    ]

    summary.update(
        {
            "status": "completed" if n_complete >= int(n_trials) else "partial",
            "n_complete": n_complete,
            "n_fail": n_fail,
            "n_pruned": n_pruned,
            "n_attempted": len(study.trials),
            "best_objective": best_value,
            "best_trial_number": best_number,
            "best_fidelity": best_f,
            "best_utility": best_u,
            "best_privacy_risk": best_p,
            "study_wall_clock_s": wall,
            "mean_trial_duration_s": float(sum(durations) / len(durations)) if durations else None,
            "sum_trial_duration_s": float(sum(durations)) if durations else None,
        }
    )
    write_raw_trials_csv(
        study, raw_csv, dataset=dataset, generator=generator, sampler=sampler_name, seed=seed
    )
    meta_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def iter_jobs(
    datasets: Sequence[str],
    generators: Sequence[str],
    samplers: Sequence[str],
    seeds: Sequence[int],
) -> list[tuple[str, str, str, int]]:
    jobs: list[tuple[str, str, str, int]] = []
    for ds in datasets:
        for gen in generators:
            for samp in samplers:
                for seed in seeds:
                    jobs.append((ds, gen, samp, int(seed)))
    return jobs


def run_grid(
    *,
    results_root: Path,
    datasets: Sequence[str] | None = None,
    generators: Sequence[str] | None = None,
    samplers: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    n_startup_trials: int = N_STARTUP_TRIALS,
    trial_timeout_s: float | None = 7200.0,
    show_progress: bool = False,
    objective_preset: str | None = None,
    force_sdv_only: bool = True,
) -> list[dict[str, Any]]:
    """Run the full factorial grid sequentially (default)."""
    cfg = load_dataset_config()
    datasets = list(datasets) if datasets else list(cfg.keys())
    generators = list(generators) if generators else list(GENERATORS_NO_FOREST)
    samplers = list(samplers) if samplers else list(SAMPLER_NAMES)
    seeds = [int(s) for s in (seeds if seeds is not None else SEEDS)]

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "raw").mkdir(parents=True, exist_ok=True)

    jobs = iter_jobs(datasets, generators, samplers, seeds)
    rows: list[dict[str, Any]] = []
    print(
        f"multiseed grid: {len(jobs)} studies | datasets={datasets} "
        f"generators={generators} samplers={samplers} seeds={seeds} "
        f"preset={objective_preset or 'DEFAULT'} sdv_only={force_sdv_only} "
        f"n_trials={n_trials} timeout_s={trial_timeout_s} → {results_root}"
    )
    for i, (ds, gen, samp, seed) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {ds} × {gen} × {samp} × seed={seed}", flush=True)
        try:
            summary = run_one_study(
                dataset=ds,
                generator=gen,
                sampler_name=samp,
                seed=seed,
                results_root=results_root,
                n_trials=n_trials,
                n_startup_trials=n_startup_trials,
                trial_timeout_s=trial_timeout_s,
                show_progress=show_progress,
                objective_preset=objective_preset,
                force_sdv_only=force_sdv_only,
            )
        except Exception as exc:
            summary = {
                "dataset": ds,
                "generator": gen,
                "sampler": samp,
                "seed": seed,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  FAILED: {summary['error']}", flush=True)
            # Persist failure so resume/aggregation see it (no silent holes).
            fail_dir = study_dir(results_root, ds, gen, samp, seed)
            fail_dir.mkdir(parents=True, exist_ok=True)
            (fail_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
        rows.append(summary)
        print(
            f"  → status={summary.get('status')} best={summary.get('best_objective')} "
            f"complete={summary.get('n_complete')}",
            flush=True,
        )
    return rows
