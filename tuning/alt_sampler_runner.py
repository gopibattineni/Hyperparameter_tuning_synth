"""Shared runner for alternative Bayesian samplers (multivariate TPE, GP).

Keeps results under a dedicated ``results_root`` so the original standard-TPE
artefacts under ``results/`` are never overwritten.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from config_utils import PACKAGE_ROOT, load_config
from generators import list_generators
from generators.registry import get_generator_class
from tuning.artifacts import is_completed, run_dir, save_json
from tuning.optimizer import run_study

LOG = logging.getLogger("alt_sampler_runner")

# Full 8-generator research set (matches the original TPE baseline).
DEFAULT_GENERATORS = [
    "gaussian_copula",
    "copulagan",
    "ctgan",
    "ctabgan",
    "tvae",
    "wgan_gp",
    "forest_diffusion",
    "tabddpm",
]

DEFAULT_DATASETS = ["cancer", "alzheimers"]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


_CUDA_RESERVE_HOLD: list = []


def reserve_cuda_footprint(mb: int | None = None) -> str | None:
    """Pin ~``mb`` MiB of CUDA caching-allocator memory for this process.

    Used when packing several HPO workers onto one GPU so each card stays near
    the 2.0–2.3 GiB occupancy target even while a generator is between kernels.
    Set ``SYNTH_CUDA_RESERVE_MB=0`` to disable.
    """
    import os

    if mb is None:
        mb = int(os.environ.get("SYNTH_CUDA_RESERVE_MB", "0"))
    if mb <= 0:
        return None
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    # Allocate then drop the tensor so the caching allocator keeps the block
    # without holding a Python reference that fights model init.
    n = (mb * 1024 * 1024) // 4
    buf = torch.empty(n, dtype=torch.float32, device="cuda")
    # Touch once so the page is really reserved, then drop.
    buf.fill_(0)
    del buf
    # Keep a tiny live tensor so the context stays warm; reserved pool remains.
    _CUDA_RESERVE_HOLD.append(torch.empty(1, device="cuda"))
    reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
    return f"CUDA reserve target={mb}MiB; memory_reserved≈{reserved:.0f}MiB"


def apply_torch_device_compat() -> str | None:
    """Allow Optuna GPSampler to run on torch < 2.1.

    Optuna 5's ``GPSampler.sample_relative`` uses ``with torch.device("cpu"):``,
    which requires PyTorch ≥ 2.1. On older torch we monkey-patch that method to
    call ``_sample_relative_impl`` directly (still GPSampler — not a substitute).

    Avoid replacing ``torch.device`` itself: wrapper types break
    ``tensor.to(device)`` in CTGAN/TVAE/etc.
    """
    try:
        import torch
    except ImportError:
        return "torch not installed"

    try:
        with torch.device("cpu"):
            return None
    except AttributeError:
        pass

    try:
        from optuna.samplers._gp import sampler as gps_mod
        from optuna.trial import TrialState
        import json
    except ImportError as exc:
        return f"Could not patch GPSampler ({exc})"

    if getattr(gps_mod.GPSampler.sample_relative, "_synth_torch113_patched", False):
        return "GPSampler already patched for torch < 2.1"

    _orig = gps_mod.GPSampler.sample_relative

    def sample_relative(self, study, trial, search_space):  # type: ignore[no-untyped-def]
        if search_space == {}:
            return {}

        states = (TrialState.COMPLETE, TrialState.RUNNING)
        trials = study._get_trials(deepcopy=False, states=states, use_cache=False)
        completed_trials = [t for t in trials if t.state == TrialState.COMPLETE]
        running_trials = [
            t
            for t in trials
            if t.state == TrialState.RUNNING
            and search_space.keys() <= gps_mod._get_params(t).keys()
        ]

        if len(completed_trials) < self._n_startup_trials:
            return {}

        # Skip `with torch.device("cpu"):` — unsupported on torch < 2.1.
        params = self._sample_relative_impl(
            study, completed_trials, running_trials, search_space
        )

        if params != {}:
            params_str = json.dumps(params)
            max_len = gps_mod._SYSTEM_ATTR_MAX_LENGTH
            key = gps_mod._RELATIVE_PARAMS_KEY
            for i in range(0, len(params_str), max_len):
                study._storage.set_trial_system_attr(
                    trial._trial_id,
                    f"{key}:{i // max_len}",
                    params_str[i : i + max_len],
                )
        return params

    sample_relative._synth_torch113_patched = True  # type: ignore[attr-defined]
    sample_relative._synth_torch113_orig = _orig  # type: ignore[attr-defined]
    gps_mod.GPSampler.sample_relative = sample_relative  # type: ignore[method-assign]
    return (
        f"Patched Optuna GPSampler.sample_relative for torch {torch.__version__} "
        "(avoids torch.device context manager; still uses GPSampler)."
    )


def _resource_snapshot() -> dict[str, Any]:
    out: dict[str, Any] = {"pid": os.getpid()}
    try:
        import psutil

        proc = psutil.Process()
        mem = proc.memory_info()
        out.update(
            {
                "rss_mb": round(mem.rss / (1024**2), 2),
                "cpu_percent": proc.cpu_percent(interval=0.1),
                "system_cpu_percent": psutil.cpu_percent(interval=0.1),
                "system_mem_percent": psutil.virtual_memory().percent,
            }
        )
    except Exception as exc:
        out["resource_error"] = f"{type(exc).__name__}: {exc}"
    return out


def compute_cost_metrics(out_dir: Path, study_summary: Mapping[str, Any]) -> dict[str, Any]:
    """Derive wall-clock / trial-cost stats from trials.csv + run summary."""
    import optuna

    trials_path = out_dir / "trials.csv"
    cost: dict[str, Any] = {
        "elapsed_sec": study_summary.get("elapsed_sec"),
        "n_trials_attempted": study_summary.get("n_trials"),
        "best_value": study_summary.get("best_value"),
        "status": study_summary.get("status"),
        "resources_end": _resource_snapshot(),
    }
    if not trials_path.is_file():
        cost["note"] = "trials.csv missing"
        return cost

    df = pd.read_csv(trials_path)
    states = df["state"].astype(str) if "state" in df.columns else pd.Series(["COMPLETE"] * len(df))
    cost["n_complete"] = int((states == "COMPLETE").sum())
    cost["n_pruned"] = int((states == "PRUNED").sum())
    cost["n_fail"] = int((states == "FAIL").sum())
    cost["n_attempted"] = int(len(df))

    if "duration" in df.columns:
        # Optuna CSV duration looks like "0 days 00:01:07.877316"
        secs = pd.to_timedelta(df["duration"], errors="coerce").dt.total_seconds()
        complete_secs = secs[states == "COMPLETE"].dropna()
        cost["mean_trial_sec"] = float(complete_secs.mean()) if len(complete_secs) else None
        cost["median_trial_sec"] = float(complete_secs.median()) if len(complete_secs) else None
        cost["sum_trial_sec"] = float(complete_secs.sum()) if len(complete_secs) else None
        cost["trial_durations_sec"] = [
            None if pd.isna(v) else float(v) for v in secs.tolist()
        ]

    if "value" in df.columns and "number" in df.columns:
        complete = df[states == "COMPLETE"].copy()
        if len(complete):
            # Objective is maximized in this project.
            idx = complete["value"].idxmax()
            cost["best_trial_number"] = int(complete.loc[idx, "number"])
            cost["best_objective"] = float(complete.loc[idx, "value"])
            # Cumulative best vs trial index
            running = []
            best_so_far = float("-inf")
            for _, row in complete.sort_values("number").iterrows():
                best_so_far = max(best_so_far, float(row["value"]))
                running.append({"trial": int(row["number"]), "best_so_far": best_so_far})
            cost["cumulative_best"] = running

    # Failed / pruned error messages
    errors = []
    if "user_attrs_error" in df.columns:
        for _, row in df.iterrows():
            err = row.get("user_attrs_error")
            if isinstance(err, str) and err.strip() and err.lower() != "nan":
                errors.append({"trial": int(row.get("number", -1)), "state": row.get("state"), "error": err})
    cost["trial_errors"] = errors

    # Intermediate values: this pipeline does not call trial.report().
    cost["intermediate_values_reported"] = False
    cost["pruning_note"] = (
        "TrialObjective does not call trial.report()/should_prune(). "
        "MedianPruner therefore cannot prune mid-trial on intermediate metrics. "
        "Generator failures are recorded as PRUNED via TrialPruned. "
        "To enable true pruning, report intermediate fidelity/utility during fit "
        "and call trial.report(value, step) + trial.should_prune()."
    )
    # Silence unused import warning path for type checkers
    _ = optuna
    return cost


def resolve_n_trials(
    budget: Mapping[str, Any],
    dataset: str,
    generator: str,
    default: int = 20,
) -> int:
    """Per-pair trial budget; falls back to generator then global default."""
    by_pair = budget.get("by_pair") or {}
    key = f"{dataset}__{generator}"
    if key in by_pair:
        return int(by_pair[key])
    by_gen = budget.get("by_generator") or {}
    if generator in by_gen:
        return int(by_gen[generator])
    return int(budget.get("default", default))


def run_pairs(
    *,
    experiment_root: Path,
    optuna_cfg_path: Path,
    budget_path: Path | None = None,
    datasets: Sequence[str] | None = None,
    generators: Sequence[str] | None = None,
    n_trials_override: int | None = None,
    resume: bool = True,
    show_trial_progress: bool = True,
    force_torch_compat: bool = False,
) -> list[dict[str, Any]]:
    experiment_root = Path(experiment_root)
    results_root = experiment_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    log_dir = experiment_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "runner.log", encoding="utf-8"),
        ],
        force=True,
    )

    # Cap BLAS/OpenMP/Torch threads before any heavy import side-effects grow pools.
    try:
        from tuning.cpu_limits import apply_thread_limits

        LOG.info(apply_thread_limits(int(os.environ.get("SYNTH_NUM_THREADS", "1"))))
    except Exception as exc:
        LOG.warning("apply_thread_limits failed: %s", exc)

    if force_torch_compat:
        note = apply_torch_device_compat()
        if note:
            LOG.warning(note)

    reserve_note = reserve_cuda_footprint()
    if reserve_note:
        LOG.info(reserve_note)

    optuna_cfg = load_yaml(optuna_cfg_path)
    # Merge missing keys from global optuna.yaml (artifact flags etc.)
    base = load_config("optuna")
    merged = dict(base)
    merged.update(optuna_cfg)
    if "sampler" in optuna_cfg:
        merged["sampler"] = dict(base.get("sampler") or {})
        merged["sampler"].update(optuna_cfg["sampler"])
    if "pruner" in optuna_cfg:
        merged["pruner"] = dict(base.get("pruner") or {})
        merged["pruner"].update(optuna_cfg["pruner"])
    if "storage" in optuna_cfg:
        merged["storage"] = dict(optuna_cfg["storage"])
    if "artifact" in optuna_cfg:
        merged["artifact"] = dict(base.get("artifact") or {})
        merged["artifact"].update(optuna_cfg["artifact"])

    budget = load_yaml(budget_path) if budget_path and budget_path.is_file() else {}
    datasets = list(datasets or DEFAULT_DATASETS)
    generators = list(generators or DEFAULT_GENERATORS)

    registered = set(list_generators())
    rows: list[dict[str, Any]] = []
    master_path = results_root / "all_experiments.csv"

    for dataset in datasets:
        for generator in generators:
            if generator not in registered:
                LOG.warning("skip %s × %s (not registered)", dataset, generator)
                continue
            cls = get_generator_class(generator)
            if hasattr(cls, "is_available") and not cls.is_available():
                LOG.warning("skip %s × %s (dependencies unavailable)", dataset, generator)
                continue

            out = run_dir(results_root, dataset, generator)
            if resume and is_completed(results_root, dataset, generator):
                LOG.info("resume-skip completed %s × %s", dataset, generator)
                rows.append(
                    {
                        "dataset": dataset,
                        "generator": generator,
                        "status": "skipped_resume",
                        "run_dir": str(out),
                    }
                )
                continue

            n_trials = (
                int(n_trials_override)
                if n_trials_override is not None
                else resolve_n_trials(budget, dataset, generator, default=int(merged.get("n_trials", 20)))
            )
            LOG.info(
                "START %s × %s  n_trials=%s  sampler=%s  out=%s",
                dataset,
                generator,
                n_trials,
                (merged.get("sampler") or {}).get("name"),
                out,
            )
            t0 = time.perf_counter()
            resources_start = _resource_snapshot()
            try:
                summary = run_study(
                    generator,
                    dataset,
                    results_root=results_root,
                    n_trials=n_trials,
                    optuna_cfg=merged,
                    show_trial_progress=show_trial_progress,
                )
            except Exception as exc:
                LOG.exception("FAILED %s × %s", dataset, generator)
                summary = {
                    "dataset": dataset,
                    "generator": generator,
                    "status": "failed",
                    "best_value": None,
                    "elapsed_sec": time.perf_counter() - t0,
                    "n_trials": 0,
                    "run_dir": str(out),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                out.mkdir(parents=True, exist_ok=True)
                save_json(out / "failed.json", summary)

            cost = compute_cost_metrics(Path(summary.get("run_dir") or out), summary)
            cost["resources_start"] = resources_start
            cost["wall_clock_sec"] = time.perf_counter() - t0
            save_json(Path(summary.get("run_dir") or out) / "cost_metrics.json", cost)
            summary["cost"] = cost
            rows.append(summary)
            LOG.info(
                "DONE %s × %s status=%s best=%s elapsed=%.1fs",
                dataset,
                generator,
                summary.get("status"),
                summary.get("best_value"),
                float(summary.get("elapsed_sec") or 0.0),
            )

            # Append master CSV incrementally
            pd.DataFrame(
                [
                    {
                        k: summary.get(k)
                        for k in (
                            "dataset",
                            "generator",
                            "status",
                            "best_value",
                            "elapsed_sec",
                            "n_trials",
                            "objective",
                            "fidelity",
                            "utility",
                            "privacy_risk",
                            "run_dir",
                            "error",
                        )
                    }
                ]
            ).to_csv(
                master_path,
                mode="a",
                header=not master_path.is_file(),
                index=False,
            )

    save_json(results_root / "run_summary.json", {"rows": rows})
    return rows
