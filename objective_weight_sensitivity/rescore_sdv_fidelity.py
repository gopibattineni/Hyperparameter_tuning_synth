#!/usr/bin/env python3
"""Re-score SDV QualityReport for historical O1 trials (F = SDV; KS not in F).

Workers claim lines from a TSV queue built by ``build_sdv_rescore_queue.py``.
Each line: dataset, generator, sampler, seed, trial_number.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import time
import traceback
from pathlib import Path

import optuna
import pandas as pd
from optuna.trial import TrialState

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDIES = _ROOT / "results/multiseed_20261027/studies"
DEFAULT_CACHE = (
    _ROOT
    / "results/objective_weight_sensitivity/tables/o1_posthoc_reweight"
    / "sdv_rescore_cache.csv"
)
DEFAULT_QUEUE = (
    _ROOT
    / "results/objective_weight_sensitivity/tables/o1_posthoc_reweight"
    / "sdv_rescore_queue.tsv"
)


def _claim_line(queue: Path) -> str | None:
    if not queue.is_file():
        return None
    with queue.open("r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            lines = f.readlines()
            while lines and not lines[0].strip():
                lines.pop(0)
            if not lines:
                return None
            claimed = lines.pop(0).strip()
            f.seek(0)
            f.truncate()
            f.writelines(lines)
            f.flush()
            os.fsync(f.fileno())
            return claimed
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _append_rows(cache: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    lock = cache.with_suffix(cache.suffix + ".lock")
    with lock.open("a+", encoding="utf-8") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            write_header = (not cache.is_file()) or cache.stat().st_size == 0
            df.to_csv(cache, mode="a", header=write_header, index=False)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _load_trial(
    studies_root: Path,
    dataset: str,
    generator: str,
    sampler: str,
    seed: int,
    trial_number: int,
):
    db = studies_root / dataset / generator / sampler / f"seed{seed}" / "optuna.db"
    name = f"{dataset}__{generator}__{sampler}__seed{seed}"
    study = optuna.load_study(study_name=name, storage=f"sqlite:///{db.resolve()}")
    for t in study.trials:
        if int(t.number) == int(trial_number) and t.state == TrialState.COMPLETE:
            return t
    return None


def _rescore_one(
    studies_root: Path,
    dataset: str,
    generator: str,
    sampler: str,
    seed: int,
    trial_number: int,
) -> dict:
    from datasets.loader import load_train_test
    from evaluation.fidelity import evaluate_fidelity
    from generators.registry import get_generator_class
    from multiseed_comparison.seeding import set_global_seed

    t0 = time.perf_counter()
    out: dict = {
        "dataset": dataset,
        "generator": generator,
        "sampler": sampler,
        "seed": seed,
        "trial_number": trial_number,
        "utility": None,
        "privacy_risk": None,
        "fidelity_legacy": None,
        "ks_similarity_stored": None,
        "corr_similarity_stored": None,
        "fidelity": None,
        "sdv_quality": None,
        "ks_similarity": None,
        "corr_similarity": None,
        "sdv_unavailable": True,
        "error": None,
        "duration_s": None,
    }
    try:
        trial = _load_trial(
            studies_root, dataset, generator, sampler, seed, trial_number
        )
        if trial is None:
            out["error"] = "trial_not_found"
            out["duration_s"] = round(time.perf_counter() - t0, 3)
            return out
        attrs = dict(trial.user_attrs or {})
        out["utility"] = attrs.get("utility")
        out["privacy_risk"] = attrs.get("privacy_risk")
        out["fidelity_legacy"] = attrs.get("fidelity")
        out["ks_similarity_stored"] = attrs.get("ks_similarity")
        out["corr_similarity_stored"] = attrs.get("corr_similarity")

        trial_seed = attrs.get("trial_seed")
        ts = (
            int(trial_seed)
            if trial_seed is not None
            else int(seed) * 100000 + int(trial_number)
        )
        set_global_seed(ts)
        train_df, _test_df, metadata = load_train_test(dataset)
        cls = get_generator_class(generator)
        gen = cls.from_params(dict(trial.params or {}))
        try:
            gen.fit(train_df, metadata)
            synth = gen.sample(n=len(train_df), seed=ts)
        finally:
            if hasattr(gen, "release"):
                try:
                    gen.release()
                except Exception:
                    pass
        for col in train_df.columns:
            if col not in synth.columns:
                synth[col] = pd.NA
        synth = synth[train_df.columns]
        fid = evaluate_fidelity(
            train_df,
            synth,
            metadata,
            {"fidelity": {"aggregation": "sdv_only", "require_sdv": True}},
        )
        out["fidelity"] = fid.get("fidelity")
        out["sdv_quality"] = fid.get("sdv_quality")
        out["ks_similarity"] = fid.get("ks_similarity")
        out["corr_similarity"] = fid.get("corr_similarity")
        out["sdv_unavailable"] = bool(fid.get("sdv_unavailable"))
        if out["sdv_unavailable"]:
            out["error"] = "sdv_unavailable"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-1500:]
    out["duration_s"] = round(time.perf_counter() - t0, 3)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--studies-root", type=Path, default=DEFAULT_STUDIES)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    p.add_argument("--gpu", type=str, default=None)
    p.add_argument("--max-jobs", type=int, default=0)
    p.add_argument("--flush-every", type=int, default=1)
    args = p.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        os.environ["SYNTH_REQUIRE_CUDA"] = "1"

    print(f"worker queue={args.queue} cache={args.cache} gpu={args.gpu}", flush=True)
    buf: list[dict] = []
    ok = fail = done = 0
    while True:
        if args.max_jobs and done >= args.max_jobs:
            break
        claimed = _claim_line(args.queue)
        if not claimed:
            break
        parts = claimed.split("\t")
        if len(parts) != 5:
            print(f"skip bad line {claimed!r}", flush=True)
            continue
        dataset, generator, sampler, seed_s, trial_s = parts
        row = _rescore_one(
            args.studies_root,
            dataset,
            generator,
            sampler,
            int(seed_s),
            int(trial_s),
        )
        buf.append(row)
        done += 1
        if row.get("error"):
            fail += 1
        else:
            ok += 1
        if len(buf) >= max(1, args.flush_every):
            _append_rows(args.cache, buf)
            buf = []
        if done % 5 == 0 or row.get("error"):
            print(
                f"[done={done} ok={ok} fail={fail}] "
                f"{dataset}×{generator}×{sampler} seed={seed_s} trial={trial_s} "
                f"sdv={row.get('sdv_quality')} err={row.get('error')} "
                f"({row.get('duration_s')}s)",
                flush=True,
            )
    if buf:
        _append_rows(args.cache, buf)
    print(f"worker exit done={done} ok={ok} fail={fail}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
