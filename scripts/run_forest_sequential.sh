#!/usr/bin/env bash
# Run remaining ForestDiffusion pairs ONE AT A TIME (host has ~23 GiB RAM;
# each forest fit peaks ~5–8 GiB — 3–4 concurrent workers OOM-kill the box).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"
export SYNTH_CUDA_RESERVE_MB=0
export FOREST_XGB_NTHREAD=1
LOGDIR="$ROOT/sampler_comparison/packed_gpu/logs"
PIDDIR="$ROOT/sampler_comparison/packed_gpu/pids"
mkdir -p "$LOGDIR" "$PIDDIR"

# Prefer free GPUs: 0,1,2,4 (skip busy foreign cards when possible)
GPUS=(${FOREST_GPUS:-0 1 2 4})

fail_running() {
  "$PY" - <<'PY'
import optuna
from optuna.trial import TrialState
from pathlib import Path
for root in ("multivariate_tpe", "gaussian_process"):
    for db in Path(root).glob("results/classification/*/forest_diffusion/optuna.db"):
        storage = f"sqlite:///{db.resolve()}"
        try:
            st = optuna.study.get_all_study_summaries(storage=storage)
            if not st:
                continue
            study = optuna.load_study(study_name=st[0].study_name, storage=storage)
            n = 0
            for t in study.trials:
                if t.state == TrialState.RUNNING:
                    study._storage.set_trial_state_values(t._trial_id, TrialState.FAIL)
                    n += 1
            if n:
                print(f"{db}: failed {n} RUNNING")
        except Exception as e:
            print(f"{db}: {e}")
PY
}

is_done() {
  local sampler="$1" ds="$2"
  "$PY" - <<PY
from pathlib import Path
from tuning.artifacts import is_completed
import sys
sys.exit(0 if is_completed(Path("$sampler")/"results", "$ds", "forest_diffusion") else 1)
PY
}

run_one() {
  local sampler="$1" ds="$2" gpu="$3"
  local tag="${sampler}_${ds}_forest_gpu${gpu}"
  local log="$LOGDIR/${tag}.out"
  local pidf="$PIDDIR/${tag}.pid"
  if is_done "$sampler" "$ds"; then
    echo "skip completed $sampler $ds"
    return 0
  fi
  echo "$(date -Is) START $sampler $ds on GPU $gpu (sequential)"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u "$ROOT/${sampler}/run.py" \
    --datasets "$ds" --generators forest_diffusion --quiet-trials \
    >>"$log" 2>&1 &
  local pid=$!
  echo "$pid" >"$pidf"
  wait "$pid"
  local rc=$?
  echo "$(date -Is) END $sampler $ds rc=$rc"
  return 0
}

# Stop any leftover parallel forest workers
pkill -f 'Hyperparameter_tuning_synth/.*/run.py --datasets .* --generators forest_diffusion' 2>/dev/null || true
sleep 2
fail_running

JOBS=(
  "multivariate_tpe cancer ${GPUS[0]}"
  "multivariate_tpe alzheimers ${GPUS[1]:-1}"
  "gaussian_process cancer ${GPUS[2]:-2}"
  "gaussian_process alzheimers ${GPUS[3]:-4}"
)

LOG="$LOGDIR/forest_sequential_master.out"
{
  echo "$(date -Is) sequential forest launcher"
  for job in "${JOBS[@]}"; do
    set -- $job
    run_one "$1" "$2" "$3"
  done
  echo "$(date -Is) all forest jobs finished"
  "$PY" -u scripts/compare_bayesian_samplers.py || true
} >>"$LOG" 2>&1 &
echo $! >"$PIDDIR/forest_sequential.pid"
disown || true
echo "sequential forest launcher pid=$(cat "$PIDDIR/forest_sequential.pid") log=$LOG"
