#!/usr/bin/env bash
# Run one dataset × generator pair on a single GPU.
# Usage: run_one_pair.sh <gpu_id> <dataset> <generator> [n_trials]
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <gpu_id> <dataset> <generator> [n_trials]" >&2
  exit 2
fi

GPU_ID="$1"
DATASET="$2"
GENERATOR="$3"
N_TRIALS="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

if [ -z "$N_TRIALS" ]; then
  case "$GENERATOR" in
    tabddpm|forest_diffusion|ctabgan) N_TRIALS=10 ;;
    *) N_TRIALS=20 ;;
  esac
fi

LOG_DIR="${WORKDIR}/results/gpu_parallel/logs"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${DATASET}__${GENERATOR}__gpu${GPU_ID}.log"

cd "$WORKDIR"
echo "==== $(date -Is) GPU=${GPU_ID} ${DATASET} × ${GENERATOR} n_trials=${N_TRIALS} ====" >> "$LOG"
set +e
"$PYTHON" -u scripts/run_experiments.py \
  --datasets "$DATASET" \
  --generators "$GENERATOR" \
  --n-trials "$N_TRIALS" \
  --resume \
  >> "$LOG" 2>&1
status=$?
set -e
echo "==== $(date -Is) GPU=${GPU_ID} ${DATASET} × ${GENERATOR} exit=${status} ====" >> "$LOG"
exit "$status"
