#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DIR="${SCRIPT_DIR}"
LOG="${DIR}/ctabgan_fast_run.log"
PYTHON="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"
cd "$WORKDIR"
cp config/generators/ctabgan_fast.yaml config/generators/ctabgan.yaml
echo "==== start $(date -Is) forest_cover × ctabgan FAST (10 trials) NEW discrete encoding ====" >> "$LOG"
"$PYTHON" -u scripts/run_experiments.py \
  --datasets forest_cover \
  --generators ctabgan \
  --n-trials 10 \
  --resume \
  >> "$LOG" 2>&1
status=$?
echo "==== finished ctabgan $(date -Is) exit=${status} ====" >> "$LOG"
# restore full config for other datasets
if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi
exit "$status"
