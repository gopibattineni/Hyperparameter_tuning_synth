#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/classification/4_forest_cover"
LOG="${DIR}/ctabgan_fast_run.log"
cd "$WORKDIR"
cp config/generators/ctabgan_fast.yaml config/generators/ctabgan.yaml
echo "==== start $(date -Is) forest_cover × ctabgan FAST (10 trials) NEW discrete encoding ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
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
