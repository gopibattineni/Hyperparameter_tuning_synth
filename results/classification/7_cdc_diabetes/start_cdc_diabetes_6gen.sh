#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/classification/7_cdc_diabetes"
LOG="${DIR}/cdc_diabetes_8gen_run.log"
PIDFILE="${DIR}/cdc_diabetes_8gen.pid"

cd "$WORKDIR"
mkdir -p "$DIR"

if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi

echo "==== start $(date -Is) cdc_diabetes 7 generators (1000 balanced samples) ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
  --datasets cdc_diabetes \
  --generators gaussian_copula copulagan ctgan tvae ctabgan wgan_gp tabddpm \
  --n-trials 20 \
  --resume \
  >> "$LOG" 2>&1
status=$?
echo "==== finished $(date -Is) exit=${status} ====" >> "$LOG"

if [ "$status" -eq 0 ]; then
  echo "==== writing HPO reports $(date -Is) ====" >> "$LOG"
  python3 -u "${DIR}/write_cdc_diabetes_reports.py" >> "$LOG" 2>&1 || \
    echo "==== report write failed $(date -Is) ====" >> "$LOG"
fi

exit "$status"
