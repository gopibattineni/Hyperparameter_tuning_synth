#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/regression/13_concrete"
LOG="${DIR}/concrete_6gen_run.log"
PIDFILE="${DIR}/concrete_6gen.pid"

cd "$WORKDIR"
mkdir -p "$DIR"

if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi

echo "==== start $(date -Is) concrete 6 generators (1000 samples, regression) ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
  --datasets concrete \
  --generators gaussian_copula copulagan ctgan tvae ctabgan wgan_gp \
  --n-trials 20 \
  --resume \
  >> "$LOG" 2>&1
status=$?
echo "==== finished $(date -Is) exit=${status} ====" >> "$LOG"

if [ "$status" -eq 0 ]; then
  echo "==== writing HPO reports $(date -Is) ====" >> "$LOG"
  python3 -u "${DIR}/write_concrete_reports.py" >> "$LOG" 2>&1 || \
    echo "==== report write failed $(date -Is) ====" >> "$LOG"
fi

exit "$status"
