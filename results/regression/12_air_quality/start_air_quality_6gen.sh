#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/regression/12_air_quality"
LOG="${DIR}/air_quality_6gen_run.log"
PIDFILE="${DIR}/air_quality_6gen.pid"

cd "$WORKDIR"
mkdir -p "$DIR"

if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi

echo "==== start $(date -Is) air_quality 6 generators (1000 samples, Date/Time dropped) ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
  --datasets air_quality \
  --generators gaussian_copula copulagan ctgan tvae ctabgan wgan_gp \
  --n-trials 20 \
  --resume \
  >> "$LOG" 2>&1
status=$?
echo "==== finished $(date -Is) exit=${status} ====" >> "$LOG"

if [ "$status" -eq 0 ]; then
  echo "==== writing HPO reports $(date -Is) ====" >> "$LOG"
  python3 -u "${DIR}/write_air_quality_reports.py" >> "$LOG" 2>&1 || \
    echo "==== report write failed $(date -Is) ====" >> "$LOG"
fi

exit "$status"
