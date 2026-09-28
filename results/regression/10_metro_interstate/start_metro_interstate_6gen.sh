#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/regression/10_metro_interstate"
LOG="${DIR}/metro_interstate_6gen_nodatetime_run.log"
PIDFILE="${DIR}/metro_interstate_8gen.pid"

cd "$WORKDIR"
mkdir -p "$DIR"

if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi

echo "==== start $(date -Is) metro_interstate FULL 6-gen rerun (date_time dropped) ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
  --datasets metro_interstate \
  --generators gaussian_copula copulagan ctgan tvae ctabgan wgan_gp forest_diffusion tabddpm \
  --n-trials 20 \
  --no-resume \
  >> "$LOG" 2>&1
status=$?
echo "==== finished $(date -Is) exit=${status} ====" >> "$LOG"

if [ "$status" -eq 0 ]; then
  echo "==== writing HPO reports $(date -Is) ====" >> "$LOG"
  python3 -u "${DIR}/write_metro_interstate_reports.py" >> "$LOG" 2>&1 || \
    echo "==== report write failed $(date -Is) ====" >> "$LOG"
fi

exit "$status"
