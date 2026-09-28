#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/regression/11_online_shopping"
LOG="${DIR}/online_shopping_6gen_fair_rerun.log"
PIDFILE="${DIR}/online_shopping_8gen.pid"

cd "$WORKDIR"
mkdir -p "$DIR"

if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi

echo "==== start $(date -Is) online_shopping FULL 6-gen fair rerun (drop session ID + page 2; discrete day/order) ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
  --datasets online_shopping \
  --generators gaussian_copula copulagan ctgan tvae ctabgan wgan_gp tabddpm \
  --n-trials 20 \
  --no-resume \
  >> "$LOG" 2>&1
status=$?
echo "==== finished $(date -Is) exit=${status} ====" >> "$LOG"

if [ "$status" -eq 0 ]; then
  echo "==== writing HPO reports $(date -Is) ====" >> "$LOG"
  python3 -u "${DIR}/write_online_shopping_reports.py" >> "$LOG" 2>&1 || \
    echo "==== report write failed $(date -Is) ====" >> "$LOG"
fi

exit "$status"
