#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/regression/10_metro_interstate"
LOG="${DIR}/metro_interstate_ctabgan_rerun.log"

cd "$WORKDIR"
mkdir -p "$DIR"

if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi

echo "==== start $(date -Is) metro_interstate CTAB-GAN rerun (date_time dropped) ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
  --datasets metro_interstate \
  --generators ctabgan \
  --n-trials 20 \
  --resume \
  >> "$LOG" 2>&1
status=$?
echo "==== finished $(date -Is) exit=${status} ====" >> "$LOG"

if [ "$status" -eq 0 ] && [ -f "${DIR}/ctabgan/completed.json" ]; then
  echo "==== writing HPO reports $(date -Is) ====" >> "$LOG"
  python3 -u "${DIR}/write_metro_interstate_reports.py" >> "$LOG" 2>&1 || \
    echo "==== report write failed $(date -Is) ====" >> "$LOG"
fi

exit "$status"
