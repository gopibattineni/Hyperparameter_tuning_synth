#!/usr/bin/env bash
# Resume CDC Diabetes: fast CTAB-GAN (10 trials) → WGAN-GP (20) → reports.
# Detached under systemd via setsid — survives laptop lock.
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/classification/7_cdc_diabetes"
LOG="${DIR}/cdc_diabetes_6gen_run.log"

cd "$WORKDIR"
mkdir -p "$DIR"

echo "==== $(date -Is) kill leftover cdc_diabetes workers (if any) ====" >> "$LOG"
pkill -f 'run_experiments.py --datasets cdc_diabetes' 2>/dev/null || true

# Fast CTAB-GAN profile
cp config/generators/ctabgan_fast.yaml config/generators/ctabgan.yaml
echo "==== start $(date -Is) cdc_diabetes × ctabgan FAST (10 trials) ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
  --datasets cdc_diabetes \
  --generators ctabgan \
  --n-trials 10 \
  --resume \
  >> "$LOG" 2>&1
status_ctab=$?
echo "==== finished ctabgan $(date -Is) exit=${status_ctab} ====" >> "$LOG"

# Restore full CTAB-GAN config for other datasets
if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi

echo "==== start $(date -Is) cdc_diabetes × wgan_gp (20 trials) ====" >> "$LOG"
python3 -u scripts/run_experiments.py \
  --datasets cdc_diabetes \
  --generators wgan_gp \
  --n-trials 20 \
  --resume \
  >> "$LOG" 2>&1
status_wgan=$?
echo "==== finished wgan_gp $(date -Is) exit=${status_wgan} ====" >> "$LOG"

if [ "$status_ctab" -eq 0 ] && [ "$status_wgan" -eq 0 ]; then
  echo "==== writing HPO reports $(date -Is) ====" >> "$LOG"
  python3 -u "${DIR}/write_cdc_diabetes_reports.py" >> "$LOG" 2>&1 || \
    echo "==== report write failed $(date -Is) ====" >> "$LOG"
fi

exit $(( status_ctab != 0 ? status_ctab : status_wgan ))
