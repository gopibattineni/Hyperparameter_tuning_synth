#!/usr/bin/env bash
set -euo pipefail
WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
DIR="${WORKDIR}/results/classification/5_bank_marketing"
LOG="${DIR}/bank_marketing_8gen_run.log"
PIDFILE="${DIR}/bank_marketing_8gen.pid"

cd "$WORKDIR"
mkdir -p "$DIR"

if [ -f config/generators/ctabgan_full.yaml ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
fi

echo "==== start $(date -Is) bank_marketing 8 generators (1000 balanced samples) ====" >> "$LOG"
exec python3 -u scripts/run_experiments.py \
  --datasets bank_marketing \
  --generators gaussian_copula copulagan ctgan tvae ctabgan wgan_gp forest_diffusion tabddpm \
  --n-trials 20 \
  --resume \
  >> "$LOG" 2>&1
