#!/usr/bin/env bash
# Forest Cover: TabDDPM (GPU) then ForestDiffusion (CPU, after CTAB-GAN), then 8-gen report.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DIR="${SCRIPT_DIR}"
LOG="${DIR}/diffusion_run.log"
PYTHON="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"
cd "$WORKDIR"

echo "==== start $(date -Is) forest_cover × tabddpm FAST (10 trials) ====" >> "$LOG"
cp config/generators/tabddpm_fast.yaml config/generators/tabddpm.yaml
set +e
"$PYTHON" -u scripts/run_experiments.py \
  --datasets forest_cover \
  --generators tabddpm \
  --n-trials 10 \
  --resume \
  >> "$LOG" 2>&1
status_tab=$?
set -e
echo "==== finished tabddpm $(date -Is) exit=${status_tab} ====" >> "$LOG"
if [ -f config/generators/tabddpm_full.yaml ]; then
  cp config/generators/tabddpm_full.yaml config/generators/tabddpm.yaml
fi

if [ -f "${DIR}/forest_diffusion/completed.json" ] \
   || pgrep -f 'scripts/run_experiments.py --datasets forest_cover --generators forest_diffusion' >/dev/null 2>&1; then
  echo "==== skip forest_diffusion $(date -Is) (already running or completed) ====" >> "$LOG"
  status_fd=0
else
  echo "==== waiting for ctabgan completed.json before ForestDiffusion ====" >> "$LOG"
  while [ ! -f "${DIR}/ctabgan/completed.json" ]; do
    if [ -f "${DIR}/forest_diffusion/completed.json" ] \
       || pgrep -f 'scripts/run_experiments.py --datasets forest_cover --generators forest_diffusion' >/dev/null 2>&1; then
      echo "==== skip wait $(date -Is) forest_diffusion already started elsewhere ====" >> "$LOG"
      break
    fi
    sleep 60
  done
fi

if [ ! -f "${DIR}/forest_diffusion/completed.json" ] \
   && ! pgrep -f 'scripts/run_experiments.py --datasets forest_cover --generators forest_diffusion' >/dev/null 2>&1; then
  echo "==== ctabgan done $(date -Is); starting forest_diffusion ====" >> "$LOG"
  cp config/generators/forest_diffusion_fast.yaml config/generators/forest_diffusion.yaml
  set +e
  "$PYTHON" -u scripts/run_experiments.py \
    --datasets forest_cover \
    --generators forest_diffusion \
    --n-trials 10 \
    --resume \
    >> "$LOG" 2>&1
  status_fd=$?
  set -e
  echo "==== finished forest_diffusion $(date -Is) exit=${status_fd} ====" >> "$LOG"
else
  status_fd=0
  echo "==== forest_diffusion handled on another GPU $(date -Is) ====" >> "$LOG"
fi
if [ -f config/generators/forest_diffusion_full.yaml ]; then
  cp config/generators/forest_diffusion_full.yaml config/generators/forest_diffusion.yaml
fi

if [ -f "${DIR}/ctabgan/completed.json" ] \
   && [ -f "${DIR}/tabddpm/completed.json" ] \
   && [ -f "${DIR}/forest_diffusion/completed.json" ]; then
  echo "==== writing 8-generator HPO reports $(date -Is) ====" >> "$LOG"
  "$PYTHON" -u "${DIR}/write_forest_cover_reports.py" >> "$LOG" 2>&1 || \
    echo "==== report write failed $(date -Is) ====" >> "$LOG"
fi

exit $(( status_tab != 0 ? status_tab : status_fd ))
