#!/usr/bin/env bash
# Fill every missing pair of the 15×8 sweep across healthy GPUs.
# Already-running pairs (e.g. forest_cover CTAB-GAN+ / ForestDiffusion) are skipped.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$WORKDIR"

# Fast diffusion profiles — 10 trials, same budget as forest_cover leftovers.
if [ -f config/generators/tabddpm_fast.yaml ]; then
  cp config/generators/tabddpm_fast.yaml config/generators/tabddpm.yaml
fi
if [ -f config/generators/forest_diffusion_fast.yaml ]; then
  cp config/generators/forest_diffusion_fast.yaml config/generators/forest_diffusion.yaml
fi

export GENERATORS="${GENERATORS:-tabddpm forest_diffusion}"
export N_TRIALS="${N_TRIALS:-10}"
export GPUS="${GPUS:-0,1,2,3,4,5}"
exec "$SCRIPT_DIR/run_6gpu_parallel.sh" start
