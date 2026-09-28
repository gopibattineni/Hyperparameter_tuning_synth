#!/usr/bin/env bash
# Fill every missing pair of the 15×7 sweep across healthy GPUs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$WORKDIR"

if [ -f config/generators/tabddpm_fast.yaml ]; then
  cp config/generators/tabddpm_fast.yaml config/generators/tabddpm.yaml
fi

export GENERATORS="${GENERATORS:-tabddpm}"
export N_TRIALS="${N_TRIALS:-10}"
export GPUS="${GPUS:-0,1,2,3,5}"
exec "$SCRIPT_DIR/run_6gpu_parallel.sh" start
