#!/usr/bin/env bash
# Rebuild every dataset HPO workbook in the Adult Excel structure.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKDIR="$(cd "${EXP}/.." && pwd)"
PYTHON="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"
cd "$WORKDIR"
exec "$PYTHON" -u "${SCRIPT_DIR}/write_hpo_workbook.py" --all
