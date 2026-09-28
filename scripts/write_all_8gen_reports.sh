#!/usr/bin/env bash
# Rebuild every dataset HPO workbook in the Adult Excel structure.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"
cd "$WORKDIR"
exec "$PYTHON" -u scripts/write_hpo_workbook.py --all
