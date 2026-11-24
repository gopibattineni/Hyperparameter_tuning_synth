#!/usr/bin/env bash
# Poll multivariate_tpe + gaussian_process full runs; regenerate comparison when both exit.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"
cd "$ROOT"

MT_PID_FILE="$ROOT/multivariate_tpe/logs/full_run.pid"
GP_PID_FILE="$ROOT/gaussian_process/logs/full_run.pid"

still_running() {
  local f="$1"
  [ -f "$f" ] || return 1
  local pid
  pid="$(cat "$f")"
  ps -p "$pid" >/dev/null 2>&1
}

echo "==== watch start $(date -Is) ===="
while still_running "$MT_PID_FILE" || still_running "$GP_PID_FILE"; do
  echo "$(date +%H:%M:%S) multivariate=$(still_running "$MT_PID_FILE" && echo RUN || echo DONE) gp=$(still_running "$GP_PID_FILE" && echo RUN || echo DONE)"
  find multivariate_tpe/results gaussian_process/results -name completed.json 2>/dev/null | wc -l | awk '{print "  completed_pairs="$1}'
  # refresh partial comparison periodically
  "$PY" -u scripts/compare_bayesian_samplers.py >/tmp/bayes_compare.out 2>&1 || true
  sleep 300
done

echo "==== both finished $(date -Is); final comparison ===="
"$PY" -u scripts/compare_bayesian_samplers.py
echo "==== watch end $(date -Is) ===="
