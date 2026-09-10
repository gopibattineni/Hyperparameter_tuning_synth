#!/usr/bin/env bash
# Keep forest_cover × ctabgan running for up to 18h; restart with --resume if the worker dies.
set -u

WORKDIR="/home/gopi.battineni/Hyperparameter_tuning_synth"
RESULTS="${WORKDIR}/results/classification/4_forest_cover"
LOG="${RESULTS}/ctabgan_watchdog.log"
PIDFILE="${RESULTS}/ctabgan.pid"
WATCH_PIDFILE="${RESULTS}/ctabgan_watchdog.pid"
DONE="${RESULTS}/ctabgan/completed.json"
HOURS="${1:-18}"
CHECK_SEC="${2:-120}"

cd "$WORKDIR" || exit 1
echo "$$" > "$WATCH_PIDFILE"
DEADLINE=$(($(date +%s) + HOURS * 3600))

worker_running() {
  pgrep -f 'python3 -u scripts/run_experiments.py --datasets forest_cover --generators ctabgan' >/dev/null 2>&1
}

start_worker() {
  cp config/generators/ctabgan_fast.yaml config/generators/ctabgan.yaml
  nohup python3 -u scripts/run_experiments.py \
    --datasets forest_cover \
    --generators ctabgan \
    --n-trials 10 \
    --resume \
    >> "${RESULTS}/ctabgan_fast_run.log" 2>&1 &
  echo "$!" > "$PIDFILE"
  disown
  echo "$(date -Is) started worker pid=$(cat "$PIDFILE")" >> "$LOG"
}

echo "==== watchdog start $(date -Is) deadline=${HOURS}h check=${CHECK_SEC}s ====" >> "$LOG"

while [ ! -f "$DONE" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
  if worker_running; then
    echo "$(date -Is) worker alive pid=$(cat "$PIDFILE" 2>/dev/null || echo '?')" >> "$LOG"
  else
    echo "$(date -Is) worker missing — restarting with --resume" >> "$LOG"
    echo "==== watchdog restart $(date -Is) ====" >> "${RESULTS}/ctabgan_fast_run.log"
    start_worker
  fi
  sleep "$CHECK_SEC"
done

if [ -f "$DONE" ]; then
  cp config/generators/ctabgan_full.yaml config/generators/ctabgan.yaml
  echo "$(date -Is) DONE — ctabgan completed.json found" >> "$LOG"
else
  echo "$(date -Is) STOP — ${HOURS}h deadline reached (not completed)" >> "$LOG"
fi
