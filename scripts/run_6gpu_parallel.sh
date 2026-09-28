#!/usr/bin/env bash
# Fan remaining HPO pairs across up to 6 GPUs (one worker per healthy GPU).
#
#   scripts/run_6gpu_parallel.sh start    # build queue + launch workers
#   scripts/run_6gpu_parallel.sh status
#   scripts/run_6gpu_parallel.sh stop
#
# Optional environment:
#   GPUS="0,1,2,3,4,5"          # default: probe 0-5 and skip dead cards
#   GENERATORS="tabddpm"
#   DATASETS="cancer adult forest_cover"
#   N_TRIALS=10
#   SYNTH_PYTHON=/path/to/python
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"

RUN_DIR="${WORKDIR}/results/gpu_parallel"
QUEUE="${RUN_DIR}/queue.txt"
LOCK="${RUN_DIR}/queue.lock"
MASTER_LOG="${RUN_DIR}/launcher.log"
ACTION="${1:-start}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/pids"

healthy_gpus() {
  local requested="${GPUS:-0,1,2,3,4,5}"
  local gpu
  IFS=',' read -r -a requested_arr <<< "$requested"
  for gpu in "${requested_arr[@]}"; do
    gpu="$(echo "$gpu" | tr -d '[:space:]')"
    [ -n "$gpu" ] || continue
    if nvidia-smi -i "$gpu" >/dev/null 2>&1; then
      echo "$gpu"
    else
      echo "skip GPU ${gpu} (nvidia-smi failed)" >> "$MASTER_LOG"
    fi
  done
}

pair_running() {
  local ds="$1" gen="$2"
  pgrep -f "scripts/run_experiments.py --datasets ${ds} --generators ${gen}" >/dev/null 2>&1
}

build_queue() {
  local gens="${GENERATORS:-tabddpm}"
  local dsets="${DATASETS:-}"
  local n_trials="${N_TRIALS:-10}"
  "$PYTHON" - "$WORKDIR" "$gens" "$dsets" "$n_trials" <<'PY'
import sys
from pathlib import Path

workdir = Path(sys.argv[1])
sys.path.insert(0, str(workdir))
from datasets.loader import load_dataset_config
from tuning.artifacts import is_completed

gens = sys.argv[2].split()
dsets_arg = sys.argv[3].strip()
n_trials = sys.argv[4]
all_ds = list(load_dataset_config())
datasets = dsets_arg.split() if dsets_arg else all_ds
unknown = [d for d in datasets if d not in all_ds]
if unknown:
    raise SystemExit(f"Unknown datasets: {unknown}")
results = workdir / "results"
# Generator-major order so GPU jobs (TabDDPM) fill the cards first.
for gen in gens:
    for ds in datasets:
        if not is_completed(results, ds, gen):
            print(f"{ds} {gen} {n_trials}")
PY
}

pop_job() {
  flock "$LOCK" bash -c '
    q="$1"
    if [ ! -s "$q" ]; then
      exit 1
    fi
    IFS= read -r line < "$q" || exit 1
    tail -n +2 "$q" > "${q}.tmp"
    mv "${q}.tmp" "$q"
    printf "%s\n" "$line"
  ' bash "$QUEUE"
}

worker_loop() {
  local gpu="$1"
  local pidfile="${RUN_DIR}/pids/gpu${gpu}.pid"
  local wlog="${RUN_DIR}/logs/worker_gpu${gpu}.log"
  echo "$$" > "$pidfile"
  echo "$(date -Is) worker GPU=${gpu} start" >> "$wlog"
  while true; do
    local job
    if ! job="$(pop_job)"; then
      echo "$(date -Is) worker GPU=${gpu} queue empty — exit" >> "$wlog"
      break
    fi
    local ds gen n
    ds="$(echo "$job" | awk '{print $1}')"
    gen="$(echo "$job" | awk '{print $2}')"
    n="$(echo "$job" | awk '{print $3}')"
    if pair_running "$ds" "$gen"; then
      echo "$(date -Is) skip in-progress ${ds} × ${gen}" >> "$wlog"
      continue
    fi
    echo "$(date -Is) GPU=${gpu} start ${ds} × ${gen} n=${n}" >> "$wlog"
    if ! "${SCRIPT_DIR}/run_one_pair.sh" "$gpu" "$ds" "$gen" "$n"; then
      echo "$(date -Is) GPU=${gpu} FAIL ${ds} × ${gen}" >> "$wlog"
    else
      echo "$(date -Is) GPU=${gpu} done ${ds} × ${gen}" >> "$wlog"
    fi
  done
  rm -f "$pidfile"
}

cmd_status() {
  echo "Queue file : $QUEUE"
  if [ -f "$QUEUE" ]; then
    echo "Queued     : $(wc -l < "$QUEUE") jobs"
    sed -n '1,20p' "$QUEUE"
    local left
    left="$(wc -l < "$QUEUE")"
    if [ "$left" -gt 20 ]; then
      echo "  … +$((left - 20)) more"
    fi
  else
    echo "Queued     : (no queue yet)"
  fi
  echo
  echo "Workers:"
  local gpu pidfile
  for pidfile in "${RUN_DIR}/pids"/gpu*.pid; do
    [ -f "$pidfile" ] || continue
    gpu="$(basename "$pidfile" .pid)"
    if ps -p "$(cat "$pidfile")" >/dev/null 2>&1; then
      echo "  ${gpu} pid=$(cat "$pidfile") RUNNING"
    else
      echo "  ${gpu} pid=$(cat "$pidfile") dead"
    fi
  done
  pgrep -af 'scripts/run_experiments.py --datasets' || true
}

cmd_stop() {
  echo "$(date -Is) stop requested" >> "$MASTER_LOG"
  : > "$QUEUE"
  local pidfile
  for pidfile in "${RUN_DIR}/pids"/gpu*.pid; do
    [ -f "$pidfile" ] || continue
    local pid
    pid="$(cat "$pidfile")"
    if ps -p "$pid" >/dev/null 2>&1; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  pkill -f 'scripts/run_one_pair.sh' 2>/dev/null || true
  echo "Stopped workers and cleared the queue."
  echo "In-flight python jobs (if any) were left running so a trial is not killed mid-write."
  echo "To force-kill those too: pkill -f 'scripts/run_experiments.py'"
}

cmd_start() {
  mapfile -t GPULIST < <(healthy_gpus)
  if [ "${#GPULIST[@]}" -eq 0 ]; then
    echo "No healthy GPUs found (tried ${GPUS:-0,1,2,3,4,5})." >&2
    exit 1
  fi

  local raw
  raw="$(build_queue)"
  : > "$QUEUE"
  local line
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    local ds gen
    ds="$(echo "$line" | awk '{print $1}')"
    gen="$(echo "$line" | awk '{print $2}')"
    if pair_running "$ds" "$gen"; then
      echo "skip already-running ${ds} × ${gen}" | tee -a "$MASTER_LOG"
      continue
    fi
    echo "$line" >> "$QUEUE"
  done <<< "$raw"

  local njobs
  njobs="$(wc -l < "$QUEUE" | tr -d ' ')"
  echo "$(date -Is) start GPUs=${GPULIST[*]} jobs=${njobs}" | tee -a "$MASTER_LOG"
  if [ "$njobs" -eq 0 ]; then
    echo "Nothing to run (all selected pairs completed or already running)."
    exit 0
  fi
  echo "Queue:"
  cat "$QUEUE"

  local gpu
  for gpu in "${GPULIST[@]}"; do
    nohup "$0" _worker "$gpu" >> "${RUN_DIR}/logs/worker_gpu${gpu}.log" 2>&1 &
    echo $! > "${RUN_DIR}/pids/gpu${gpu}.pid"
    echo "  launched worker GPU=${gpu} pid=$!"
  done
  echo
  echo "Status:  $0 status"
  echo "Stop:    $0 stop"
  echo "Logs:    ${RUN_DIR}/logs/"
}

case "$ACTION" in
  start) cmd_start ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  _worker)
    worker_loop "${2:?gpu id required}"
    ;;
  *)
    echo "Usage: $0 start|status|stop" >&2
    exit 2
    ;;
esac
