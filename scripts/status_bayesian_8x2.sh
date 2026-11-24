#!/usr/bin/env bash
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "=== processes (2-GPU layout) ==="
for f in multivariate_tpe/logs/*_gpu*.pid gaussian_process/logs/*_gpu*.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  name=$(basename "$f" .pid)
  if ps -p "$pid" >/dev/null 2>&1; then
    ps -p "$pid" -o pid=,etime=,cmd= | awk -v n="$name" '{print n": "$0}'
  else
    echo "$name: not running (pid $pid)"
  fi
done
# fallback if old single-pid layout
for f in multivariate_tpe/logs/full_run.pid gaussian_process/logs/full_run.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  ps -p "$pid" >/dev/null 2>&1 && ps -p "$pid" -o pid,etime,cmd
done
echo
echo "=== completed (want 16 per sampler = 8 gens × 2 datasets) ==="
echo -n "multivariate_tpe: "; find multivariate_tpe/results -name completed.json 2>/dev/null | wc -l
find multivariate_tpe/results -name completed.json 2>/dev/null | sed 's|.*/classification/||;s|/completed.json||' | sort
echo -n "gaussian_process: "; find gaussian_process/results -name completed.json 2>/dev/null | wc -l
find gaussian_process/results -name completed.json 2>/dev/null | sed 's|.*/classification/||;s|/completed.json||' | sort
echo
echo "=== failed ==="
find multivariate_tpe/results gaussian_process/results -name failed.json 2>/dev/null | sort || true
