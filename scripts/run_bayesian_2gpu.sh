#!/usr/bin/env bash
# One worker per GPU (resume-safe). Defaults avoid GPUs 3/5 if busy elsewhere.
# Cancer MT → GPU_MT_CANCER, Cancer GP → GPU_GP_CANCER,
# Alzheimer's MT → GPU_MT_ALZ, Alzheimer's GP → GPU_GP_ALZ.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${SYNTH_PYTHON:-/home/gopi_b/SYNTH_BENCHMARK/.venv/bin/python}"
export SYNTH_REPO_ROOT="${SYNTH_REPO_ROOT:-/home/gopi_b/SYNTH_BENCHMARK}"

# Prefer idle cards 0,1,2,4 (3 and 5 often occupied on this host).
GPU_MT_CANCER="${GPU_MT_CANCER:-${GPU_CANCER:-0}}"
GPU_MT_ALZ="${GPU_MT_ALZ:-${GPU_ALZ:-1}}"
GPU_GP_CANCER="${GPU_GP_CANCER:-2}"
GPU_GP_ALZ="${GPU_GP_ALZ:-4}"
GENS=(gaussian_copula copulagan ctgan ctabgan tvae wgan_gp forest_diffusion tabddpm)

launch() {
  local sampler="$1" gpu="$2" dataset="$3"
  local logdir="$ROOT/${sampler}/logs"
  mkdir -p "$logdir"
  local log="$logdir/${dataset}_gpu${gpu}.out"
  local pidf="$logdir/${dataset}_gpu${gpu}.pid"
  # Clear stale pid files for this dataset under this sampler.
  rm -f "$logdir"/${dataset}_gpu*.pid
  echo "launch $sampler dataset=$dataset GPU=$gpu CUDA_VISIBLE_DEVICES=$gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u "$ROOT/${sampler}/run.py" \
    --datasets "$dataset" --generators "${GENS[@]}" --quiet-trials \
    >"$log" 2>&1 &
  echo $! >"$pidf"
}

launch multivariate_tpe "$GPU_MT_CANCER" cancer
launch multivariate_tpe "$GPU_MT_ALZ" alzheimers
sleep 2
launch gaussian_process "$GPU_GP_CANCER" cancer
launch gaussian_process "$GPU_GP_ALZ" alzheimers

echo "PIDs:"
for f in multivariate_tpe/logs/*_gpu*.pid gaussian_process/logs/*_gpu*.pid; do
  [ -f "$f" ] || continue
  echo "  $f -> $(cat "$f") $(ps -p "$(cat "$f")" -o etime= 2>/dev/null || echo dead)"
done
echo "GPU map: MT cancer=$GPU_MT_CANCER MT alz=$GPU_MT_ALZ GP cancer=$GPU_GP_CANCER GP alz=$GPU_GP_ALZ"
