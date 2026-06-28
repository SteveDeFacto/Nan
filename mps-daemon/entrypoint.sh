#!/bin/sh
# NAN MPS control daemon. Runs in its own GPU container so per-tenant worker
# processes (in the worker container, sharing /tmp/nvidia-mps + ipc:host) get
# hardware-enforced SM% (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE) and VRAM
# (CUDA_MPS_PINNED_DEVICE_MEM_LIMIT) caps — both validated enforced under CC.
set -e
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "[mps] starting control daemon (pipe=$CUDA_MPS_PIPE_DIRECTORY)"
nvidia-cuda-mps-control -d || { echo "[mps] FAILED to start daemon"; exit 1; }

# Keep the container alive and the daemon healthy; restart it if it dies.
while true; do
  if ! echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "[mps] daemon not responding — restarting"
    nvidia-cuda-mps-control -d || true
  fi
  sleep 10
done
