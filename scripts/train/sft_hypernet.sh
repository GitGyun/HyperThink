#!/usr/bin/env bash
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

unset CUDA_LAUNCH_BLOCKING
unset NCCL_P2P_DISABLE
unset NCCL_SHM_DISABLE
unset NCCL_IB_DISABLE

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1   # optional, often helps overlap
export ACCELERATE_LOG_LEVEL=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR   # silence c10d "[W ... socket.cpp] ... IPv6 ... errno 97" rendezvous warning

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Number of GPUs: ${N_GPUS}"

BASE_PORT=29500
GPU_ID=$(echo $CUDA_VISIBLE_DEVICES | awk -F, '{print $1}')
PORT=$((BASE_PORT + GPU_ID))
echo "Using port: ${PORT}"

# accelerate refuses --multi_gpu with a single process, so only add it when >1 GPU is visible.
MULTI_GPU=()
if [[ "${N_GPUS}" -gt 1 ]]; then MULTI_GPU=(--multi_gpu); fi
exec accelerate launch "${MULTI_GPU[@]}" --main_process_port "${PORT}"   --num_processes "${N_GPUS}" "${REPO_ROOT}/train/sft_hypernet.py" --bf16 --tf32 "$@"
