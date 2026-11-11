#!/usr/bin/env bash
# Evaluation: run evaluation/evaluate_model.py on one GPU, or on one shard of a
# multi-GPU run.
#
# Runs evaluate_model.py on the current GPU with --num_shards N --shard_id <gpu_id> and
# forwards every extra arg verbatim. Each process writes one shard file; score them
# afterwards with evaluation/print_accuracy.py, which globs all shards (no merge step).
#
# Usage:
#   bash scripts/eval/distributed_evaluate.sh [args forwarded to evaluate_model.py]
#
# Examples:
#   bash scripts/eval/distributed_evaluate.sh \
#       --ckpt outputs/checkpoints/qwen3-0.6b-math.ckpt \
#       --task aime_24_25         # writes under --output_dir (default outputs/results)
#   for i in 0 1 2 3; do
#     CUDA_VISIBLE_DEVICES=$i NUM_SHARDS=4 bash scripts/eval/distributed_evaluate.sh --task bbh --ckpt <ckpt> &
#   done; wait
#
# Env overrides:
#   NUM_SHARDS=1         total shards, one process per GPU (default 1: one process scores the whole benchmark)
#   GPU_OFFSET=0         GPU id of shard 0 (shard id = GPU id - GPU_OFFSET)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NUM_SHARDS="${NUM_SHARDS:-1}"
GPU_OFFSET="${GPU_OFFSET:-0}"
GPU_ID=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print $1}')
SHARD_ID=$((GPU_ID - GPU_OFFSET))
if (( NUM_SHARDS == 1 )); then SHARD_ID=0; fi

echo "[eval] NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${SHARD_ID}"
echo "[eval] extra args: $*"

cmd="python -m evaluation.evaluate_model --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID} $*"
echo "${cmd}"
python -m evaluation.evaluate_model --num_shards "${NUM_SHARDS}" --shard_id "${SHARD_ID}" "$@"