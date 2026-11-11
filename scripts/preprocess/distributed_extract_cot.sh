#!/usr/bin/env bash
# Run one extraction shard. Pass --distill for training data.
# MODEL, NUM_SHARDS (8), GPU_OFFSET (0), MAX_NEW_TOKENS (4096),
# OUTPUT_DIR_BASE (outputs/results), TRAIN_OUTPUT_DIR (outputs/distill_data)
# can be overridden via environment. One seed by default.
# Example: CUDA_VISIBLE_DEVICES=0 NUM_SHARDS=2 bash "$0" gsm8k --distill
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
NUM_SHARDS="${NUM_SHARDS:-8}"
GPU_OFFSET="${GPU_OFFSET:-0}"

# Split args: positional task names until the first --flag; everything from
# that flag onward is forwarded verbatim to extract_cot_sharded.py.
TASKS_ARR=()
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == --* ]]; then
    EXTRA_ARGS+=("$@")
    break
  fi
  TASKS_ARR+=("$1")
  shift
done

if [[ ${#TASKS_ARR[@]} -eq 0 ]]; then
  if [[ -n "${TASKS:-}" ]]; then
    read -r -a TASKS_ARR <<< "$TASKS"
  else
    if [[ " ${EXTRA_ARGS[*]} " == *" --distill "* ]]; then
      TASKS_ARR=("gsm8k" "deepscaler")
    else
      TASKS_ARR=("gsm8k-test" "math500" "aime_24_25")
    fi
  fi
fi

# Derive model_tag for the output directory
case "$MODEL" in
  *Qwen3-0.6B*)      MODEL_TAG=qwen0.6b ;;
  *SmolLM3-3B*)      MODEL_TAG=smollm3b ;;
  *Olmo-3-7B-Think*) MODEL_TAG=olmo3-7b-think ;;
  *)                 MODEL_TAG="${MODEL##*/}" ;;
esac

OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-outputs/results}"

OUTPUT_DIR="${OUTPUT_DIR_BASE}"
TRAIN_OUTPUT_DIR="${TRAIN_OUTPUT_DIR:-outputs/distill_data}"

GPU_ID=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print $1}')
SHARD_ID=$((GPU_ID - GPU_OFFSET))

PY_SCRIPT=./preprocess/extract_cot_sharded.py

# Seed loop: one default seed. If --seed is among the extra args,
# run once and skip the loop.
HAS_SEED_FLAG=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--seed" ]]; then HAS_SEED_FLAG=true; break; fi
done

if [[ "$HAS_SEED_FLAG" == "true" ]]; then
  SEEDS_ARR=("")
else
  read -r -a SEEDS_ARR <<< "${SEEDS:-''}"
fi

echo "[extract] MODEL=${MODEL} (tag=${MODEL_TAG})"
echo "[extract] TASKS=${TASKS_ARR[*]}"
echo "[extract] SEEDS=${SEEDS_ARR[*]}"
echo "[extract] MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "[extract] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[extract] TRAIN_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}"
echo "[extract] GPU_ID=${GPU_ID}"
echo "[extract] NUM_SHARDS=${NUM_SHARDS}, SHARD_ID=${SHARD_ID}"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && echo "[extract] EXTRA_ARGS=${EXTRA_ARGS[*]}"

for task in "${TASKS_ARR[@]}"; do
  for seed in "${SEEDS_ARR[@]}"; do
    seed_flag=""
    if [[ -n "$seed" && "$seed" != "''" ]]; then
      seed_flag="--seed ${seed}"
    fi
    cmd="python ${PY_SCRIPT} \
         --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID} \
         --max_new_tokens ${MAX_NEW_TOKENS} \
         --model ${MODEL} \
         --task ${task} \
         --output_dir ${OUTPUT_DIR} \
         --train_output_dir ${TRAIN_OUTPUT_DIR} \
         ${seed_flag} \
         ${EXTRA_ARGS[*]}"
    echo "${cmd}"
    eval "${cmd}"
  done
done
