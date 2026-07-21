#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 MODEL_PATH_OR_ID DATASET_JSONL [additional ms-swift arguments...]" >&2
  echo "Example: CUDA_VISIBLE_DEVICES=0 $0 /path/to/model outputs/release_dpo.jsonl" >&2
  exit 2
fi

model=$1
dataset=$2
shift 2

if [[ ! -f "$dataset" ]]; then
  echo "DPO dataset not found: $dataset" >&2
  exit 2
fi

output_dir=${OUTPUT_DIR:-outputs/dpo_lora}

cmd=(
  swift rlhf
  --rlhf_type dpo
  --model "$model"
  --dataset "$dataset"
  --tuner_type lora
  --torch_dtype bfloat16
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-16}"
  --learning_rate "${LEARNING_RATE:-1e-5}"
  --lora_rank "${LORA_RANK:-8}"
  --lora_alpha "${LORA_ALPHA:-32}"
  --target_modules all-linear
  --freeze_vit true
  --freeze_aligner true
  --gradient_checkpointing true
  --max_length "${MAX_LENGTH:-16384}"
  --beta "${DPO_BETA:-0.1}"
  --warmup_ratio "${WARMUP_RATIO:-0.05}"
  --logging_steps "${LOGGING_STEPS:-5}"
  --save_steps "${SAVE_STEPS:-100}"
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}"
  --output_dir "$output_dir"
  --report_to none
  "$@"
)

if [[ ${DRY_RUN:-0} == 1 ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
  exit 0
fi

if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift is not installed. Run: pip install -r requirements-train.txt" >&2
  exit 127
fi

echo "Starting DPO training; output: $output_dir"
exec "${cmd[@]}"
