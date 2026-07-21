<h1 align="center">Heuristic Reminders</h1>

<h3 align="center">Guiding Vision-Language Models Towards Safe and Faithful Reasoning</h3>

<p align="center">
  Accepted at the
  <a href="https://secure-and-trustworthy-llm.github.io/">2nd Workshop on Secure and Trustworthy Large Language Models (SeT-LLM 2026)</a>, KDD 2026
</p>

This repository contains the minimal release for generating heuristic-reminder preference data and training VLMs with DPO.

| Component | Purpose | Entry point |
| --- | --- | --- |
| Safety pipeline | Detect unsafe reasoning and inject safety reminders | `pipelines/1_safety_pipeline/` |
| Benign pipeline | Detect visual-attention decay and inject image reminders | `pipelines/2_benign_pipeline/` |
| Data export | Build multimodal DPO JSONL | `pipelines/3_organize/merge_dpo.py` |
| Training | Run LoRA DPO with ms-swift | `training/train_dpo.sh` |

## 1. Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/andylinx/Heuristic_Reminder.git
cd Heuristic_Reminder

pip install -r requirements.txt
```

Install the additional training dependency on the training machine:

```bash
pip install -r requirements-train.txt
```

The data pipeline requires:

- a reasoning-capable VLM served through vLLM;
- a local Llama Guard 4 checkpoint for safety scoring;
- a local Qwen2.5-VL-compatible checkpoint for image-attention extraction.

## 2. Input Data

Safety data should be a JSON list:

```json
[
  {
    "question": "User question",
    "images": ["relative/path/to/image.png"],
    "source": "dataset_name",
    "category": "Violence",
    "sub_category": "Weapon-Related Violence"
  }
]
```

Benign data should additionally include the ground-truth answer:

```json
[
  {
    "qid": "sample-1",
    "question": "What number is shown in the image?",
    "answer": "3",
    "images": ["relative/path/to/image.png"],
    "source": "dataset_name"
  }
]
```

## 3. Start the vLLM Server

```bash
mkdir -p outputs

vllm serve /path/to/reasoning-vlm \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --allowed-local-media-path /path/to/images \
  > outputs/vllm.log 2>&1 &

VLLM_PID=$!
```

The pipeline defaults to `http://localhost:8000` and automatically resolves the first model exposed by the server.

## 4. Safety Preference Pipeline

```bash
export LLAMA_GUARD_MODEL_PATH=/path/to/Llama-Guard-4-12B
DATA_ROOT=/path/to/images
VLLM_URL=http://localhost:8000

# 1. Generate initial reasoning responses.
python pipelines/1_safety_pipeline/step1_get_responses.py \
  --input data/safety.json \
  --output outputs/safety_step1_responses.json \
  --data-base-path "$DATA_ROOT" \
  --vllm-url "$VLLM_URL" \
  --max-tokens 10240 \
  --num-threads 4

# 2. Retain responses classified as unsafe.
python pipelines/1_safety_pipeline/step2_filter_unsafe.py \
  --input outputs/safety_step1_responses.json \
  --output outputs/safety_step2_unsafe.json \
  --guard-model-path "$LLAMA_GUARD_MODEL_PATH"

# 3. Compute step-level unsafe probabilities.
python pipelines/1_safety_pipeline/step3_get_safety_probs.py \
  --input outputs/safety_step2_unsafe.json \
  --output outputs/safety_step3_probs.json \
  --guard-model-path "$LLAMA_GUARD_MODEL_PATH"

# 4. Inject safety reminders and continue generation.
python pipelines/1_safety_pipeline/step4_inject_safety.py \
  --input outputs/safety_step3_probs.json \
  --output outputs/safety_step4_injected.json \
  --data-base-path "$DATA_ROOT" \
  --vllm-url "$VLLM_URL" \
  --threshold 0.6

# 5. Build safety preference pairs.
python pipelines/1_safety_pipeline/step5_curate_preferences.py \
  --input outputs/safety_step4_injected.json \
  --output outputs/safety_preferences.json \
  --strategy first_after_threshold \
  --guard-model-path "$LLAMA_GUARD_MODEL_PATH"
```

The output contains the original unsafe response as `rejected` and the safer reminded response as `accepted`.

## 5. Benign Preference Pipeline

```bash
DATA_ROOT=/path/to/images
VLLM_URL=http://localhost:8000

# 1. Generate multiple responses for each question.
python pipelines/2_benign_pipeline/step1_get_responses.py \
  --input data/benign.json \
  --output outputs/benign_step1_responses.json \
  --data-base-path "$DATA_ROOT" \
  --vllm-url "$VLLM_URL" \
  --num-samples 3 \
  --max-tokens 8192

# 2. Remove questions for which every response is correct.
python pipelines/2_benign_pipeline/step2_filter_simple.py \
  --input outputs/benign_step1_responses.json \
  --output outputs/benign_step2_filtered.json \
  --vllm-url "$VLLM_URL"

# 3. Extract step-level image attention.
python attn_calc/calc_step_only.py \
  --model-path /path/to/qwen2.5-vl \
  --question-file outputs/benign_step2_filtered.json \
  --output-file outputs/benign_step3_attention.json \
  --device cuda:0

# 4. Inject image-recall reminders and continue generation.
python pipelines/2_benign_pipeline/step4_inject_benign.py \
  --input outputs/benign_step3_attention.json \
  --output outputs/benign_step4_injected.json \
  --data-base-path "$DATA_ROOT" \
  --vllm-url "$VLLM_URL" \
  --threshold 0.01

# 5. Build benign preference pairs.
python pipelines/2_benign_pipeline/step5_curate_preferences.py \
  --injected outputs/benign_step4_injected.json \
  --original outputs/benign_step2_filtered.json \
  --output outputs/benign_preferences.json
```

Optional post-processing can rewrite repetitive reasoning and refine the final answer:

```bash
python pipelines/2_benign_pipeline/step6_reasoning.py \
  --input outputs/benign_preferences.json \
  --output outputs/benign_preferences_rewritten.json \
  --vllm-url "$VLLM_URL" \
  --field accepted

python pipelines/2_benign_pipeline/step7_answer.py \
  --input outputs/benign_preferences_rewritten.json \
  --output outputs/benign_preferences_final.json \
  --vllm-url "$VLLM_URL" \
  --field accepted
```

If post-processing is skipped, use `outputs/benign_preferences.json` in the next step.

## 6. Build the DPO Dataset

```bash
python pipelines/3_organize/merge_dpo.py \
  --safety outputs/safety_preferences.json \
  --benign outputs/benign_preferences_final.json \
  --output outputs/heuristic_reminders_dpo.jsonl \
  --data-base-path /path/to/images
```

Each JSONL record follows the multimodal ms-swift DPO format:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nUser question"},
    {"role": "assistant", "content": "Preferred response"}
  ],
  "images": ["/absolute/path/to/image.png"],
  "rejected_response": "Rejected response"
}
```

The exporter validates response fields, image paths, and the number of `<image>` placeholders before writing the dataset.

## 7. Train with DPO

```bash
CUDA_VISIBLE_DEVICES=0 \
  bash training/train_dpo.sh \
  /path/to/qwen-vl-model \
  outputs/heuristic_reminders_dpo.jsonl
```

For multi-GPU training:

```bash
OUTPUT_DIR=outputs/dpo_run \
LEARNING_RATE=5e-6 \
GRADIENT_ACCUMULATION_STEPS=32 \
MAX_LENGTH=32768 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
  bash training/train_dpo.sh \
  /path/to/qwen-vl-model \
  outputs/heuristic_reminders_dpo.jsonl \
  --deepspeed zero2
```

Set `DRY_RUN=1` to inspect the assembled ms-swift command without starting a GPU job.

## 8. Smoke Test

Run the complete mocked pipeline without a model server or checkpoint:

```bash
python scripts/smoke_test_pipeline.py
```

## 9. Stop the vLLM Server

Always stop the server after data generation and verify that its GPU memory has been released:

```bash
kill "$VLLM_PID"
wait "$VLLM_PID" 2>/dev/null || true

ps -p "$VLLM_PID"
nvidia-smi
```

Do not reuse a vLLM service or GPU owned by another user. Record the server startup, shutdown, exit reason, and GPU cleanup status in the experiment log.
