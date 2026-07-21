# DPO training

The release training entry point uses LoRA DPO through `ms-swift`.

```bash
pip install -r requirements-train.txt

CUDA_VISIBLE_DEVICES=0 \
  bash training/train_dpo.sh \
  /path/to/qwen-vl-model \
  outputs/release_dpo.jsonl
```

The script keeps the vision tower and multimodal aligner frozen and trains LoRA
adapters on the language model. Its defaults are intended as portable release
defaults, not as a claim about the exact settings used for every paper result.

Common overrides are environment variables:

```bash
OUTPUT_DIR=outputs/my_run \
LEARNING_RATE=5e-6 \
GRADIENT_ACCUMULATION_STEPS=32 \
MAX_LENGTH=32768 \
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
  bash training/train_dpo.sh MODEL DATA --deepspeed zero2
```

Set `DRY_RUN=1` to print the final command without starting a GPU job.
