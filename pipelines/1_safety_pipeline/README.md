# Safety Pipeline

This directory contains the release version of the safety data-construction pipeline.

For the full setup, input format, and end-to-end commands, see the repository root `README.md`.

Quick entry points:

```bash
python pipelines/1_safety_pipeline/step1_get_responses.py --help
python pipelines/1_safety_pipeline/step2_filter_unsafe.py --help
python pipelines/1_safety_pipeline/step3_get_safety_probs.py --help
python pipelines/1_safety_pipeline/step4_inject_safety.py --help
python pipelines/1_safety_pipeline/step5_curate_preferences.py --help
```

Workflow summary:

```text
input
  -> step1 responses
  -> step2 unsafe-only subset
  -> step3 segment probabilities
  -> step4 injected continuations
  -> step5 safety preference pairs
```
