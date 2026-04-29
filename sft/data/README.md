# SFT Data Preparation

This directory contains utilities for preparing CRAFT SFT datasets from generated trajectory files. The generated `.jsonl` datasets are intentionally not included in this repository because they can be large and may contain experiment-specific outputs.

The expected training record format is OpenAI-style chat JSONL:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "tools": "..."}
```

Use `prepare_experiments.py` after generating trajectories. Paths can be configured through environment variables:

- `CRAFT_ROOT`: repository root.
- `CRAFT_SFT_OUTPUT_DIR`: output directory for prepared datasets.
- `CRAFT_MAIN_REAL_DATA`: main iterative-learning JSONL file.
- `CRAFT_MAIN_SYNTHETIC_DATA`: main synthetic JSONL file.
- `CRAFT_ABLATION_DIR`: directory containing ablation outputs.
- `CRAFT_CONTRAST_DIR`: directory containing contrast outputs.
