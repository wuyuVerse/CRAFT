# CRAFT

CRAFT is a data-generation and evaluation toolkit for improving tool-use agents on tau2-bench-style multi-turn tasks. It includes:

- multi-agent synthesis of tau2-format tasks,
- iterative recovery-data generation with error injection,
- trajectory formatting and SFT data preparation utilities,
- tau2-bench evaluation wrappers with synthetic test-set support.

This repository is sanitized for public release. It does not contain private checkpoints, training logs, cluster scripts, API keys, or full generated SFT datasets.

## Repository Layout

```text
CRAFT/
  src/synthetic_gen/          # synthetic tau2 task generation
  src/iterative_learning/     # iterative recovery-data generation
  src/eval/                   # tau2-bench evaluation wrapper and metrics
  scripts/generate/           # runnable generation scripts
  scripts/eval/               # runnable evaluation scripts
  configs/                    # public example configs
  sft/data/                   # SFT dataset preparation utilities
  docs/                       # format notes
```

## Installation

```bash
cd CRAFT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/src:$PYTHONPATH"
```

CRAFT expects `tau2-bench` to be installed or available locally. One simple setup is:

```bash
git clone https://github.com/sierra-research/tau2-bench.git
PIP_CONSTRAINT= pip install -e tau2-bench
export TAU2_DATA_DIR="$PWD/tau2-bench/data"
export TAU2_BENCH_DIR="$PWD/tau2-bench"
```

`PIP_CONSTRAINT=` is only needed on machines that set a global pip constraint file; it is harmless otherwise.

Use your own OpenAI-compatible model endpoint:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="your-agent-model"
export OPENAI_USER_MODEL="your-user-simulator-model"
```

## Generate Synthetic Tasks

```bash
bash scripts/generate/run_synthetic_gen_v2.sh configs/generate/synthetic_tasks.example.yaml airline 10
```

The output is written to `output/synthetic_tasks/<domain>/` and includes tau2-format task JSON files.

## Generate Recovery Data

```bash
bash scripts/generate/run_iterative_learning.sh configs/generate/iterative_learning.example.yaml
```

This runs tau2 tasks, analyzes failures, optionally injects recoverable errors, and writes SFT-ready trajectories under `output/iterative_learning/`.

## Merge SFT Data

```bash
bash scripts/generate/run_merge.sh configs/generate/merge.example.yaml
```

For paper-scale experiments, `sft/data/prepare_experiments.py` can assemble main, scaling, ablation, and contrast datasets from generated JSONL files. The released repo intentionally excludes generated `.jsonl` data.

## Run CRAFT-Eval

For a one-task smoke test:

```bash
bash scripts/eval/eval.sh configs/eval/craft_eval.yaml --domains airline --num-tasks 1 --max-concurrency 1
```

For the full CRAFT-Eval suite:

```bash
bash scripts/eval/eval.sh configs/eval/craft_eval.yaml
```

`configs/eval/craft_eval.yaml` runs the included 1,200-task CRAFT-Eval synthetic test set (400 tasks per domain). It does not run native tau2-bench tasks unless you set `also_test_native: true` in the YAML. Results are written to `eval_results/`.

## What Is Not Included

- model checkpoints and tokenizer files,
- large generated SFT JSONL/Arrow datasets,
- private endpoint configs and cluster launch scripts,
- local caches, logs, and paper build artifacts.

All credentials should be passed through environment variables or local config files ignored by git.

## License

The CRAFT code in this repository is released under the MIT License. tau2-bench is a separate dependency; follow its license when using its code or data.
