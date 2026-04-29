# CRAFT

CRAFT (**C**ontrastive **R**ecovery **A**nd **F**ailure **T**raining) is a synthetic-data framework for training tool-use agents to recover from execution errors. It builds supervision around an explicit tool-use failure taxonomy, then combines executable task synthesis, controlled failure injection, LLM-guided recovery generation, and contrastive analysis of naturally observed failures.

The paper studies CRAFT in stateful multi-turn tool-use domains: airline, retail, and telecom. Starting from a small seed set, CRAFT expands task coverage with grounded synthetic tasks and trains agents on a mixture of clean executions, injected error-recovery traces, and analysis-conditioned retries. The released code includes the CRAFT data pipeline and CRAFT-Eval, a 1,200-task synthetic evaluation suite with 400 tasks per domain.

This repository is sanitized for public release. It does not contain private checkpoints, training logs, cluster scripts, API keys, or full paper-scale generated SFT datasets.

## Method Overview

CRAFT follows the four-stage framework described in the paper:

1. **Seed distillation:** extract reusable task patterns from a small seed set and ground them with valid domain parameters.
2. **Structured synthesis:** use specialized agents to generate task goals, user scenarios, and executable evaluation criteria.
3. **Failure injection and recovery:** inject taxonomy-aligned tool-use errors, then generate recovery segments that identify the error, revise the plan, and continue execution.
4. **Contrastive recovery learning:** analyze natural failures from model rollouts and convert them into contrastive supervision for diagnosis and correction.

The failure taxonomy covers four decision points in tool use: tool selection, argument grounding, dependency order, and domain-policy compliance.

## Paper Results

In the paper, CRAFT trains 14B, 32B, and 40B coding backbones using synthetic trajectories only, without additional human annotation. The training mixture contains roughly 200K trajectories spanning clean executions, error-recovery pairs, and contrastive analysis samples.

Key reported findings:

- **Large tool-use gains:** CRAFT-40B improves from 21.77% to 78.66% task success on the main multi-turn tool-use benchmark.
- **Strong recovery behavior:** CRAFT-40B reaches 92.63% error recovery rate, showing that recovery traces teach models to repair failed tool calls rather than only imitate clean executions.
- **CRAFT-Eval generalization:** CRAFT-40B scores 59.25% on CRAFT-Eval, the released 1,200-task synthetic suite.
- **Ablation support:** removing iterative learning, error injection, contrastive analysis, or multi-agent synthesis consistently hurts performance.
- **Code retention:** tool-use SFT does not cause catastrophic forgetting on standard code benchmarks.

## What Is Included

- **Failure-aware task synthesis:** seed distillation, parameter grounding, multi-agent task generation, and task validation.
- **Recovery data construction:** controlled error injection, recovery generation, failure analysis, and contrastive trajectory construction.
- **CRAFT-Eval:** the released 1,200-task synthetic benchmark for evaluating out-of-distribution tool-use generalization.
- **SFT data utilities:** scripts for formatting, merging, and preparing generated trajectories for supervised fine-tuning.
- **Evaluation harness:** runnable scripts for CRAFT-Eval and compatible multi-turn tool-use environments.

## Repository Layout

```text
CRAFT/
  src/synthetic_gen/          # seed distillation and multi-agent task synthesis
  src/iterative_learning/     # error injection, recovery generation, iterative learning
  src/eval/                   # CRAFT-Eval runner, metrics, and trajectory extraction
  scripts/generate/           # runnable generation scripts
  scripts/eval/               # CRAFT-Eval scripts and analysis utilities
  configs/                    # public example configs for generation and evaluation
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

CRAFT uses an external executable environment package for multi-turn tool simulation. Install it locally:

```bash
git clone https://github.com/sierra-research/tau2-bench.git
PIP_CONSTRAINT= pip install -e tau2-bench
export TAU2_DATA_DIR="$PWD/tau2-bench/data"
export TAU2_BENCH_DIR="$PWD/tau2-bench"
```

`PIP_CONSTRAINT=` is only needed on machines that set a global pip constraint file; it is harmless otherwise.

This dependency provides the runtime environment and task execution interface. CRAFT itself is the data-construction, recovery-training, and evaluation code in this repository.

Configure an OpenAI-compatible endpoint for the agent and user simulator:

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_MODEL="your-agent-model"
export OPENAI_USER_MODEL="your-user-simulator-model"
```

## Run CRAFT-Eval

CRAFT-Eval contains 1,200 synthetic tasks across airline, retail, and telecom. It is used in the paper to test generalization beyond the original task distribution.

For a one-task smoke test:

```bash
bash scripts/eval/eval.sh configs/eval/craft_eval.yaml --domains airline --num-tasks 1 --max-concurrency 1
```

For the full CRAFT-Eval suite:

```bash
bash scripts/eval/eval.sh configs/eval/craft_eval.yaml
```

Results are written to `eval_results/`. The main score is task success rate. The evaluator also reports tool-call accuracy, parameter accuracy, conversation efficiency, and error recovery rate.

## Generate CRAFT Synthetic Tasks

The synthesis pipeline turns seed-derived task patterns into executable tasks through specialized generators for task goals, user scenarios, and evaluation criteria.

```bash
bash scripts/generate/run_synthetic_gen_v2.sh configs/generate/synthetic_tasks.example.yaml airline 10
```

The output is written to `output/synthetic_tasks/<domain>/`.

## Generate Recovery Trajectories

CRAFT trains recovery behavior by starting from executable trajectories, injecting taxonomy-aligned failures, and generating repair segments that demonstrate error recognition, correction planning, and corrected execution.

```bash
bash scripts/generate/run_iterative_learning.sh configs/generate/iterative_learning.example.yaml
```

This writes recovery-oriented trajectories under `output/iterative_learning/`.

## Prepare SFT Data

Merge clean synthetic trajectories, recovery trajectories, and contrastive records into an SFT-ready dataset:

```bash
bash scripts/generate/run_merge.sh configs/generate/merge.example.yaml
```

For paper-scale experiments, `sft/data/prepare_experiments.py` can assemble main, scaling, ablation, and contrast datasets from generated JSONL files. The released repo intentionally excludes generated `.jsonl` training data.

## Paper Components Mapped to Code

| Paper component | Code path |
| --- | --- |
| Seed distillation and parameter grounding | `src/synthetic_gen/core/` |
| Multi-agent task synthesis | `src/synthetic_gen/core/agents.py`, `src/synthetic_gen/runners/` |
| Task validation and filtering | `src/synthetic_gen/core/action_validator.py`, `src/synthetic_gen/runners/quality_filter.py` |
| Error injection | `src/iterative_learning/injection/` |
| Recovery generation and contrastive analysis | `src/iterative_learning/agents/`, `src/iterative_learning/analysis/` |
| CRAFT-Eval | `src/eval/synthetic_testset/`, `configs/eval/craft_eval.yaml` |
| SFT data preparation | `sft/data/prepare_experiments.py` |

## What Is Not Included

- model checkpoints and tokenizer files,
- large generated SFT JSONL/Arrow datasets,
- private endpoint configs and cluster launch scripts,
- local caches, logs, and paper build artifacts.

All credentials should be passed through environment variables or local config files ignored by git.

## Citation

```bibtex
@misc{craft2026,
  title        = {CRAFT: Contrastive Recovery And Failure Training for Tool-Use Agents},
  author       = {Anonymous Authors},
  year         = {2026},
  note         = {Anonymous review release}
}
```

## License

The CRAFT code in this repository is released under the MIT License. tau2-bench is a separate dependency; follow its license when using its code or data.
