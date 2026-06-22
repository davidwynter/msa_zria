# msa_zria

`msa_zria` is now scoped as a Gemma-first implementation scaffold for combining:

- MSA-style LLM parsing and model synthesis
- ZRIA symbolic reasoning
- Pyro-based probabilistic execution

The previous repository state was a single research memo written around `LLaMA-2-14B`. There was no runnable code, no environment definition, no config model, no dataset builder, and no Gemma-compatible training path. This update changes the target model to **Gemma 4 12B Unified** and adds the minimum Python structure needed to start implementing the pipeline.

## Project Analysis

Before this update, the repository had only:

- `README.md`
- `.gitignore`

That meant the project was still at the concept stage. In practice:

- The LLM dependency was outdated and hardcoded to LLaMA-2 14B.
- The training guidance used `AutoModelForCausalLM` and `AutoTokenizer`, which is the wrong default for Gemma 4.
- There was no typed configuration to keep experiments reproducible.
- There was no dataset formatting code for triples, natural language, or hybrid inputs.
- There was no code path for loading a model suitable for QLoRA fine-tuning.

## Migration Target

This repository now targets:

- Base model: `google/gemma-4-12B`
- Processor template source: `google/gemma-4-12B-it`
- Training style: QLoRA with Hugging Face `transformers`, `peft`, and `trl`

Why this model:

- It is the closest size match to the old single-model plan, but materially more current.
- It is positioned for reasoning, coding, and agentic workflows.
- It supports long context and modern chat/system-role formatting.
- It keeps the repo aligned with Google’s current Gemma 4 documentation instead of legacy Gemma 1/2/3 or LLaMA-specific assumptions.

## What Was Added

This update adds a minimal implementation scaffold:

- `pyproject.toml`
- `configs/gemma4_12b.yaml`
- `src/msa_zria/config.py`
- `src/msa_zria/data.py`
- `src/msa_zria/training.py`
- `src/msa_zria/__init__.py`

The scaffold intentionally does not pretend the full MSA/ZRIA/Pyro system already exists. It only covers the parts needed to move from planning into implementation:

1. Typed experiment configuration
2. Triple/NL/hybrid training example formatting
3. Gemma 4 12B model, processor, quantization, and LoRA setup

## Repository Layout

```text
msa_zria/
├── README.md
├── configs/
│   └── gemma4_12b.yaml
├── pyproject.toml
└── src/
    └── msa_zria/
        ├── __init__.py
        ├── config.py
        ├── data.py
        └── training.py
```

## Design Choices

### 1. Use `AutoModelForMultimodalLM`

Gemma 4 12B is not treated here like an older text-only decoder. Even if your first workload is text-only, the correct loading path is the multimodal Gemma 4 stack:

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor
```

### 2. Use `AutoProcessor`, not just `AutoTokenizer`

Gemma 4 uses an official processor and chat template. This is the right place to keep prompt formatting consistent across training and inference.

### 3. Keep the first implementation text-first

The project goal mentions customer support parsing, code synthesis, and evaluation. Those can all start as text tasks even though Gemma 4 12B also supports image and audio inputs.

### 4. Keep the scaffold small

There is still no full training loop, Pyro executor, or DSPy orchestration module in this repo. That is intentional. The current step is to replace outdated model assumptions with a clean, typed base.

## Quick Start

Create an environment and install the package:

```bash
cd msa_zria
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Load the example config and initialize the Gemma 4 12B training objects:

```python
from msa_zria.config import load_experiment_config
from msa_zria.training import (
    build_lora_config,
    build_sft_config,
    load_model_and_processor,
)

config = load_experiment_config("configs/gemma4_12b.yaml")
model, processor = load_model_and_processor(config.model)
peft_config = build_lora_config(config.training)
sft_config = build_sft_config(config.training)
```

Build a training example from triples:

```python
from msa_zria.data import Triple, TrainingExample

example = TrainingExample(
    instruction="Extract the device, issue, and recommended action.",
    triples=[
        Triple(subject="Printer_X", predicate="issue", object="PaperJam"),
        Triple(subject="Printer_X", predicate="resolution", object="ClearPaperPath"),
    ],
    response='{"device":"Printer_X","issue":"PaperJam","resolution":"ClearPaperPath"}',
)

messages = example.to_messages(input_mode="hybrid")
```

## Config Model

The example config separates model and training concerns:

```yaml
experiment_name: msa-zria-gemma4-12b
seed: 42
model:
  base_model_id: google/gemma-4-12B
  processor_id: google/gemma-4-12B-it
  load_in_4bit: true
training:
  output_dir: outputs/gemma4_12b
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

This is enough to keep runs reproducible without overengineering the project.

## Dataset Formatting

`src/msa_zria/data.py` supports three input modes:

- `triples`
- `text`
- `hybrid`

That maps directly to the research intent already described in the original README:

- structured triples only
- natural language only
- both together

Each `TrainingExample` renders to OpenAI-style chat messages:

- `system`
- `user`
- `assistant`

This keeps the dataset compatible with modern supervised fine-tuning pipelines and Gemma 4 chat templating.

## Gemma 4 12B Loading Notes

The model-loading scaffold does the following:

- selects `bfloat16` when supported, otherwise `float16`
- configures BitsAndBytes 4-bit NF4 quantization
- loads `AutoModelForMultimodalLM`
- loads the Gemma 4 instruction processor
- optionally calls `prepare_model_for_kbit_training`

The LoRA config intentionally does **not** hardcode target modules. Current Gemma 4 QLoRA guidance allows PEFT’s Gemma defaults to scope the LM layers.

## Gaps That Still Need Implementation

This repo is now correctly pointed at Gemma 4 12B, but it is still missing the larger system pieces:

- a real dataset ingestion script
- an actual `trl.SFTTrainer` training entrypoint
- DSPy modules for parse/code/evaluate
- Pyro execution harness for generated models
- ZRIA adapter interface
- ablation runner and metrics reports

Those should be added only after the dataset shape and evaluation contract are fixed.

## Recommended Next Steps

1. Add a `train.py` entrypoint that reads `ExperimentConfig` and launches `SFTTrainer`.
2. Add a JSONL dataset schema for customer-support examples.
3. Add one narrow end-to-end slice: parse -> generated Pyro sketch -> evaluation.
4. Add a small ablation harness only after the first Gemma fine-tuning path is stable.

## Notes

- The project is now updated to Gemma 4 12B assumptions.
- It is still intentionally a scaffold, not a finished reasoning system.
- The next useful milestone is a runnable training command, not more architecture prose.
