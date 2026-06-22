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

- selects the accelerator explicitly: `cuda`, `xpu`, or `auto`
- uses `bfloat16` on newer CUDA GPUs when available, `float16` on V100 and Intel Arc
- configures BitsAndBytes 4-bit NF4 quantization
- loads `AutoModelForMultimodalLM`
- loads the Gemma 4 instruction processor
- optionally calls `prepare_model_for_kbit_training`

The LoRA config intentionally does **not** hardcode target modules. Current Gemma 4 QLoRA guidance allows PEFT’s Gemma defaults to scope the LM layers.

## 16GB GPU Profiles

The fine-tuning stage now has separate accelerator-aware profiles for both of your planned cards:

- [configs/gemma4_12b_a770.yaml](/home/david-wynter/yambina_dev/tools/msa_zria/configs/gemma4_12b_a770.yaml)
- [configs/gemma4_12b_v100.yaml](/home/david-wynter/yambina_dev/tools/msa_zria/configs/gemma4_12b_v100.yaml)

Practical defaults:

- Intel Arc A770 16GB: `training.accelerator: xpu`, `float16`, `adamw_torch`
- NVIDIA V100 16GB: `training.accelerator: cuda`, `float16`, `paged_adamw_8bit`

The code keeps 4-bit QLoRA enabled for both GPU paths. If `training.accelerator` is `auto`, CUDA is preferred when available, then XPU, then CPU.
For Intel GPU runs, make sure the environment exposes `torch.xpu`; upstream Intel XPU support in bitsandbytes exists, but is currently documented as less mature than the CUDA path.

## What msa_zria Can Solve

`msa_zria` is a fit for problems where a plain LLM answer is not enough, and where you need some combination of:

- structured fact extraction
- explicit reasoning over rules or probabilities
- auditable intermediate steps
- domain adaptation on proprietary data

In practice, that means `msa_zria` is most useful for systems that must turn messy inputs into structured state, reason over that state, and then return an answer or action recommendation.

### Good Problem Types

- Customer support diagnosis where product facts, known issues, and resolution policies matter
- Operational decision support where uncertainty must be modeled explicitly
- Compliance or policy checking where the system must explain why an action is allowed or blocked
- Incident triage where free-text reports must be converted into structured cases before routing
- Knowledge-grounded assistants that need both retrieval from a graph and a reasoning layer on top

### Example Use Cases

#### 1. Technical Support Triage

Input:

```text
"After last night's storm, my router keeps dropping out every 10 minutes."
```

`msa_zria` flow:

- parse the report into device, issue, trigger, and severity
- retrieve router and outage facts from the knowledge graph
- synthesize a probabilistic or rule-based model of likely causes
- produce a recommendation such as `check line stability`, `power cycle router`, or `escalate to field technician`

Why `msa_zria` fits:

- the input is unstructured
- the answer depends on domain facts
- the output benefits from explicit reasoning instead of pure text generation

#### 2. Equipment Failure Prediction

Input:

```text
"Pump A overheats when run above 80% load for more than 15 minutes."
```

`msa_zria` flow:

- extract variables and thresholds
- map them to equipment facts and prior failure modes
- generate a Pyro model for failure likelihood under different loads
- return a risk estimate and mitigation recommendation

Why `msa_zria` fits:

- uncertainty matters
- you want a model-derived answer, not just a plausible sentence

#### 3. Policy and Procedure Guidance

Input:

```text
"Can this refund be approved if the item was opened after 45 days?"
```

`msa_zria` flow:

- parse the request into policy fields
- retrieve refund rules and exception clauses
- run symbolic or probabilistic reasoning over the policy state
- return `approve`, `deny`, or `manual review`, plus explanation

Why `msa_zria` fits:

- policy reasoning is structured
- the output should be inspectable and defensible

## From Data to Production

The intended delivery pipeline for `msa_zria` is:

### 1. Gather Raw Domain Data

Collect the inputs your target system actually uses:

- support tickets
- chat transcripts
- troubleshooting playbooks
- policy documents
- tabular incident logs
- knowledge graph triples
- outcome labels such as `resolved`, `failed`, `escalated`, or `refund_approved`

At this stage, the goal is not model training yet. The goal is to capture the real domain states, rules, and outcomes.

### 2. Normalize and Structure the Facts

Convert the raw data into reusable assets:

- triples for entities, relations, conditions, and actions
- natural language summaries of those triples
- supervised examples for parse, code generation, and evaluation tasks

Typical outputs here:

- `triples.jsonl`
- `nl.jsonl`
- `hybrid.jsonl`
- evaluation sets with expected answers or outcomes

### 3. Define the Reasoning Contract

Before training, define exactly what the system must produce:

- parse schema: for example `device`, `issue`, `cause`, `severity`
- code or reasoning schema: for example a Pyro program, a rule trace, or a structured decision plan
- final output schema: for example `answer`, `confidence`, `recommended_action`, `justification`

This is the most important design step. If the schemas are vague, the training data and evaluation will drift.

### 4. Build Training Data for the Three Main Tasks

`msa_zria` is designed around three task families:

- parsing
- reasoning/code synthesis
- evaluation

For each real scenario, generate examples such as:

- parse the user report into structured fields
- generate a reasoning program or rule trace from the parsed facts
- judge whether a proposed answer solves the problem

This gives the model a role in the MSA loop instead of training it as a generic chatbot.

### 5. Fine-Tune Gemma 4 12B

Use the Gemma 4 12B QLoRA path in this repo to adapt the base model to your domain:

- choose the GPU profile for A770 or V100
- train on the parse/code/evaluate dataset
- save the adapter or merged model
- run small regression checks on held-out cases

The fine-tuned model should be better at producing the exact structured outputs your reasoning stack needs.

### 6. Add the External Reasoning Layer

Once the model can reliably parse and synthesize reasoning steps, wire it into:

- ZRIA for symbolic reasoning
- Pyro for probabilistic reasoning
- KG retrieval for domain facts

The production system should not rely on the LLM alone when the task needs exact logic or uncertainty handling.

### 7. Evaluate the Full Pipeline

Test the end-to-end system on realistic cases:

- does parsing recover the right state?
- does retrieval supply the right supporting facts?
- does the Pyro or ZRIA stage produce the right decision?
- does the final explanation match the computed result?

This is where you compare:

- Gemma-only
- ZRIA-only
- Pyro-only
- hybrid `msa_zria`

### 8. Deploy as a Modular Service

A production `msa_zria` deployment should be modular rather than monolithic:

- ingestion service for raw requests
- retrieval service for triples and domain facts
- Gemma inference service for parse/code/evaluate stages
- reasoning service for Pyro or ZRIA execution
- API layer for the final user-facing answer

That structure makes it easier to inspect failures, replace components, and version models independently.

### 9. Log, Review, and Retrain

After deployment, capture:

- user queries
- parsed outputs
- retrieved facts
- generated reasoning artifacts
- final decisions
- human overrides or corrections

Those records become the next round of supervised data, which is how the system should improve over time.

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
