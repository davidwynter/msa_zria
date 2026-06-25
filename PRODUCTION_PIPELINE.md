# Production Pipeline: Text Corpus to Gemma 4 with Zria_Graph

This document describes a typical production path for `msa_zria` using the current implementation:

- Gemma 4 fine-tuning for `parse`, `code`, and `evaluate`
- `Zria_Graph` for graph-native decision reasoning
- rules as the fallback control layer
- WWKG workspace and branch awareness across ingestion, training, and inference

## 1. Define the Production Problem

Start by fixing the operational task before collecting data. A good target for `msa_zria` has all of the following:

- unstructured user or operator text
- a need to extract structured state
- a need to reason over knowledge graph facts
- a need for a typed decision such as `resolved`, `unresolved`, `insufficient_information`, or `escalate`

Typical examples:

- customer support triage
- operational fault diagnosis
- policy or compliance decision support
- incident routing with graph-backed context

The key design choice is the output contract. In this repository the production system is built around three typed outputs:

- `ParseTarget`
- `CodeTarget`
- `EvaluationTarget`

Those contracts are implemented in [data.py](src/msa_zria/data.py).

## 2. Gather the Raw Text Corpus

Collect the real text that the production system will see, for example:

- support tickets
- chat transcripts
- email cases
- incident summaries
- operator notes
- policy text or procedural guidance

At the same time, collect the structured sources that support reasoning:

- WWKG triples
- known device, issue, cause, and policy entities
- decision outcomes from human operators
- rule conditions that must always hold

For each source case, you want enough information to define:

- the original message
- the expected parsed state
- the expected reasoning program shape
- the expected final evaluation or decision

## 3. Convert Raw Cases into Canonical Source Examples

The current ingestion path expects source rows shaped like [CustomerSupportCase](src/msa_zria/ingest.py).

Each source case should contain:

- `customer_message`
- `candidate_answer`
- `triples`
- optional `context`
- optional `kg_scope`
- `parse_target`
- `code_target`
- `evaluation_target`

In practice, this means your first operational dataset is not the final Gemma dataset yet. It is a curated case file that ties text, graph facts, and expected outcomes together.

The repository example is:

- [customer_support_cases.jsonl](examples/customer_support_cases.jsonl)

## 4. Attach WWKG Scope Early

If the system is branch or workspace sensitive, attach WWKG context during ingestion, not later.

Useful fields are:

- `workspace`
- `branch`
- `commit`
- `as_of`

This is important because the same user query may need different reasoning depending on which branch of the graph is active.

The ingestion CLI supports that directly:

```bash
python -m msa_zria.ingest \
  --input examples/customer_support_cases.jsonl \
  --output examples/customer_support_records_train.jsonl \
  --input-mode hybrid \
  --workspace urn:wwkg:workspace:example \
  --branch support-hotfix
```

This produces canonical training records and audit lineage.

## 5. Produce the Gemma 4 Fine-Tuning Dataset

The ingestion stage expands each source case into structured `parse`, `code`, and `evaluate` training records through [build_records_for_case](src/msa_zria/ingest.py).

The output contract is [DatasetRecord](src/msa_zria/data.py), which contains:

- chat-style `messages`
- typed `target`
- metadata including case id and KG scope

This is the dataset Gemma 4 trains on.

In production, you typically create:

- a training split
- an evaluation split
- optionally a holdout acceptance split

Repository examples:

- [customer_support_records_train.jsonl](examples/customer_support_records_train.jsonl)
- [customer_support_records_eval.jsonl](examples/customer_support_records_eval.jsonl)

### Recommended Gemma 4 Record Counts

For the Gemma fine-tuning stage, the right number depends on domain breadth and answer variability, but the practical ranges are:

- minimum viable pilot: `1,000-3,000` canonical `DatasetRecord` rows
- first production pass: `10,000-30,000` canonical rows
- broader production coverage: `30,000-100,000+` canonical rows

Because one source case usually expands into multiple `parse`, `code`, and `evaluate` records, those row counts typically mean:

- pilot: roughly `350-1,000` source cases
- first production pass: roughly `3,500-10,000` source cases
- broader production coverage: roughly `10,000-35,000+` source cases

For evaluation and acceptance:

- keep at least `200-500` evaluation records even for a small pilot
- prefer `10-20%` of training scale for the evaluation split
- maintain a separate holdout set for final promotion decisions

Quantity is not enough by itself. The dataset should also cover:

- multiple workspaces and branches
- normal cases, hard negatives, and escalation boundaries
- stale-graph or policy-change cases
- operator-override examples where human intervention is required

## 6. Fine-Tune Gemma 4 for the MSA Stages

The current fine-tuning path is [train.py](src/msa_zria/train.py). It trains Gemma 4 on the canonical record dataset using `trl.SFTTrainer`.

The model target is configured in:

- [configs/gemma4_12b.yaml](configs/gemma4_12b.yaml)
- [configs/gemma4_12b_a770.yaml](configs/gemma4_12b_a770.yaml)
- [configs/gemma4_12b_v100.yaml](configs/gemma4_12b_v100.yaml)

Typical production flow:

1. Use the A770 or V100 config that matches the training machine.
2. Point `data.train_path` and `data.eval_path` at your canonical record files.
3. Run fine-tuning.
4. Save the resulting Gemma adapter or output model artifact.

Command:

```bash
python -m msa_zria.train --config configs/gemma4_12b.yaml
```

What this gives you:

- a Gemma model specialized for `parse`, `code`, and `evaluate`
- audit lineage for model config, dataset version, and artifact hash

What it does not give you:

- the final graph decision policy by itself

That part is handled by `Zria_Graph` and the rules backend.

## 7. Build the Zria_Graph Training Set

The `Zria_Graph` backend trains separately from Gemma. Its current example format is [ZRIAExample](src/msa_zria/zria.py).

Each example should contain:

- `query`
- `parsed`
- `target`
- optional `kg_scope`
- `neighborhood`

The important production step is to generate graph neighborhoods that reflect the same WWKG retrieval policy you will use in production.

That means for each training case:

1. resolve the intended workspace and branch
2. retrieve the local WWKG neighborhood
3. save that neighborhood into the training example
4. attach the correct final `EvaluationTarget`

Repository examples:

- [zria_examples_train.jsonl](examples/zria_examples_train.jsonl)
- [zria_examples_eval.jsonl](examples/zria_examples_eval.jsonl)

### Recommended Zria_Graph Record Counts

For `Zria_Graph`, the labeled set can be smaller than the Gemma corpus because each example already carries structured parsed state and a WWKG neighborhood, but you still need enough cases to cover graph and policy variation:

- minimum viable pilot: `500-2,000` labeled graph examples
- first production pass: `5,000-20,000` labeled graph examples
- broader production coverage: `20,000-100,000+` labeled graph examples

If you enable WWKG self-supervised pretraining, the unlabeled graph sample can be much larger:

- useful starting range: `50,000-500,000+` neighborhood samples

For evaluation:

- keep at least `200-500` labeled graph examples
- ensure every major workspace, branch, and policy family appears in eval
- add targeted comparison slices for `rules` versus `learned_graph`

The highest-value labeled cases are usually:

- ambiguous neighborhoods where retrieval is noisy
- branch-sensitive decisions
- low-confidence examples that should fall back to rules
- examples with operator overrides or explicit compliance constraints

## 8. Train Zria_Graph

The upgraded `learned_graph` path now uses:

- induced subgraph tensors
- relation ids
- spectral positional features
- harmonic regularization
- parsed-state to graph fusion
- optional self-supervised graph pretraining
- confidence calibration
- graph explanation output

Training entrypoint:

```bash
python -m msa_zria.zria train \
  --train examples/zria_examples_train.jsonl \
  --eval examples/zria_examples_eval.jsonl \
  --backend-type learned_graph \
  --spectral-k 16 \
  --dropout 0.2 \
  --graph-layers 2 \
  --harmonic-reg-weight 5e-4 \
  --pretrain-epochs 10 \
  --memory-momentum 0.9 \
  --patience 10 \
  --output outputs/zria_graph_learned
```

What this stage produces:

- a `learned_graph` artifact
- relation vocabulary and graph shape metadata
- calibrated confidence temperature
- persisted ZRIA memory vectors
- self-supervised pretraining metadata

## 9. Keep Rules as the Fallback Safety Layer

Even with `Zria_Graph`, rules remain important in production.

The rules backend should capture:

- hard policy constraints
- must-escalate cases
- branch-specific overrides
- temporary operational controls
- emergency fallback behavior when graph inference is low-confidence

Rules are currently loaded from:

- [zria_rules.json](examples/zria_rules.json)

In production, the normal deployment pattern is:

- primary decision backend: `learned_graph`
- fallback backend: `rules`

This gives you:

- adaptive graph reasoning when the model is confident
- deterministic control when the model is uncertain or retrieval fails

## 10. Wire Gemma and Zria_Graph into the Runtime Pipeline

The end-to-end reasoning loop is implemented by [ReasoningPipeline](src/msa_zria/reasoning_pipeline.py).

A typical production request flows like this:

1. user text arrives
2. Gemma `parse` module converts it into `ParseTarget`
3. Gemma `code` module produces a constrained `CodeTarget`
4. the controlled Pyro runner executes the program
5. Gemma `evaluate` module assesses the Pyro answer
6. `Zria_Graph` is invoked when a graph-backed final decision is required
7. rules take over if `Zria_Graph` is below threshold or retrieval fails

Operationally, there are two common patterns:

- MSA-first pattern:
  Gemma handles parse, code, and evaluation, and `Zria_Graph` is the final decision layer.
- Graph-first pattern:
  Gemma handles parsing, then `Zria_Graph` produces the production decision directly, with rules as fallback.

The second pattern is often the best fit when the final answer is mainly a graph-backed business decision.

### Runtime Reasoning Branches

The runtime now separates backend `mode` from reasoning `branch`.

- `mode` still chooses `pyro`, `zria`, or `hybrid`
- `reasoning_branch` chooses which Gemma path supplies parse/code/evaluate behavior

Recommended naming:

- `non_thinking`: the current narrow production path
- `thinking`: the specialist reasoning path trained on deeper verified traces

This keeps the external contract stable while letting the caller choose how much reasoning depth is worth paying for on a request.

The specialist path is not just another checkpoint name. It should be trained from specialist source cases such as:

- focused domain rules
- verified claims and evidence snippets
- exemplar reasoning traces
- verification checks that define what a defensible answer must preserve

The repository supports that with:

- source specialist cases: `examples/thinking_cases_train.jsonl`
- specialist ingest: `msa-zria thinking-ingest`
- specialist training config: `configs/gemma4_12b_thinking.yaml`
- specialist runtime model path: `LM_PATH_THINKING`

Example:

```bash
LM_PATH_NON_THINKING=outputs/gemma4_12b \
LM_PATH_THINKING=outputs/gemma4_12b_thinking \
msa-zria infer \
  --query "The monitor failed after the box was opened." \
  --mode pyro \
  --reasoning-branch thinking
```

## 11. Validate Before Promotion

Before a model is production-ready, run three separate checks.

### Contract Validation

Confirm the parse, code, and evaluation outputs still conform to the typed contracts.

### Backend Validation

Compare `learned_graph` against rules on held-out cases.

### End-to-End Validation

Run the full pipeline with realistic branch-aware WWKG contexts.

Useful commands:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

```bash
python -m msa_zria.zria compare \
  --model outputs/zria_graph_learned \
  --rules examples/zria_rules.json \
  --input examples/zria_examples_eval.jsonl
```

```bash
python -m msa_zria.ablation --config configs/gemma4_12b.yaml
```

If WWKG is live, also run the live checks:

```bash
WWKG_BASE_URL=http://127.0.0.1:4242 \
WWKG_WORKSPACE=urn:wwkg:workspace:example \
WWKG_BRANCH=main \
./scripts/run_live_wwkg_checks.sh
```

## 12. Package the Production System

A production deployment should package these assets together:

- Gemma fine-tuned model artifacts
- `learned_graph` artifact directory
- rules bundle
- experiment config
- WWKG connection and scope defaults
- audit output configuration

At runtime, the production service should know:

- which Gemma artifact to use
- which `learned_graph` artifact to use
- which rules bundle to use
- which WWKG workspace and branch are active
- what confidence threshold triggers rules fallback

## 13. Run in Production

A typical production inference path is:

1. receive request text
2. attach tenant, workspace, branch, and any operator context
3. call Gemma parse
4. retrieve the WWKG neighborhood in the active branch
5. call `Zria_Graph`
6. if `Zria_Graph` confidence is below threshold, fall back to rules
7. optionally run Pyro reasoning for probabilistic support
8. return the final `EvaluationTarget`
9. store audit and lineage events

The key production property of the current implementation is that the final decision is not just a model label. It carries:

- configured backend
- effective backend
- confidence
- calibrated confidence inputs
- whether fallback fired
- graph explanation edges for `learned_graph`

## 14. Monitor and Retrain

Once live, treat every decision as future training data.

Capture:

- input query
- parsed state
- workspace and branch
- retrieved output record ids
- final backend used
- whether rules fallback fired
- graph explanation edges
- operator override if any
- final disposition

Then use those records to:

- expand the canonical case corpus
- regenerate `DatasetRecord` training data for Gemma
- regenerate `ZRIAExample` graph training data
- retrain Gemma
- retrain `Zria_Graph`
- compare the new branch against the current production branch before promotion

## 15. Practical Production Recipe

If you were starting from a fresh text corpus today, the shortest sensible production path would be:

1. Curate source cases into `CustomerSupportCase` JSONL with expected parse, code, and evaluation outputs.
2. Ingest them into canonical `DatasetRecord` JSONL with WWKG scope metadata.
3. Fine-tune Gemma 4 on the canonical records.
4. Build branch-aware `ZRIAExample` files by retrieving and storing WWKG neighborhoods for each case.
5. Train `learned_graph` with self-supervised pretraining enabled.
6. Keep rules as the fallback backend.
7. Validate contracts, backend comparison, and the full reasoning pipeline.
8. Promote a tested WWKG branch and matching model artifacts into production together.

That gives you a production system where:

- Gemma 4 handles structured understanding and controlled generation
- `Zria_Graph` performs graph-native reasoning over WWKG neighborhoods
- rules enforce deterministic safety and policy controls
- audit lineage tracks the whole path from corpus to deployed decision system
