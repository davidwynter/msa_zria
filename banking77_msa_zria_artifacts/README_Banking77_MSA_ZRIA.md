# BANKING77 artifacts for `msa_zria`

This artifact pack prepares a recognised customer-service intent benchmark for the production pipeline described in `PRODUCTION_PIPELINE.md`.

## Dataset selected

**BANKING77**: a single-domain banking customer-service intent dataset introduced by PolyAI. It contains customer support-style utterances classified into 77 intents. It is a good fit for `msa_zria` because the raw input is unstructured support text, the label can be converted into a structured parse target, and the intent can drive graph-backed decisions and rule fallback.

## What is included

```text
banking77_msa_zria_artifacts/
├── configs/
│   ├── gemma4_12b_banking77.yaml
│   ├── gemma4_12b_banking77_a770.yaml
│   └── gemma4_12b_banking77_v100.yaml
├── docs/
│   └── DATASET_CARD.md
├── examples/
│   ├── banking77_ablation_cases.jsonl
│   ├── banking77_customer_support_cases.jsonl
│   ├── banking77_customer_support_records_eval.jsonl
│   ├── banking77_customer_support_records_train.jsonl
│   ├── banking77_manifest.json
│   ├── banking77_zria_examples_eval.jsonl
│   ├── banking77_zria_examples_train.jsonl
│   └── banking77_zria_rules.json
├── scripts/
│   ├── build_banking77_artifacts.py
│   └── validate_banking77_artifacts.py
└── Makefile
```

## Important distinction

The files currently under `examples/` are an **offline smoke-test corpus** generated from the BANKING77 intent taxonomy. They are not claimed to be real BANKING77 rows. They are included so that you can immediately test schema compatibility, config paths, JSONL loading, ZRIA example shape, rule loading, and ablation wiring without needing internet access.

To build the real benchmark artifacts from BANKING77, run:

```bash
cd banking77_msa_zria_artifacts
pip install datasets
make hf
```

The converter tries both Hugging Face dataset IDs `PolyAI/banking77` and `banking77`.

## Local validation

```bash
cd banking77_msa_zria_artifacts
make validate
```

Expected output for the bundled smoke corpus:

```json
{
  "ok": true,
  "case_count": 12,
  "train_records": 27,
  "eval_records": 9,
  "zria_train": 9,
  "zria_eval": 3,
  "ablation_cases": 3,
  "rules": 12
}
```

## Pipeline commands

From inside your `msa_zria` repository root, copy or overlay the contents of this artifact pack so that `examples/`, `configs/`, and `scripts/` sit beside your existing repository paths.

Validate JSONL/contracts:

```bash
python scripts/validate_banking77_artifacts.py examples
```

Run Gemma 4 fine-tuning:

```bash
python -m msa_zria.train --config configs/gemma4_12b_banking77.yaml
```

Run V100 profile:

```bash
python -m msa_zria.train --config configs/gemma4_12b_banking77_v100.yaml
```

Run A770 profile:

```bash
python -m msa_zria.train --config configs/gemma4_12b_banking77_a770.yaml
```

Train graph-native ZRIA:

```bash
python -m msa_zria.zria train \
  --train examples/banking77_zria_examples_train.jsonl \
  --eval examples/banking77_zria_examples_eval.jsonl \
  --backend-type learned_graph \
  --spectral-k 16 \
  --dropout 0.2 \
  --graph-layers 2 \
  --harmonic-reg-weight 5e-4 \
  --pretrain-epochs 10 \
  --memory-momentum 0.9 \
  --patience 10 \
  --output outputs/zria_graph_banking77
```

Run ablation:

```bash
python -m msa_zria.ablation --config configs/gemma4_12b_banking77.yaml
```

Compare learned graph backend to rules after training:

```bash
python -m msa_zria.zria compare \
  --model outputs/zria_graph_banking77 \
  --rules examples/banking77_zria_rules.json \
  --input examples/banking77_zria_examples_eval.jsonl
```

## Output contracts produced

The pack creates the same production artifacts requested in `PRODUCTION_PIPELINE.md`:

- curated `CustomerSupportCase` JSONL
- canonical Gemma `DatasetRecord` train/eval JSONL
- branch-aware `ZRIAExample` train/eval JSONL with graph neighborhoods
- rules fallback bundle
- ablation case file
- Gemma/ZRIA experiment configs
- manifest with SHA-256 hashes
- local validation script

## Recommended next step

Use the smoke corpus first to confirm the pipeline wiring. Then run `make hf` to generate real BANKING77-derived train/eval artifacts and rerun validation before training.
