# Dataset card: BANKING77 for `msa_zria`

## Dataset

BANKING77 is a customer-service intent classification benchmark for the banking domain. The original paper describes it as a challenging single-domain intent detection dataset with 13,083 annotated examples across 77 intents.

## Why this dataset was selected

It matches the `msa_zria` production shape:

- unstructured customer text
- a support triage context
- stable labels that can be transformed into typed parse/evaluation contracts
- natural fit for graph neighborhoods such as `intent -> recommendedVerdict`, `intent -> shouldEscalate`, and `service -> hasIssue`
- suitable for LLM fine-tuning, adapter fine-tuning, and downstream comparison with deterministic rules

## Artifact mapping

| Pipeline requirement | Produced artifact |
|---|---|
| Source cases | `examples/banking77_customer_support_cases.jsonl` |
| Gemma train records | `examples/banking77_customer_support_records_train.jsonl` |
| Gemma eval records | `examples/banking77_customer_support_records_eval.jsonl` |
| ZRIA train examples | `examples/banking77_zria_examples_train.jsonl` |
| ZRIA eval examples | `examples/banking77_zria_examples_eval.jsonl` |
| Rules fallback | `examples/banking77_zria_rules.json` |
| Ablation input | `examples/banking77_ablation_cases.jsonl` |
| Training config | `configs/gemma4_12b_banking77.yaml` |
| Hardware configs | `configs/gemma4_12b_banking77_a770.yaml`, `configs/gemma4_12b_banking77_v100.yaml` |

## Label-to-contract transformation

Each BANKING77 row is converted as follows:

1. `text` becomes `customer_message` and ZRIA `query`.
2. `label` becomes an intent string.
3. The intent is converted into `ParseTarget.issue`.
4. Rule-derived risk patterns assign `EvaluationTarget`:
   - high-risk/exception intents -> `escalate`
   - account-specific informational intents -> `insufficient_information`
   - normal workflow intents -> `resolved`
5. A small graph neighborhood is attached to each case.
6. Each source case expands to three Gemma records: `parse`, `code`, and `evaluate`.

## Caveat

The bundled files are a smoke corpus for immediate offline testing. Run `make hf` to generate the benchmark artifacts from the real BANKING77 dataset.
