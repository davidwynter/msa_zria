from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Any

from msa_zria.audit import AuditRecorder, sha256_directory, sha256_json, stable_dataset_version
from msa_zria.config import ExperimentConfig, load_experiment_config

if TYPE_CHECKING:
    from datasets import Dataset


def _limit_dataset(dataset: "Dataset", limit: int | None) -> "Dataset":
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _render_chat_text(dataset: "Dataset", processor: Any) -> "Dataset":
    columns = dataset.column_names

    def convert_record(record: dict) -> dict[str, str]:
        text = processor.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(convert_record, remove_columns=columns)


def load_record_datasets(config: ExperimentConfig, processor: Any) -> tuple["Dataset", "Dataset" | None]:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load JSONL training records."
        ) from exc

    train_dataset = load_dataset("json", data_files=config.data.train_path, split="train")
    train_dataset = _limit_dataset(train_dataset, config.data.max_train_samples)
    train_dataset = _render_chat_text(train_dataset, processor)

    eval_dataset = None
    if config.data.eval_path and Path(config.data.eval_path).exists():
        eval_dataset = load_dataset("json", data_files=config.data.eval_path, split="train")
        eval_dataset = _limit_dataset(eval_dataset, config.data.max_eval_samples)
        eval_dataset = _render_chat_text(eval_dataset, processor)

    return train_dataset, eval_dataset


def train_from_config(config: ExperimentConfig, audit_recorder: AuditRecorder | None = None) -> str:
    try:
        from trl import SFTTrainer
        from msa_zria.training import (
            build_lora_config,
            build_sft_config,
            load_model_and_processor,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Training requires the full fine-tuning stack: torch, transformers, peft, datasets, and trl."
        ) from exc

    model, processor = load_model_and_processor(
        config.model,
        accelerator=config.training.accelerator,
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    train_dataset, eval_dataset = load_record_datasets(config, processor)
    peft_config = build_lora_config(config.training)
    sft_config = build_sft_config(config.training)

    if eval_dataset is None:
        sft_config.eval_strategy = "no"

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        dataset_text_field="text",
    )
    trainer.train()
    trainer.save_model()
    processor.save_pretrained(config.training.output_dir)
    if audit_recorder is not None:
        audit_recorder.record_model_lineage(
            experiment_config_hash=sha256_json(config.model_dump(mode="json")),
            training_dataset_version=stable_dataset_version(
                [config.data.train_path, config.data.eval_path] if config.data.eval_path else [config.data.train_path]
            ),
            model_artifact_path=config.training.output_dir,
            model_artifact_hash=sha256_directory(config.training.output_dir),
            backend_type=config.zria.backend,
            fallback_setting=config.zria.fallback_to_rules,
            confidence_threshold=config.zria.confidence_threshold,
        )
    return config.training.output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 12B with canonical msa_zria JSONL records.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to ExperimentConfig YAML.",
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    audit_recorder = AuditRecorder.from_experiment(config)
    output_dir = train_from_config(config, audit_recorder=audit_recorder)
    print(f"Training complete. Saved model artifacts to {output_dir}")


if __name__ == "__main__":
    main()
