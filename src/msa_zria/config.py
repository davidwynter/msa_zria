from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_model_id: str = "google/gemma-4-12B"
    processor_id: str = "google/gemma-4-12B-it"
    load_in_4bit: bool = True
    use_double_quant: bool = True
    quant_type: str = "nf4"
    prepare_for_kbit_training: bool = True


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: str = "outputs/gemma4_12b"
    accelerator: str = "auto"
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_length: int = 2048
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    report_to: str = "none"
    remove_unused_columns: bool = False
    skip_prepare_dataset: bool = True
    modules_to_save: list[str] = Field(
        default_factory=lambda: ["lm_head", "embed_tokens"]
    )


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    seed: int = 42
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(raw)
