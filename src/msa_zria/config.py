from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class KGScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: str | None = None
    branch: str | None = None
    commit: str | None = None
    as_of: str | None = None

    def to_metadata(self) -> dict[str, str]:
        metadata: dict[str, str] = {}
        if self.workspace:
            metadata["kg_workspace"] = self.workspace
        if self.branch:
            metadata["kg_branch"] = self.branch
        if self.commit:
            metadata["kg_commit"] = self.commit
        if self.as_of:
            metadata["kg_as_of"] = self.as_of
        return metadata


class KGConfig(KGScope):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["oxigraph", "wwkg"] = "wwkg"
    graph_path: str | None = None
    graph_format: str = "nquads"
    graph_iri: str | None = None
    sparql_query: str | None = None
    base_url: str = "http://127.0.0.1:4242"
    api_key: str | None = None
    timeout_seconds: float = 30.0
    retry_attempts: int = 2
    retry_backoff_seconds: float = 0.25
    user_agent: str = "msa-zria/0.1.0"

    def resolved_sparql_query(self) -> str:
        if self.sparql_query:
            return self.sparql_query
        if self.graph_iri:
            return (
                "SELECT ?subject ?predicate ?object WHERE { "
                f"GRAPH <{self.graph_iri}> {{ ?subject ?predicate ?object }} "
                "}"
            )
        return "SELECT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object }"

    def to_metadata(self) -> dict[str, str]:
        metadata = super().to_metadata()
        metadata["kg_backend"] = self.backend
        if self.graph_iri:
            metadata["kg_graph_iri"] = self.graph_iri
        return metadata


class ZRIAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["rules", "learned", "remote"] = "rules"
    rules_path: str = "examples/zria_rules.json"
    learned_model_path: str = "outputs/zria_learned"
    confidence_threshold: float = 0.6
    remote_url: str | None = None
    remote_api_key: str | None = None
    remote_timeout_seconds: float = 10.0
    remote_retry_attempts: int = 1
    remote_retry_backoff_seconds: float = 0.25
    fallback_to_rules: bool = True


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    output_path: str = "outputs/audit/audit.jsonl"
    wwkg_enabled: bool = False
    source: str = "msa_zria"
    promotion_output_path: str = "outputs/audit/promotions.jsonl"


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


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: str = "data/train.jsonl"
    eval_path: str | None = "data/eval.jsonl"
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases_path: str = "data/ablation_cases.jsonl"
    output_path: str = "outputs/ablation_report.json"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str
    seed: int = 42
    kg: KGConfig = Field(default_factory=KGConfig)
    zria: ZRIAConfig = Field(default_factory=ZRIAConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ablation: AblationConfig = Field(default_factory=AblationConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(raw)
