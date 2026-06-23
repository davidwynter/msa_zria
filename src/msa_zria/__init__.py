from msa_zria.audit import (
    AuditEvent,
    AuditRecorder,
    event_report_id,
    sha256_directory,
    sha256_file,
    sha256_json,
    sha256_text,
    stable_dataset_version,
)
from msa_zria.config import ExperimentConfig, KGConfig, KGScope, ZRIAConfig, load_experiment_config
from msa_zria.data import (
    CodeTarget,
    DatasetRecord,
    EvaluationCase,
    EvaluationResult,
    EvaluationTarget,
    ParseTarget,
    StructuredTrainingExample,
    TrainingExample,
    Triple,
    evaluate_case,
    write_jsonl_records,
)
from msa_zria.ablation import (
    AblationCase,
    AblationReport,
    build_baseline_pipeline,
    load_ablation_cases,
    run_ablation,
    write_ablation_report,
)
from msa_zria.ingest import (
    CustomerSupportCase,
    build_records_for_case,
    ingest_customer_support_cases,
    load_customer_support_cases,
)
from msa_zria.kg import kg_context_metadata, load_triples
from msa_zria.pyro_runtime import PyroExecutionResult, execute_pyro_program
from msa_zria.reasoning_pipeline import PipelineResult, ReasoningPipeline
from msa_zria.validate_contracts import ContractValidationDetail, ContractValidationReport, ContractValidationSummary, validate_contracts
from msa_zria.zria import (
    ZRIAComparisonDetail,
    ZRIAComparisonReport,
    ZRIAComparisonSummary,
    ZRIAExample,
    compare_backends,
    evaluate_learned_backend,
    load_zria_examples,
    train_model as train_zria_model,
)
from msa_zria.zria_adapter import BaseZRIAAdapter, ConfiguredZRIAAdapter, HeuristicZRIAAdapter, RuleBasedZRIAAdapter
from msa_zria.zria_backend import (
    BaseZRIABackend,
    LearnedZRIABackend,
    RemoteZRIABackend,
    RuleBasedZRIABackend,
    ZRIABundle,
    ZRIAPredictionTrace,
    ZRIARule,
    load_zria_backend,
)

try:
    from msa_zria.training import (
        build_lora_config,
        build_sft_config,
        load_model_and_processor,
    )
except ModuleNotFoundError:
    build_lora_config = None
    build_sft_config = None
    load_model_and_processor = None

try:
    from msa_zria.train import train_from_config
except ModuleNotFoundError:
    train_from_config = None

__all__ = [
    "AblationCase",
    "AblationReport",
    "AuditEvent",
    "AuditRecorder",
    "BaseZRIAAdapter",
    "BaseZRIABackend",
    "CodeTarget",
    "ConfiguredZRIAAdapter",
    "ContractValidationDetail",
    "ContractValidationReport",
    "ContractValidationSummary",
    "CustomerSupportCase",
    "DatasetRecord",
    "ExperimentConfig",
    "EvaluationCase",
    "EvaluationResult",
    "EvaluationTarget",
    "HeuristicZRIAAdapter",
    "KGConfig",
    "KGScope",
    "LearnedZRIABackend",
    "RuleBasedZRIABackend",
    "RuleBasedZRIAAdapter",
    "RemoteZRIABackend",
    "ParseTarget",
    "PipelineResult",
    "PyroExecutionResult",
    "ReasoningPipeline",
    "StructuredTrainingExample",
    "TrainingExample",
    "Triple",
    "ZRIAComparisonDetail",
    "ZRIAComparisonReport",
    "ZRIAComparisonSummary",
    "ZRIAExample",
    "build_baseline_pipeline",
    "build_lora_config",
    "build_sft_config",
    "build_records_for_case",
    "compare_backends",
    "event_report_id",
    "evaluate_case",
    "evaluate_learned_backend",
    "execute_pyro_program",
    "ingest_customer_support_cases",
    "kg_context_metadata",
    "load_ablation_cases",
    "load_customer_support_cases",
    "load_experiment_config",
    "load_model_and_processor",
    "load_triples",
    "load_zria_examples",
    "load_zria_backend",
    "run_ablation",
    "sha256_directory",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "stable_dataset_version",
    "train_zria_model",
    "train_from_config",
    "validate_contracts",
    "write_ablation_report",
    "write_jsonl_records",
    "ZRIABundle",
    "ZRIAConfig",
    "ZRIAPredictionTrace",
    "ZRIARule",
]
