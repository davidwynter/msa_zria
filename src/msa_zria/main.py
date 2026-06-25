from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from msa_zria.audit import AuditRecorder
from msa_zria.config import KGConfig, KGScope
from msa_zria.data import EvaluationCase, EvaluationResult, evaluate_case
from msa_zria.kg import load_triples
from msa_zria.runtime import (
    BranchConfigurationError,
    evaluate_answer,
    InferenceExecutionError,
    InferenceMode,
    InferenceModuleNotConfiguredError,
    ReasoningBranch,
    UnsupportedModeError,
    build_runtime_dependencies,
    infer as run_inference,
    parse_query,
    row_metadata,
    run_pyro as run_pyro_reasoning,
    synthesize_code,
)
from msa_zria.zria_adapter import BaseZRIAAdapter


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_path: str
    format: str
    graph_path: str | None = None
    kg: KGConfig = Field(default_factory=KGConfig)

    def resolved_kg(self) -> KGConfig:
        if self.graph_path and not self.kg.graph_path:
            return self.kg.model_copy(update={"graph_path": self.graph_path})
        return self.kg


class FineTuneConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    output_dir: str
    accelerator: str = "auto"
    load_in_4bit: bool = True
    epochs: int = 3
    lora_r: int = 16
    learning_rate: float = 2e-4
    batch_size: int = 1
    gradient_checkpointing: bool = True
    seed: int = 42


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    mode: InferenceMode
    kg_scope: KGScope | None = None
    operator_override: dict[str, Any] | None = None
    reasoning_branch: ReasoningBranch = "non_thinking"


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    kg_scope: KGScope | None = None
    reasoning_branch: ReasoningBranch = "non_thinking"


class CodeGenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parsed: dict[str, Any]
    kg_scope: KGScope | None = None
    reasoning_branch: ReasoningBranch = "non_thinking"


class EvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    answer: Any
    kg_scope: KGScope | None = None
    reasoning_branch: ReasoningBranch = "non_thinking"


class ContractEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: EvaluationCase


def _raise_runtime_http_error(exc: Exception) -> None:
    if isinstance(exc, (BranchConfigurationError, InferenceModuleNotConfiguredError)):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, InferenceExecutionError):
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    raise exc


def create_app(
    *,
    parse_module: Any | None = None,
    code_module: Any | None = None,
    eval_module: Any | None = None,
    thinking_parse_module: Any | None = None,
    thinking_code_module: Any | None = None,
    thinking_eval_module: Any | None = None,
    zria_adapter: BaseZRIAAdapter | None = None,
    audit_recorder: AuditRecorder | None = None,
    initialize_ray: bool = True,
) -> FastAPI:
    if initialize_ray:
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
        except ModuleNotFoundError:
            pass

    runtime = build_runtime_dependencies(
        parse_module=parse_module,
        code_module=code_module,
        eval_module=eval_module,
        thinking_parse_module=thinking_parse_module,
        thinking_code_module=thinking_code_module,
        thinking_eval_module=thinking_eval_module,
        zria_adapter=zria_adapter,
        audit_recorder=audit_recorder,
    )
    app = FastAPI(title="MSA Reasoning Service")

    @app.post("/produce_dataset")
    def produce_dataset(cfg: DatasetConfig) -> dict[str, Any]:
        kg = cfg.resolved_kg()
        triples = load_triples(kg)
        metadata = row_metadata(kg)
        os.makedirs(cfg.output_path, exist_ok=True)

        if cfg.format in ["triples", "hybrid"]:
            with open(os.path.join(cfg.output_path, "triples.jsonl"), "w", encoding="utf-8") as handle:
                for triple in triples:
                    row = {
                        "subject": triple.subject,
                        "predicate": triple.predicate,
                        "object": triple.object,
                    }
                    row.update(metadata)
                    handle.write(json.dumps(row) + "\n")

        if cfg.format in ["nl", "hybrid"]:
            with open(os.path.join(cfg.output_path, "nl.jsonl"), "w", encoding="utf-8") as handle:
                for triple in triples:
                    row = {"text": triple.as_sentence()}
                    row.update(metadata)
                    handle.write(json.dumps(row) + "\n")

        return {"status": "dataset_produced", "count": len(triples), "kg_metadata": metadata}

    @app.post("/parse")
    def parse(req: ParseRequest) -> dict[str, Any]:
        try:
            parsed = parse_query(
                runtime,
                req.text,
                kg_scope=req.kg_scope,
                reasoning_branch=req.reasoning_branch,
            )
        except Exception as exc:
            _raise_runtime_http_error(exc)
        return {
            "parsed": parsed.model_dump(mode="json"),
            "kg_scope": row_metadata(req.kg_scope),
            "reasoning_branch": req.reasoning_branch,
        }

    @app.post("/code_synthesis")
    def code_synthesis(req: CodeGenRequest) -> dict[str, Any]:
        try:
            code_target = synthesize_code(
                runtime,
                req.parsed,
                kg_scope=req.kg_scope,
                reasoning_branch=req.reasoning_branch,
            )
        except Exception as exc:
            _raise_runtime_http_error(exc)
        return {
            "code": code_target.model_dump(mode="json"),
            "kg_scope": row_metadata(req.kg_scope),
            "reasoning_branch": req.reasoning_branch,
        }

    @app.post("/run_pyro")
    def run_pyro_endpoint(req: CodeGenRequest) -> dict[str, Any]:
        try:
            return run_pyro_reasoning(
                runtime,
                req.parsed,
                kg_scope=req.kg_scope,
                reasoning_branch=req.reasoning_branch,
            )
        except Exception as exc:
            _raise_runtime_http_error(exc)

    @app.post("/evaluate")
    def evaluate(req: EvalRequest) -> dict[str, Any]:
        try:
            evaluation = evaluate_answer(
                runtime,
                req.query,
                req.answer,
                kg_scope=req.kg_scope,
                reasoning_branch=req.reasoning_branch,
            )
        except Exception as exc:
            _raise_runtime_http_error(exc)
        return {
            "evaluation": evaluation.model_dump(mode="json"),
            "kg_scope": row_metadata(req.kg_scope),
            "reasoning_branch": req.reasoning_branch,
        }

    @app.post("/evaluate_contract")
    def evaluate_contract(req: ContractEvalRequest) -> dict[str, Any]:
        result: EvaluationResult = evaluate_case(req.case)
        return result.model_dump()

    @app.post("/infer")
    def inference(req: InferenceRequest) -> dict[str, Any]:
        try:
            return run_inference(
                runtime,
                req.query,
                mode=req.mode,
                kg_scope=req.kg_scope,
                operator_override=req.operator_override,
                reasoning_branch=req.reasoning_branch,
            )
        except UnsupportedModeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            _raise_runtime_http_error(exc)

    @app.post("/fine_tune")
    def fine_tune_endpoint(cfg: FineTuneConfig) -> Any:
        from msa_zria.fine_tuning import fine_tune as run_fine_tune

        return run_fine_tune(cfg)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
