from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from msa_zria.audit import AuditRecorder
from msa_zria.config import KGConfig, KGScope, ZRIAConfig, load_experiment_config
from msa_zria.data import CodeTarget, EvaluationCase, EvaluationResult, EvaluationTarget, ParseTarget, evaluate_case
from msa_zria.kg import kg_context_metadata, load_triples
from msa_zria.pyro_runtime import execute_pyro_program
from msa_zria.zria_adapter import BaseZRIAAdapter, ConfiguredZRIAAdapter


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
    mode: str
    kg_scope: KGScope | None = None
    operator_override: dict[str, Any] | None = None


class ParseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    kg_scope: KGScope | None = None


class CodeGenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parsed: dict[str, Any]
    kg_scope: KGScope | None = None


class EvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    answer: Any
    kg_scope: KGScope | None = None


class ContractEvalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: EvaluationCase


def _row_metadata(scope: KGScope | KGConfig | None) -> dict[str, str]:
    return kg_context_metadata(scope)


def _call_module(module: Any, kg_scope: KGScope | None, **kwargs: Any) -> Any:
    if module is None:
        raise HTTPException(status_code=503, detail="Inference module is not configured.")
    try:
        signature = inspect.signature(module)
    except (TypeError, ValueError):
        signature = None
    if signature and "kg_context" in signature.parameters and kg_scope is not None:
        kwargs["kg_context"] = kg_scope.model_dump(exclude_none=True)
    try:
        return module(**kwargs)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Module execution error: {exc}") from exc


def _extract_payload(result: Any, key: str) -> Any:
    if isinstance(result, dict) and key in result:
        return result[key]
    return result


def _coerce_code_target(result: Any) -> CodeTarget:
    payload = _extract_payload(result, "code_str")
    if isinstance(payload, str):
        return CodeTarget(program=payload)
    return CodeTarget.model_validate(payload)


def _default_zria_adapter() -> BaseZRIAAdapter:
    config_path = os.getenv("MSA_ZRIA_CONFIG")
    if config_path and Path(config_path).exists():
        experiment_config = load_experiment_config(config_path)
        return ConfiguredZRIAAdapter.from_config(experiment_config.zria, experiment_config.kg)
    return ConfiguredZRIAAdapter.from_config(ZRIAConfig())


def _default_audit_recorder() -> AuditRecorder | None:
    config_path = os.getenv("MSA_ZRIA_CONFIG")
    if config_path and Path(config_path).exists():
        experiment_config = load_experiment_config(config_path)
        return AuditRecorder.from_experiment(experiment_config)
    return None


def _default_modules() -> tuple[Any | None, Any | None, Any | None]:
    try:
        import dspy
        from msa_zria.dspy_modules import CodeGenModule, EvalModule, ParseModule
    except ModuleNotFoundError:
        return None, None, None

    lm_path = os.getenv("LM_PATH", "outputs/gemma4_12b")
    llm = dspy.LM(model=lm_path)
    return ParseModule(llm=llm), CodeGenModule(llm=llm), EvalModule(llm=llm)


def create_app(
    *,
    parse_module: Any | None = None,
    code_module: Any | None = None,
    eval_module: Any | None = None,
    zria_adapter: BaseZRIAAdapter | None = None,
    audit_recorder: AuditRecorder | None = None,
    initialize_ray: bool = True,
) -> FastAPI:
    if parse_module is None or code_module is None or eval_module is None:
        default_parse, default_code, default_eval = _default_modules()
        parse_module = parse_module or default_parse
        code_module = code_module or default_code
        eval_module = eval_module or default_eval

    if zria_adapter is None:
        zria_adapter = _default_zria_adapter()
    if audit_recorder is None:
        audit_recorder = _default_audit_recorder()

    if initialize_ray:
        try:
            import ray

            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
        except ModuleNotFoundError:
            pass

    app = FastAPI(title="MSA Reasoning Service")

    def _audit_control_events(control_events: list[dict[str, Any]], kg_scope: KGScope | None) -> None:
        if audit_recorder is None:
            return
        for event in control_events:
            audit_recorder.record_control_event(
                control_type=event.get("control_type", "unknown_control"),
                backend_type=event.get("backend_type"),
                details=event.get("details", {}),
                kg_scope=kg_scope,
            )

    def _evaluate_final(query: str, answer: Any, kg_scope: KGScope | None) -> EvaluationTarget:
        if isinstance(answer, EvaluationTarget):
            return answer
        if eval_module is None:
            return EvaluationTarget(
                verdict="insufficient_information",
                resolved=False,
                should_escalate=False,
                explanation="Evaluation module was not configured.",
            )
        result = _call_module(eval_module, kg_scope, query=query, answer=answer)
        return EvaluationTarget.model_validate(_extract_payload(result, "evaluation"))

    @app.post("/produce_dataset")
    def produce_dataset(cfg: DatasetConfig) -> dict[str, Any]:
        kg = cfg.resolved_kg()
        triples = load_triples(kg)
        metadata = _row_metadata(kg)
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
        result = _call_module(parse_module, req.kg_scope, text=req.text)
        return {"parsed": _extract_payload(result, "parsed_result"), "kg_scope": _row_metadata(req.kg_scope)}

    @app.post("/code_synthesis")
    def code_synthesis(req: CodeGenRequest) -> dict[str, Any]:
        result = _call_module(code_module, req.kg_scope, parsed=req.parsed)
        return {"code": _extract_payload(result, "code_str"), "kg_scope": _row_metadata(req.kg_scope)}

    @app.post("/run_pyro")
    def run_pyro(req: CodeGenRequest) -> dict[str, Any]:
        code_target = _coerce_code_target(_call_module(code_module, req.kg_scope, parsed=req.parsed))
        result = execute_pyro_program(code_target.program, entrypoint=code_target.entrypoint)
        _audit_control_events(result.control_events, req.kg_scope)
        return {"pyro_result": result.model_dump(), "kg_scope": _row_metadata(req.kg_scope)}

    @app.post("/evaluate")
    def evaluate(req: EvalRequest) -> dict[str, Any]:
        result = _call_module(eval_module, req.kg_scope, query=req.query, answer=req.answer)
        return {"evaluation": _extract_payload(result, "evaluation"), "kg_scope": _row_metadata(req.kg_scope)}

    @app.post("/evaluate_contract")
    def evaluate_contract(req: ContractEvalRequest) -> dict[str, Any]:
        result: EvaluationResult = evaluate_case(req.case)
        return result.model_dump()

    @app.post("/infer")
    def inference(req: InferenceRequest) -> dict[str, Any]:
        parsed_payload = _extract_payload(_call_module(parse_module, req.kg_scope, text=req.query), "parsed_result")
        parsed = ParseTarget.model_validate(parsed_payload)
        if req.mode == "pyro":
            pyro_response = run_pyro(CodeGenRequest(parsed=parsed.model_dump(), kg_scope=req.kg_scope))
            final_evaluation = _evaluate_final(req.query, pyro_response["pyro_result"].get("answer"), req.kg_scope)
            if audit_recorder is not None:
                audit_recorder.record_decision_lineage(
                    query=req.query,
                    parsed_state=parsed,
                    backend_used="pyro",
                    fallback_fired=False,
                    final_evaluation=final_evaluation,
                    operator_override=req.operator_override,
                    kg_scope=req.kg_scope,
                )
            return pyro_response
        if req.mode == "zria":
            trace = zria_adapter.predict_with_trace(
                req.query,
                parsed=parsed,
                kg_scope=req.kg_scope,
            )
            if audit_recorder is not None:
                audit_recorder.record_decision_lineage(
                    query=req.query,
                    parsed_state=parsed,
                    backend_used=trace.effective_backend,
                    fallback_fired=trace.fallback_fired,
                    final_evaluation=trace.prediction,
                    operator_override=req.operator_override,
                    kg_scope=req.kg_scope,
                )
                if trace.fallback_fired:
                    details = {"fallback_reason": trace.fallback_reason}
                    if trace.confidence is not None:
                        details["confidence"] = trace.confidence
                    audit_recorder.record_control_event(
                        control_type=trace.control_event_type or "zria_fallback",
                        backend_type=trace.configured_backend,
                        details=details,
                        kg_scope=req.kg_scope,
                    )
            return {"answer": trace.prediction.model_dump(), "kg_scope": _row_metadata(req.kg_scope)}
        if req.mode == "hybrid":
            zria_trace = zria_adapter.predict_with_trace(req.query, parsed=parsed, kg_scope=req.kg_scope)
            pyro_response = run_pyro(CodeGenRequest(parsed=parsed.model_dump(), kg_scope=req.kg_scope))
            pyro_result = pyro_response["pyro_result"]
            chosen_backend = "pyro"
            if pyro_result.get("success"):
                final_evaluation = _evaluate_final(req.query, pyro_result.get("answer"), req.kg_scope)
                response = {"answer": pyro_result["answer"], "kg_scope": _row_metadata(req.kg_scope)}
            else:
                chosen_backend = zria_trace.effective_backend
                final_evaluation = zria_trace.prediction
                response = {"answer": zria_trace.prediction.model_dump(), "kg_scope": _row_metadata(req.kg_scope)}
            if audit_recorder is not None:
                audit_recorder.record_decision_lineage(
                    query=req.query,
                    parsed_state=parsed,
                    backend_used=chosen_backend,
                    fallback_fired=zria_trace.fallback_fired or not pyro_result.get("success"),
                    final_evaluation=final_evaluation,
                    operator_override=req.operator_override,
                    kg_scope=req.kg_scope,
                )
                if zria_trace.fallback_fired:
                    details = {"fallback_reason": zria_trace.fallback_reason}
                    if zria_trace.confidence is not None:
                        details["confidence"] = zria_trace.confidence
                    audit_recorder.record_control_event(
                        control_type=zria_trace.control_event_type or "zria_fallback",
                        backend_type=zria_trace.configured_backend,
                        details=details,
                        kg_scope=req.kg_scope,
                    )
            return response
        raise HTTPException(status_code=400, detail="Invalid mode")

    @app.post("/fine_tune")
    def fine_tune_endpoint(cfg: FineTuneConfig) -> Any:
        from msa_zria.fine_tuning import fine_tune as run_fine_tune

        return run_fine_tune(cfg)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
