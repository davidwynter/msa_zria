from __future__ import annotations

import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from msa_zria.audit import AuditRecorder
from msa_zria.config import KGConfig, KGScope, ZRIAConfig, load_experiment_config
from msa_zria.data import CodeTarget, EvaluationTarget, ParseTarget
from msa_zria.kg import kg_context_metadata
from msa_zria.pyro_runtime import execute_pyro_program
from msa_zria.zria_adapter import BaseZRIAAdapter, ConfiguredZRIAAdapter

ReasoningBranch = Literal["non_thinking", "thinking"]
InferenceMode = Literal["pyro", "zria", "hybrid"]


class BranchConfigurationError(RuntimeError):
    pass


class InferenceModuleNotConfiguredError(RuntimeError):
    pass


class InferenceExecutionError(RuntimeError):
    pass


class UnsupportedModeError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModuleBundle:
    parse: Any | None
    code: Any | None
    evaluate: Any | None

    def merge_missing(self, fallback: "ModuleBundle") -> "ModuleBundle":
        return ModuleBundle(
            parse=self.parse or fallback.parse,
            code=self.code or fallback.code,
            evaluate=self.evaluate or fallback.evaluate,
        )

    def has_any(self) -> bool:
        return any(module is not None for module in (self.parse, self.code, self.evaluate))

    def is_configured(self) -> bool:
        return all(module is not None for module in (self.parse, self.code, self.evaluate))


@dataclass(frozen=True)
class RuntimeDependencies:
    non_thinking: ModuleBundle
    thinking: ModuleBundle | None
    zria_adapter: BaseZRIAAdapter
    audit_recorder: AuditRecorder | None


def row_metadata(scope: KGScope | KGConfig | None) -> dict[str, str]:
    return kg_context_metadata(scope)


def _call_module(module: Any, kg_scope: KGScope | None, **kwargs: Any) -> Any:
    if module is None:
        raise InferenceModuleNotConfiguredError("Inference module is not configured.")
    try:
        signature = inspect.signature(module)
    except (TypeError, ValueError):
        signature = None
    if signature and "kg_context" in signature.parameters and kg_scope is not None:
        kwargs["kg_context"] = kg_scope.model_dump(exclude_none=True)
    try:
        return module(**kwargs)
    except Exception as exc:
        raise InferenceExecutionError(f"Module execution error: {exc}") from exc


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


def _bundle_from_lm_path(lm_path: str | None) -> ModuleBundle:
    if not lm_path:
        return ModuleBundle(parse=None, code=None, evaluate=None)
    try:
        import dspy
        from msa_zria.dspy_modules import CodeGenModule, EvalModule, ParseModule
    except ModuleNotFoundError:
        return ModuleBundle(parse=None, code=None, evaluate=None)

    llm = dspy.LM(model=lm_path)
    return ModuleBundle(
        parse=ParseModule(llm=llm),
        code=CodeGenModule(llm=llm),
        evaluate=EvalModule(llm=llm),
    )


def _default_module_bundle(reasoning_branch: ReasoningBranch) -> ModuleBundle:
    if reasoning_branch == "thinking":
        return _thinking_bundle_from_lm_path(os.getenv("LM_PATH_THINKING"))
    return _bundle_from_lm_path(
        os.getenv("LM_PATH_NON_THINKING", os.getenv("LM_PATH", "outputs/gemma4_12b"))
    )


def _thinking_bundle_from_lm_path(lm_path: str | None) -> ModuleBundle:
    if not lm_path:
        return ModuleBundle(parse=None, code=None, evaluate=None)
    try:
        import dspy
        from msa_zria.thinking_dspy_modules import (
            ThinkingCodeGenModule,
            ThinkingEvalModule,
            ThinkingParseModule,
        )
    except ModuleNotFoundError:
        return ModuleBundle(parse=None, code=None, evaluate=None)

    llm = dspy.LM(model=lm_path)
    return ModuleBundle(
        parse=ThinkingParseModule(llm=llm),
        code=ThinkingCodeGenModule(llm=llm),
        evaluate=ThinkingEvalModule(llm=llm),
    )


def build_runtime_dependencies(
    *,
    parse_module: Any | None = None,
    code_module: Any | None = None,
    eval_module: Any | None = None,
    thinking_parse_module: Any | None = None,
    thinking_code_module: Any | None = None,
    thinking_eval_module: Any | None = None,
    zria_adapter: BaseZRIAAdapter | None = None,
    audit_recorder: AuditRecorder | None = None,
) -> RuntimeDependencies:
    non_thinking = ModuleBundle(parse=parse_module, code=code_module, evaluate=eval_module).merge_missing(
        _default_module_bundle("non_thinking")
    )

    thinking_seed = ModuleBundle(
        parse=thinking_parse_module,
        code=thinking_code_module,
        evaluate=thinking_eval_module,
    )
    thinking: ModuleBundle | None = None
    if thinking_seed.has_any() or os.getenv("LM_PATH_THINKING"):
        thinking = thinking_seed.merge_missing(_default_module_bundle("thinking"))

    return RuntimeDependencies(
        non_thinking=non_thinking,
        thinking=thinking,
        zria_adapter=zria_adapter or _default_zria_adapter(),
        audit_recorder=audit_recorder if audit_recorder is not None else _default_audit_recorder(),
    )


def available_reasoning_branches(runtime: RuntimeDependencies) -> list[ReasoningBranch]:
    branches: list[ReasoningBranch] = []
    if runtime.non_thinking.is_configured():
        branches.append("non_thinking")
    if runtime.thinking is not None and runtime.thinking.is_configured():
        branches.append("thinking")
    return branches


def resolve_module_bundle(runtime: RuntimeDependencies, reasoning_branch: ReasoningBranch) -> ModuleBundle:
    if reasoning_branch == "thinking":
        if runtime.thinking is None or not runtime.thinking.is_configured():
            raise BranchConfigurationError(
                "Requested reasoning branch 'thinking' is not configured. "
                "Set LM_PATH_THINKING or provide branch-specific modules."
            )
        return runtime.thinking
    if not runtime.non_thinking.is_configured():
        raise BranchConfigurationError(
            "Requested reasoning branch 'non_thinking' is not configured. "
            "Set LM_PATH or LM_PATH_NON_THINKING or provide non-thinking modules."
        )
    return runtime.non_thinking


def parse_query(
    runtime: RuntimeDependencies,
    text: str,
    *,
    kg_scope: KGScope | None = None,
    reasoning_branch: ReasoningBranch = "non_thinking",
) -> ParseTarget:
    bundle = resolve_module_bundle(runtime, reasoning_branch)
    payload = _extract_payload(_call_module(bundle.parse, kg_scope, text=text), "parsed_result")
    return ParseTarget.model_validate(payload)


def synthesize_code(
    runtime: RuntimeDependencies,
    parsed: dict[str, Any],
    *,
    kg_scope: KGScope | None = None,
    reasoning_branch: ReasoningBranch = "non_thinking",
) -> CodeTarget:
    bundle = resolve_module_bundle(runtime, reasoning_branch)
    return _coerce_code_target(_call_module(bundle.code, kg_scope, parsed=parsed))


def evaluate_answer(
    runtime: RuntimeDependencies,
    query: str,
    answer: Any,
    *,
    kg_scope: KGScope | None = None,
    reasoning_branch: ReasoningBranch = "non_thinking",
) -> EvaluationTarget:
    if isinstance(answer, EvaluationTarget):
        return answer
    bundle = resolve_module_bundle(runtime, reasoning_branch)
    payload = _extract_payload(_call_module(bundle.evaluate, kg_scope, query=query, answer=answer), "evaluation")
    return EvaluationTarget.model_validate(payload)


def run_pyro(
    runtime: RuntimeDependencies,
    parsed: dict[str, Any],
    *,
    kg_scope: KGScope | None = None,
    reasoning_branch: ReasoningBranch = "non_thinking",
) -> dict[str, Any]:
    code_target = synthesize_code(
        runtime,
        parsed,
        kg_scope=kg_scope,
        reasoning_branch=reasoning_branch,
    )
    result = execute_pyro_program(code_target.program, entrypoint=code_target.entrypoint)
    _audit_control_events(runtime.audit_recorder, result.control_events, kg_scope)
    return {
        "pyro_result": result.model_dump(),
        "kg_scope": row_metadata(kg_scope),
        "reasoning_branch": reasoning_branch,
    }


def infer(
    runtime: RuntimeDependencies,
    query: str,
    *,
    mode: InferenceMode,
    kg_scope: KGScope | None = None,
    operator_override: dict[str, Any] | None = None,
    reasoning_branch: ReasoningBranch = "non_thinking",
) -> dict[str, Any]:
    parsed = parse_query(
        runtime,
        query,
        kg_scope=kg_scope,
        reasoning_branch=reasoning_branch,
    )

    if mode == "pyro":
        pyro_response = run_pyro(
            runtime,
            parsed.model_dump(),
            kg_scope=kg_scope,
            reasoning_branch=reasoning_branch,
        )
        final_evaluation = evaluate_answer(
            runtime,
            query,
            pyro_response["pyro_result"].get("answer"),
            kg_scope=kg_scope,
            reasoning_branch=reasoning_branch,
        )
        if runtime.audit_recorder is not None:
            runtime.audit_recorder.record_decision_lineage(
                query=query,
                parsed_state=parsed,
                backend_used="pyro",
                fallback_fired=False,
                final_evaluation=final_evaluation,
                operator_override=operator_override,
                kg_scope=kg_scope,
            )
        return pyro_response

    if mode == "zria":
        trace = runtime.zria_adapter.predict_with_trace(query, parsed=parsed, kg_scope=kg_scope)
        if runtime.audit_recorder is not None:
            runtime.audit_recorder.record_decision_lineage(
                query=query,
                parsed_state=parsed,
                backend_used=trace.effective_backend,
                fallback_fired=trace.fallback_fired,
                final_evaluation=trace.prediction,
                operator_override=operator_override,
                kg_scope=kg_scope,
            )
            if trace.fallback_fired:
                _record_zria_fallback(runtime.audit_recorder, trace, kg_scope)
        return {
            "answer": trace.prediction.model_dump(),
            "kg_scope": row_metadata(kg_scope),
            "reasoning_branch": reasoning_branch,
        }

    if mode == "hybrid":
        zria_trace = runtime.zria_adapter.predict_with_trace(query, parsed=parsed, kg_scope=kg_scope)
        pyro_response = run_pyro(
            runtime,
            parsed.model_dump(),
            kg_scope=kg_scope,
            reasoning_branch=reasoning_branch,
        )
        pyro_result = pyro_response["pyro_result"]
        chosen_backend = "pyro"
        if pyro_result.get("success"):
            final_evaluation = evaluate_answer(
                runtime,
                query,
                pyro_result.get("answer"),
                kg_scope=kg_scope,
                reasoning_branch=reasoning_branch,
            )
            response = {
                "answer": pyro_result["answer"],
                "kg_scope": row_metadata(kg_scope),
                "reasoning_branch": reasoning_branch,
            }
        else:
            chosen_backend = zria_trace.effective_backend
            final_evaluation = zria_trace.prediction
            response = {
                "answer": zria_trace.prediction.model_dump(),
                "kg_scope": row_metadata(kg_scope),
                "reasoning_branch": reasoning_branch,
            }
        if runtime.audit_recorder is not None:
            runtime.audit_recorder.record_decision_lineage(
                query=query,
                parsed_state=parsed,
                backend_used=chosen_backend,
                fallback_fired=zria_trace.fallback_fired or not pyro_result.get("success"),
                final_evaluation=final_evaluation,
                operator_override=operator_override,
                kg_scope=kg_scope,
            )
            if zria_trace.fallback_fired:
                _record_zria_fallback(runtime.audit_recorder, zria_trace, kg_scope)
        return response

    raise UnsupportedModeError(f"Invalid mode '{mode}'. Expected pyro, zria, or hybrid.")


def _audit_control_events(
    audit_recorder: AuditRecorder | None,
    control_events: list[dict[str, Any]],
    kg_scope: KGScope | None,
) -> None:
    if audit_recorder is None:
        return
    for event in control_events:
        audit_recorder.record_control_event(
            control_type=event.get("control_type", "unknown_control"),
            backend_type=event.get("backend_type"),
            details=event.get("details", {}),
            kg_scope=kg_scope,
        )


def _record_zria_fallback(
    audit_recorder: AuditRecorder,
    trace: Any,
    kg_scope: KGScope | None,
) -> None:
    details = {"fallback_reason": trace.fallback_reason}
    if trace.confidence is not None:
        details["confidence"] = trace.confidence
    audit_recorder.record_control_event(
        control_type=trace.control_event_type or "zria_fallback",
        backend_type=trace.configured_backend,
        details=details,
        kg_scope=kg_scope,
    )
