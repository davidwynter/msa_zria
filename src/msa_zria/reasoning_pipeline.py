from __future__ import annotations

import inspect
from typing import Any

from pydantic import BaseModel, ConfigDict

from msa_zria.config import KGScope
from msa_zria.data import CodeTarget, EvaluationTarget, ParseTarget
from msa_zria.pyro_runtime import PyroExecutionResult, execute_pyro_program
from msa_zria.zria_adapter import BaseZRIAAdapter, HeuristicZRIAAdapter


class PipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parsed: ParseTarget
    code: CodeTarget
    pyro: PyroExecutionResult
    evaluation: EvaluationTarget
    kg_scope: KGScope | None = None


def _extract_payload(result: Any, key: str) -> Any:
    if isinstance(result, dict) and key in result:
        return result[key]
    return result


def _call_with_optional_kg_context(module: Any, kg_scope: KGScope | None, **kwargs: Any) -> Any:
    if kg_scope is None:
        return module(**kwargs)
    try:
        signature = inspect.signature(module)
    except (TypeError, ValueError):
        return module(**kwargs)
    if "kg_context" not in signature.parameters:
        return module(**kwargs)
    return module(kg_context=kg_scope.model_dump(exclude_none=True), **kwargs)


class ReasoningPipeline:
    def __init__(
        self,
        parse_module: Any,
        code_module: Any,
        eval_module: Any,
        zria_adapter: BaseZRIAAdapter | None = None,
    ) -> None:
        self.parse_module = parse_module
        self.code_module = code_module
        self.eval_module = eval_module
        self.zria_adapter = zria_adapter or HeuristicZRIAAdapter()

    def run_parse(self, query: str, kg_scope: KGScope | None = None) -> ParseTarget:
        payload = _extract_payload(
            _call_with_optional_kg_context(self.parse_module, kg_scope, text=query),
            "parsed_result",
        )
        return ParseTarget.model_validate(payload)

    def run_code(self, parsed: ParseTarget, kg_scope: KGScope | None = None) -> CodeTarget:
        payload = _extract_payload(
            _call_with_optional_kg_context(
                self.code_module,
                kg_scope,
                parsed=parsed.model_dump(),
            ),
            "code_str",
        )
        if isinstance(payload, str):
            return CodeTarget(program=payload)
        return CodeTarget.model_validate(payload)

    def run_pyro(self, code: CodeTarget) -> PyroExecutionResult:
        return execute_pyro_program(code.program, entrypoint=code.entrypoint)

    def run_evaluate(
        self,
        query: str,
        answer: Any,
        kg_scope: KGScope | None = None,
    ) -> EvaluationTarget:
        payload = _extract_payload(
            _call_with_optional_kg_context(
                self.eval_module,
                kg_scope,
                query=query,
                answer=answer,
            ),
            "evaluation",
        )
        if isinstance(payload, str):
            return EvaluationTarget(
                verdict="insufficient_information",
                resolved=False,
                should_escalate=False,
                explanation=payload,
        )
        return EvaluationTarget.model_validate(payload)

    def run(self, query: str, kg_scope: KGScope | None = None) -> PipelineResult:
        parsed = self.run_parse(query, kg_scope=kg_scope)
        code = self.run_code(parsed, kg_scope=kg_scope)
        pyro_result = self.run_pyro(code)
        answer = pyro_result.answer if pyro_result.success else pyro_result.error
        evaluation = self.run_evaluate(query, answer, kg_scope=kg_scope)
        return PipelineResult(
            parsed=parsed,
            code=code,
            pyro=pyro_result,
            evaluation=evaluation,
            kg_scope=kg_scope,
        )
