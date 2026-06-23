from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from msa_zria.ablation import HeuristicCodeModule, HeuristicEvalModule, HeuristicParseModule
from msa_zria.audit import AuditRecorder, event_report_id
from msa_zria.data import CodeTarget, EvaluationTarget, ParseTarget
from msa_zria.ingest import CustomerSupportCase, load_customer_support_cases
from msa_zria.pyro_runtime import execute_pyro_program


class ContractValidationDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    parse_valid: bool
    code_valid: bool
    pyro_success: bool
    evaluation_valid: bool
    notes: list[str] = Field(default_factory=list)


class ContractValidationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_cases: int
    parse_valid_cases: int
    code_valid_cases: int
    pyro_success_cases: int
    evaluation_valid_cases: int


class ContractValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_source: str
    summaries: ContractValidationSummary
    details: list[ContractValidationDetail]


def _default_modules() -> tuple[Any, Any, Any]:
    import dspy
    from msa_zria.dspy_modules import CodeGenModule, EvalModule, ParseModule

    llm = dspy.LM(model=os.getenv("LM_PATH", "outputs/gemma4_12b"))
    return ParseModule(llm=llm), CodeGenModule(llm=llm), EvalModule(llm=llm)


def _extract(result: Any, key: str) -> Any:
    if isinstance(result, dict) and key in result:
        return result[key]
    return result


def _call_module(module: Any, kg_context: dict[str, Any] | None, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(module)
    except (TypeError, ValueError):
        signature = None
    if signature and "kg_context" in signature.parameters and kg_context is not None:
        kwargs["kg_context"] = kg_context
    return module(**kwargs)


def _load_modules(source: str) -> tuple[Any, Any, Any]:
    if source == "baseline":
        return HeuristicParseModule(), HeuristicCodeModule(), HeuristicEvalModule()
    if source == "default":
        return _default_modules()
    raise ValueError(f"Unsupported module source '{source}'.")


def validate_contracts(
    cases: list[CustomerSupportCase],
    *,
    module_source: str = "baseline",
    timeout_seconds: float = 5.0,
) -> ContractValidationReport:
    parse_module, code_module, eval_module = _load_modules(module_source)
    details: list[ContractValidationDetail] = []

    for case in cases:
        notes: list[str] = []
        parse_valid = False
        code_valid = False
        pyro_success = False
        evaluation_valid = False
        answer: Any = None

        try:
            parsed_payload = _extract(
                _call_module(
                    parse_module,
                    None if case.kg_scope is None else case.kg_scope.model_dump(exclude_none=True),
                    text=case.customer_message,
                ),
                "parsed_result",
            )
            parsed = ParseTarget.model_validate(parsed_payload)
            parse_valid = True
        except Exception as exc:
            notes.append(f"parse: {exc}")
            details.append(
                ContractValidationDetail(
                    case_id=case.case_id,
                    parse_valid=False,
                    code_valid=False,
                    pyro_success=False,
                    evaluation_valid=False,
                    notes=notes,
                )
            )
            continue

        try:
            code_payload = _extract(
                _call_module(
                    code_module,
                    None if case.kg_scope is None else case.kg_scope.model_dump(exclude_none=True),
                    parsed=parsed.model_dump(),
                ),
                "code_str",
            )
            code = CodeTarget.model_validate(code_payload)
            code_valid = True
        except Exception as exc:
            notes.append(f"code: {exc}")
            details.append(
                ContractValidationDetail(
                    case_id=case.case_id,
                    parse_valid=parse_valid,
                    code_valid=False,
                    pyro_success=False,
                    evaluation_valid=False,
                    notes=notes,
                )
            )
            continue

        pyro_result = execute_pyro_program(code.program, entrypoint=code.entrypoint, timeout_seconds=timeout_seconds)
        pyro_success = pyro_result.success
        if not pyro_success and pyro_result.error:
            notes.append(f"pyro: {pyro_result.error}")
        answer = pyro_result.answer if pyro_result.success else pyro_result.error

        try:
            evaluation_payload = _extract(
                _call_module(
                    eval_module,
                    None if case.kg_scope is None else case.kg_scope.model_dump(exclude_none=True),
                    query=case.customer_message,
                    answer=answer,
                ),
                "evaluation",
            )
            EvaluationTarget.model_validate(evaluation_payload)
            evaluation_valid = True
        except Exception as exc:
            notes.append(f"evaluate: {exc}")

        details.append(
            ContractValidationDetail(
                case_id=case.case_id,
                parse_valid=parse_valid,
                code_valid=code_valid,
                pyro_success=pyro_success,
                evaluation_valid=evaluation_valid,
                notes=notes,
            )
        )

    summary = ContractValidationSummary(
        total_cases=len(details),
        parse_valid_cases=sum(1 for detail in details if detail.parse_valid),
        code_valid_cases=sum(1 for detail in details if detail.code_valid),
        pyro_success_cases=sum(1 for detail in details if detail.pyro_success),
        evaluation_valid_cases=sum(1 for detail in details if detail.evaluation_valid),
    )
    return ContractValidationReport(module_source=module_source, summaries=summary, details=details)


def write_report(report: ContractValidationReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(mode="json"), handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate parse/code/evaluate contracts end to end.")
    parser.add_argument("--input", required=True, help="Path to customer support cases JSONL.")
    parser.add_argument("--module-source", choices=["baseline", "default"], default="baseline")
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--output", help="Optional path to write a JSON report.")
    parser.add_argument("--audit-path", help="Optional audit JSONL output path.")
    parser.add_argument("--audit-wwkg", action="store_true")
    args = parser.parse_args()

    report = validate_contracts(
        load_customer_support_cases(args.input),
        module_source=args.module_source,
        timeout_seconds=args.timeout_seconds,
    )
    if args.output:
        write_report(report, args.output)
    if args.audit_path:
        recorder = AuditRecorder.from_output_path(args.audit_path, wwkg_enabled=args.audit_wwkg)
        recorder.record_validation_evidence(
            report_id=event_report_id(report.model_dump(mode="json")),
            evidence_type="contract_validation",
            per_backend_scores={
                "contracts": {
                    "total_cases": report.summaries.total_cases,
                    "parse_valid_cases": report.summaries.parse_valid_cases,
                    "code_valid_cases": report.summaries.code_valid_cases,
                    "pyro_success_cases": report.summaries.pyro_success_cases,
                    "evaluation_valid_cases": report.summaries.evaluation_valid_cases,
                }
            },
            artifact_path=args.output,
        )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
