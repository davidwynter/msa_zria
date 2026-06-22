from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from msa_zria.audit import AuditRecorder, event_report_id
from msa_zria.config import KGScope, load_experiment_config
from msa_zria.data import EvaluationCase, EvaluationResult, EvaluationTarget, evaluate_case
from msa_zria.reasoning_pipeline import ReasoningPipeline
from msa_zria.zria_adapter import BaseZRIAAdapter, ConfiguredZRIAAdapter


class AblationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    query: str
    kg_scope: KGScope | None = None
    expected: EvaluationTarget


class AblationModeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: str
    total_cases: int
    passed_cases: int
    accuracy: float
    average_score: float


class AblationCaseDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    mode: str
    kg_scope: KGScope | None = None
    predicted: EvaluationTarget
    evaluation: EvaluationResult


class AblationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kg_metadata: dict[str, str] = Field(default_factory=dict)
    summaries: list[AblationModeSummary]
    details: list[AblationCaseDetail]


class HeuristicParseModule:
    def __call__(self, text: str) -> dict[str, dict[str, Any]]:
        normalized = text.lower()
        device = "unknown_device"
        if "router" in normalized:
            device = "router"
        elif "monitor" in normalized:
            device = "monitor"
        elif "printer" in normalized:
            device = "printer"

        issue = "general_support"
        if "overheat" in normalized or "smoke" in normalized:
            issue = "overheating"
        elif "refund" in normalized:
            issue = "refund_request"
        elif "jam" in normalized:
            issue = "paper_jam"

        cause = None
        if "storm" in normalized:
            cause = "weather_event"
        elif "opened" in normalized:
            cause = "opened_box"

        severity = "low"
        if "smoke" in normalized or "fire" in normalized:
            severity = "critical"
        elif "overheat" in normalized:
            severity = "high"
        elif "failed" in normalized or "jam" in normalized:
            severity = "medium"

        return {
            "parsed_result": {
                "task": "parse",
                "device": device,
                "issue": issue,
                "cause": cause,
                "severity": severity,
            }
        }


class HeuristicCodeModule:
    def __call__(self, parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
        issue = parsed.get("issue", "general_support")
        severity = parsed.get("severity", "low")
        program = f"""def run_inference():
    issue = {issue!r}
    severity = {severity!r}
    if severity == "critical":
        return {{"recommended_action": "escalate", "resolved": False, "should_escalate": True}}
    if issue == "refund_request":
        return {{"recommended_action": "approve_refund", "resolved": True, "should_escalate": False}}
    if issue == "overheating":
        return {{"recommended_action": "restart_and_escalate_if_repeated", "resolved": False, "should_escalate": True}}
    return {{"recommended_action": "collect_more_information", "resolved": False, "should_escalate": False}}
"""
        return {
            "code_str": {
                "task": "code",
                "language": "python",
                "framework": "pyro",
                "entrypoint": "run_inference",
                "query_variable": "failure",
                "required_statements": ["def run_inference"],
                "program": program,
            }
        }


class HeuristicEvalModule:
    def __call__(self, query: str, answer: Any) -> dict[str, dict[str, Any]]:
        if isinstance(answer, dict):
            should_escalate = bool(answer.get("should_escalate"))
            resolved = bool(answer.get("resolved"))
        else:
            normalized = str(answer).lower()
            should_escalate = "escalate" in normalized or "critical" in normalized
            resolved = "approve_refund" in normalized

        if should_escalate:
            verdict = "escalate"
        elif resolved:
            verdict = "resolved"
        else:
            verdict = "insufficient_information"

        return {
            "evaluation": {
                "task": "evaluate",
                "verdict": verdict,
                "resolved": resolved,
                "should_escalate": should_escalate,
                "explanation": str(answer),
            }
        }


def _merge_hybrid(pyro_eval: EvaluationTarget, zria_eval: EvaluationTarget) -> EvaluationTarget:
    if pyro_eval.resolved:
        return pyro_eval
    if zria_eval.should_escalate or pyro_eval.should_escalate:
        return EvaluationTarget(
            verdict="escalate",
            resolved=False,
            should_escalate=True,
            explanation="Hybrid mode escalated because at least one reasoning path recommended escalation.",
        )
    return zria_eval


def _score_prediction(
    case: AblationCase,
    mode: str,
    predicted: EvaluationTarget,
    kg_scope: KGScope | None = None,
) -> AblationCaseDetail:
    evaluation = evaluate_case(
        EvaluationCase(
            case_id=case.case_id,
            task="evaluate",
            expected=case.expected,
            predicted=predicted,
        )
    )
    return AblationCaseDetail(
        case_id=case.case_id,
        mode=mode,
        kg_scope=kg_scope,
        predicted=predicted,
        evaluation=evaluation,
    )


def run_ablation(
    cases: list[AblationCase],
    pipeline: ReasoningPipeline,
    zria_adapter: BaseZRIAAdapter,
    kg_scope: KGScope | None = None,
) -> AblationReport:
    details: list[AblationCaseDetail] = []

    for case in cases:
        case_scope = case.kg_scope or kg_scope
        pipeline_result = pipeline.run(case.query, kg_scope=case_scope)
        pyro_detail = _score_prediction(
            case,
            "pyro_only",
            pipeline_result.evaluation,
            kg_scope=case_scope,
        )
        details.append(pyro_detail)

        zria_prediction = zria_adapter.predict(
            case.query,
            pipeline_result.parsed,
            kg_scope=case_scope,
        )
        zria_detail = _score_prediction(case, "zria_only", zria_prediction, kg_scope=case_scope)
        details.append(zria_detail)

        hybrid_prediction = _merge_hybrid(pipeline_result.evaluation, zria_prediction)
        hybrid_detail = _score_prediction(case, "hybrid", hybrid_prediction, kg_scope=case_scope)
        details.append(hybrid_detail)

    summaries: list[AblationModeSummary] = []
    for mode in ["pyro_only", "zria_only", "hybrid"]:
        mode_details = [detail for detail in details if detail.mode == mode]
        passed_cases = sum(1 for detail in mode_details if detail.evaluation.passed)
        average_score = 0.0
        if mode_details:
            average_score = sum(detail.evaluation.score for detail in mode_details) / len(mode_details)
        summaries.append(
            AblationModeSummary(
                mode=mode,
                total_cases=len(mode_details),
                passed_cases=passed_cases,
                accuracy=0.0 if not mode_details else passed_cases / len(mode_details),
                average_score=average_score,
            )
        )

    return AblationReport(
        kg_metadata={} if kg_scope is None else kg_scope.to_metadata(),
        summaries=summaries,
        details=details,
    )


def load_ablation_cases(path: str | Path) -> list[AblationCase]:
    cases_path = Path(path)
    cases: list[AblationCase] = []
    with cases_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(AblationCase.model_validate_json(line))
    return cases


def build_baseline_pipeline() -> ReasoningPipeline:
    return ReasoningPipeline(
        parse_module=HeuristicParseModule(),
        code_module=HeuristicCodeModule(),
        eval_module=HeuristicEvalModule(),
    )


def write_ablation_report(report: AblationReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report.model_dump(), handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the msa_zria ablation harness.")
    parser.add_argument("--config", required=True, help="Path to ExperimentConfig YAML.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    cases = load_ablation_cases(config.ablation.cases_path)
    pipeline = build_baseline_pipeline()
    report = run_ablation(
        cases,
        pipeline,
        ConfiguredZRIAAdapter.from_config(config.zria),
        kg_scope=KGScope(
            workspace=config.kg.workspace,
            branch=config.kg.branch,
            commit=config.kg.commit,
            as_of=config.kg.as_of,
        ),
    )
    write_ablation_report(report, config.ablation.output_path)
    audit_recorder = AuditRecorder.from_experiment(config)
    if audit_recorder is not None:
        audit_recorder.record_validation_evidence(
            report_id=event_report_id(report.model_dump(mode="json")),
            evidence_type="ablation_report",
            per_backend_scores={
                summary.mode: {
                    "total_cases": summary.total_cases,
                    "passed_cases": summary.passed_cases,
                    "accuracy": summary.accuracy,
                    "average_score": summary.average_score,
                }
                for summary in report.summaries
            },
            artifact_path=config.ablation.output_path,
            kg_scope=KGScope(
                workspace=config.kg.workspace,
                branch=config.kg.branch,
                commit=config.kg.commit,
                as_of=config.kg.as_of,
            ),
        )
    print(f"Wrote ablation report to {config.ablation.output_path}")


if __name__ == "__main__":
    main()
