from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from msa_zria.audit import AuditRecorder, sha256_file
from msa_zria.config import KGScope
from msa_zria.data import (
    CodeTarget,
    DatasetRecord,
    EvaluationTarget,
    InputMode,
    ParseTarget,
    StructuredTrainingExample,
    Triple,
    write_jsonl_records,
)

ThinkingDifficulty = Literal["standard", "hard", "expert"]


class ThinkingRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    condition: str
    outcome: str
    priority: int = 0

    def as_text(self) -> str:
        return f"{self.rule_id} (priority {self.priority}): if {self.condition}, then {self.outcome}"


class ThinkingClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str
    evidence: list[str] = Field(default_factory=list)
    verified: bool = True

    def as_text(self) -> str:
        evidence_suffix = ""
        if self.evidence:
            evidence_suffix = f" Evidence: {'; '.join(self.evidence)}."
        verification = "verified" if self.verified else "unverified"
        return f"{self.claim_id} [{verification}]: {self.text}.{evidence_suffix}"


class ThinkingCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    customer_message: str
    candidate_answer: str
    triples: list[Triple] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    kg_scope: KGScope | None = None
    parse_target: ParseTarget
    code_target: CodeTarget
    evaluation_target: EvaluationTarget
    reasoning_goal: str
    rules: list[ThinkingRule] = Field(default_factory=list)
    claims: list[ThinkingClaim] = Field(default_factory=list)
    reasoning_trace: list[str] = Field(default_factory=list)
    verification_checks: list[str] = Field(default_factory=list)
    difficulty: ThinkingDifficulty = "hard"
    metadata: dict[str, Any] = Field(default_factory=dict)


def _merge_kg_scope(case_scope: KGScope | None, override_scope: KGScope | None) -> KGScope | None:
    if case_scope is None:
        return override_scope
    if override_scope is None:
        return case_scope
    return KGScope(
        workspace=override_scope.workspace or case_scope.workspace,
        branch=override_scope.branch or case_scope.branch,
        commit=override_scope.commit or case_scope.commit,
        as_of=override_scope.as_of or case_scope.as_of,
    )


def _base_metadata(
    case: ThinkingCase,
    task: str,
    input_mode: InputMode,
    kg_scope: KGScope | None = None,
) -> dict[str, Any]:
    metadata = dict(case.metadata)
    metadata["case_id"] = case.case_id
    metadata["task"] = task
    metadata["input_mode"] = input_mode
    metadata["reasoning_branch"] = "thinking"
    metadata["reasoning_style"] = "specialist"
    metadata["difficulty"] = case.difficulty
    resolved_scope = _merge_kg_scope(case.kg_scope, kg_scope)
    if resolved_scope is not None:
        metadata.update(resolved_scope.to_metadata())
    return metadata


def _render_rules(case: ThinkingCase) -> str:
    if not case.rules:
        return "None."
    return "\n".join(f"- {rule.as_text()}" for rule in case.rules)


def _render_claims(case: ThinkingCase) -> str:
    if not case.claims:
        return "None."
    return "\n".join(f"- {claim.as_text()}" for claim in case.claims)


def _render_trace(case: ThinkingCase) -> str:
    if not case.reasoning_trace:
        return "None."
    return "\n".join(f"{index + 1}. {step}" for index, step in enumerate(case.reasoning_trace))


def _render_checks(case: ThinkingCase) -> str:
    if not case.verification_checks:
        return "None."
    return "\n".join(f"- {check}" for check in case.verification_checks)


def _specialist_context(case: ThinkingCase) -> list[str]:
    return [
        f"Reasoning goal: {case.reasoning_goal}",
        "Rules:\n" + _render_rules(case),
        "Claims:\n" + _render_claims(case),
        "Exemplar reasoning trace:\n" + _render_trace(case),
        "Verification checks:\n" + _render_checks(case),
    ]


def _specialist_system_prompt() -> str:
    return (
        "You are the specialist thinking branch for deep-reasoning customer support decisions. "
        "Carefully reason over rules, evidence, and verification checks before deciding on the final answer. "
        "Return only valid JSON that matches the requested schema."
    )


def _parse_instruction(case: ThinkingCase) -> str:
    return (
        "Reason carefully over the message, rules, and verified claims to extract the correct parse state. "
        "Prefer the most defensible interpretation when multiple fields could fit, and preserve escalation-critical detail. "
        "Return only valid JSON for the parse contract.\n\n"
        f"Reasoning goal: {case.reasoning_goal}\n"
        f"Rules:\n{_render_rules(case)}\n\n"
        f"Claims:\n{_render_claims(case)}"
    )


def _code_instruction(case: ThinkingCase) -> str:
    parse_fields = case.parse_target.model_dump(exclude={"task"})
    return (
        "Synthesize a Pyro reasoning program that captures the domain rules, edge conditions, and escalation boundaries. "
        "Use the parsed state, verified claims, and checks to produce a robust decision program. "
        "Return only valid JSON for the code contract.\n\n"
        f"Parsed state: {parse_fields}\n"
        f"Rules:\n{_render_rules(case)}\n\n"
        f"Verification checks:\n{_render_checks(case)}"
    )


def _evaluation_instruction(case: ThinkingCase) -> str:
    return (
        "Judge whether the proposed answer is correct after reasoning through the policy rules, claims, and verification checks. "
        "Reject shallow answers that ignore a decisive rule or verified claim. "
        "Return only valid JSON for the evaluation contract.\n\n"
        f"Customer message: {case.customer_message}\n"
        f"Proposed answer: {case.candidate_answer}\n"
        f"Rules:\n{_render_rules(case)}\n\n"
        f"Verification checks:\n{_render_checks(case)}"
    )


def build_thinking_records_for_case(
    case: ThinkingCase,
    input_modes: list[InputMode] | None = None,
    kg_scope: KGScope | None = None,
) -> list[DatasetRecord]:
    modes = input_modes or ["hybrid"]
    specialist_context = [case.customer_message, *case.context, *_specialist_context(case)]
    records: list[DatasetRecord] = []

    for input_mode in modes:
        records.append(
            StructuredTrainingExample(
                example_id=f"{case.case_id}-parse-{input_mode}-thinking",
                instruction=_parse_instruction(case),
                target=case.parse_target,
                input_mode=input_mode,
                triples=case.triples,
                natural_language_context=specialist_context,
                system_prompt=_specialist_system_prompt(),
                metadata=_base_metadata(case, "parse", input_mode, kg_scope=kg_scope),
            ).to_record()
        )
        records.append(
            StructuredTrainingExample(
                example_id=f"{case.case_id}-code-{input_mode}-thinking",
                instruction=_code_instruction(case),
                target=case.code_target,
                input_mode=input_mode,
                triples=case.triples,
                natural_language_context=specialist_context,
                system_prompt=_specialist_system_prompt(),
                metadata=_base_metadata(case, "code", input_mode, kg_scope=kg_scope),
            ).to_record()
        )
        records.append(
            StructuredTrainingExample(
                example_id=f"{case.case_id}-evaluate-{input_mode}-thinking",
                instruction=_evaluation_instruction(case),
                target=case.evaluation_target,
                input_mode=input_mode,
                triples=case.triples,
                natural_language_context=specialist_context,
                system_prompt=_specialist_system_prompt(),
                metadata=_base_metadata(case, "evaluate", input_mode, kg_scope=kg_scope),
            ).to_record()
        )

    return records


def load_thinking_cases(path: str | Path) -> list[ThinkingCase]:
    input_path = Path(path)
    cases: list[ThinkingCase] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(ThinkingCase.model_validate_json(line))
    return cases


def ingest_thinking_cases(
    input_path: str | Path,
    output_path: str | Path,
    input_modes: list[InputMode] | None = None,
    kg_scope: KGScope | None = None,
    audit_recorder: AuditRecorder | None = None,
) -> int:
    records: list[DatasetRecord] = []
    input_file_hash = sha256_file(input_path)
    for case in load_thinking_cases(input_path):
        case_records = build_thinking_records_for_case(case, input_modes=input_modes, kg_scope=kg_scope)
        records.extend(case_records)
        if audit_recorder is not None:
            audit_recorder.record_dataset_lineage(
                source_case_id=case.case_id,
                input_file_hash=input_file_hash,
                output_record_ids=[record.example_id for record in case_records],
                kg_scope=_merge_kg_scope(case.kg_scope, kg_scope),
            )
    write_jsonl_records(records, output_path)
    return len(records)


def _cli_kg_scope(args: argparse.Namespace) -> KGScope | None:
    if not any([args.workspace, args.branch, args.commit, args.as_of]):
        return None
    return KGScope(
        workspace=args.workspace,
        branch=args.branch,
        commit=args.commit,
        as_of=args.as_of,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert specialist thinking cases into canonical msa_zria JSONL records.")
    parser.add_argument("--input", required=True, help="Path to source specialist thinking cases JSONL.")
    parser.add_argument("--output", required=True, help="Path to canonical specialist DatasetRecord JSONL.")
    parser.add_argument(
        "--input-mode",
        action="append",
        choices=["triples", "text", "hybrid"],
        dest="input_modes",
        help="One or more input modes to emit. Defaults to hybrid.",
    )
    parser.add_argument("--workspace", help="Optional WWKG workspace to attach to record metadata.")
    parser.add_argument("--branch", help="Optional WWKG branch to attach to record metadata.")
    parser.add_argument("--commit", help="Optional WWKG commit to attach to record metadata.")
    parser.add_argument("--as-of", dest="as_of", help="Optional WWKG as-of timestamp to attach to record metadata.")
    parser.add_argument("--audit-path", help="Optional audit JSONL output path.")
    parser.add_argument("--audit-wwkg", action="store_true", help="Mirror audit events to WWKG when contextkg is available.")
    args = parser.parse_args()
    audit_recorder = None
    if args.audit_path:
        audit_recorder = AuditRecorder.from_output_path(
            args.audit_path,
            wwkg_enabled=args.audit_wwkg,
            default_scope=_cli_kg_scope(args),
        )
    count = ingest_thinking_cases(
        args.input,
        args.output,
        input_modes=args.input_modes,
        kg_scope=_cli_kg_scope(args),
        audit_recorder=audit_recorder,
    )
    print(f"Wrote {count} records to {args.output}")


if __name__ == "__main__":
    main()
