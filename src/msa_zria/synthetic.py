from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from msa_zria.data import DatasetRecord, InputMode, StructuredTrainingExample, write_jsonl_records
from msa_zria.ingest import CustomerSupportCase, load_customer_support_cases
from msa_zria.thinking import ThinkingCase, load_thinking_cases


class SyntheticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_type: str
    input_modes: list[InputMode] = Field(default_factory=lambda: ["hybrid"])
    include_paraphrase: bool = True
    include_qa: bool = True


def build_synthetic_records(
    input_path: str | Path,
    *,
    case_type: str,
    input_modes: list[InputMode] | None = None,
    include_paraphrase: bool = True,
    include_qa: bool = True,
) -> list[DatasetRecord]:
    modes = input_modes or ["hybrid"]
    records: list[DatasetRecord] = []
    if case_type == "thinking":
        for case in load_thinking_cases(input_path):
            records.extend(
                _records_for_case(
                    case,
                    modes=modes,
                    include_paraphrase=include_paraphrase,
                    include_qa=include_qa,
                )
            )
        return records
    if case_type == "support":
        for case in load_customer_support_cases(input_path):
            records.extend(
                _records_for_case(
                    case,
                    modes=modes,
                    include_paraphrase=include_paraphrase,
                    include_qa=include_qa,
                )
            )
        return records
    raise ValueError(f"Unsupported case_type '{case_type}'. Expected support or thinking.")


def write_synthetic_records(
    input_path: str | Path,
    output_path: str | Path,
    *,
    case_type: str,
    input_modes: list[InputMode] | None = None,
    include_paraphrase: bool = True,
    include_qa: bool = True,
) -> int:
    records = build_synthetic_records(
        input_path,
        case_type=case_type,
        input_modes=input_modes,
        include_paraphrase=include_paraphrase,
        include_qa=include_qa,
    )
    write_jsonl_records(records, output_path)
    return len(records)


def _records_for_case(
    case: CustomerSupportCase | ThinkingCase,
    *,
    modes: list[InputMode],
    include_paraphrase: bool,
    include_qa: bool,
) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    augmentations: list[tuple[str, list[str]]] = []
    if include_paraphrase:
        augmentations.append(("paraphrase", [_paraphrase_case(case)]))
    if include_qa:
        qa_block = _coverage_qa(case)
        if qa_block is not None:
            augmentations.append(("qa", [qa_block]))
    for variant, extra_context in augmentations:
        for input_mode in modes:
            for task_name, target, instruction in _task_triplets(case):
                metadata = dict(getattr(case, "metadata", {}))
                metadata.update(
                    {
                        "case_id": case.case_id,
                        "task": task_name,
                        "input_mode": input_mode,
                        "augmentation_family": "clara_scp",
                        "augmentation_variant": variant,
                    }
                )
                records.append(
                    StructuredTrainingExample(
                        example_id=f"{case.case_id}-{task_name}-{input_mode}-{variant}",
                        instruction=instruction,
                        target=target,
                        input_mode=input_mode,
                        triples=list(case.triples),
                        natural_language_context=[case.customer_message, *list(getattr(case, "context", [])), *extra_context],
                        system_prompt=(
                            "You are a customer support reasoning assistant. "
                            "Use the salient facts and return only valid JSON that matches the requested schema."
                        ),
                        metadata=metadata,
                    ).to_record()
                )
    return records


def _task_triplets(case: CustomerSupportCase | ThinkingCase) -> list[tuple[str, Any, str]]:
    parse_instruction = (
        "Extract the parse contract from the message and salient facts. "
        "Preserve the fields that change downstream reasoning."
    )
    code_instruction = (
        "Generate the code contract using the message and salient facts. "
        "Keep only the logic needed for a defensible Pyro program."
    )
    evaluate_instruction = (
        "Generate the evaluation contract using the message and salient facts. "
        "Prefer the decision that is supported by the evidence."
    )
    return [
        ("parse", case.parse_target, parse_instruction),
        ("code", case.code_target, code_instruction),
        ("evaluate", case.evaluation_target, evaluate_instruction),
    ]


def _paraphrase_case(case: CustomerSupportCase | ThinkingCase) -> str:
    parse_target = case.parse_target
    parts = [
        f"The customer is reporting an issue with {parse_target.device}.",
        f"The main problem is {parse_target.issue}.",
    ]
    if parse_target.cause:
        parts.append(f"The likely cause is {parse_target.cause}.")
    if parse_target.severity:
        parts.append(f"The severity should be treated as {parse_target.severity}.")
    if isinstance(case, ThinkingCase):
        parts.append(f"Reasoning goal: {case.reasoning_goal}.")
    return "Paraphrase:\n" + " ".join(parts)


def _coverage_qa(case: CustomerSupportCase | ThinkingCase) -> str | None:
    rows: list[tuple[str, str | bool | None]] = [
        ("What device is involved?", case.parse_target.device),
        ("What issue is reported?", case.parse_target.issue),
        ("What is the cause?", case.parse_target.cause),
        ("What severity applies?", case.parse_target.severity),
        ("Should the answer be marked resolved?", case.evaluation_target.resolved),
        ("Should the answer escalate?", case.evaluation_target.should_escalate),
    ]
    lines: list[str] = []
    covered = 0
    for question, answer in rows:
        if answer is None:
            continue
        covered += 1
        lines.append(f"Q: {question}")
        lines.append(f"A: {json.dumps(answer, ensure_ascii=True)}")
    if covered < 4:
        return None
    return "Salient QA:\n" + "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SCP-style synthetic training records for msa_zria.")
    parser.add_argument("--input", required=True, help="Path to source customer-support or thinking cases JSONL.")
    parser.add_argument("--output", required=True, help="Path to output synthetic DatasetRecord JSONL.")
    parser.add_argument("--case-type", required=True, choices=["support", "thinking"])
    parser.add_argument(
        "--input-mode",
        action="append",
        choices=["triples", "text", "hybrid"],
        dest="input_modes",
        help="One or more input modes to emit. Defaults to hybrid.",
    )
    parser.add_argument("--skip-paraphrase", action="store_true", help="Disable paraphrase-style records.")
    parser.add_argument("--skip-qa", action="store_true", help="Disable salient QA-style records.")
    args = parser.parse_args()
    count = write_synthetic_records(
        args.input,
        args.output,
        case_type=args.case_type,
        input_modes=args.input_modes,
        include_paraphrase=not args.skip_paraphrase,
        include_qa=not args.skip_qa,
    )
    print(json.dumps({"output": args.output, "records_written": count}, ensure_ascii=True, sort_keys=True))
