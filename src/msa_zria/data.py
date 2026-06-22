from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

InputMode = Literal["triples", "text", "hybrid"]
TaskType = Literal["parse", "code", "evaluate"]
MessageRole = Literal["system", "user", "assistant"]
Severity = Literal["low", "medium", "high", "critical"]
EvaluationVerdict = Literal["resolved", "unresolved", "insufficient_information", "escalate"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str


class Triple(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    predicate: str
    object: str

    def as_line(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"

    def as_sentence(self) -> str:
        predicate = self.predicate.replace("_", " ")
        return f"{self.subject} {predicate} {self.object}."


class ParseTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["parse"] = "parse"
    device: str
    issue: str
    cause: str | None = None
    severity: Severity | None = None


class CodeTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["code"] = "code"
    language: str = "python"
    framework: str = "pyro"
    entrypoint: str = "run_inference"
    query_variable: str = "failure"
    required_statements: list[str] = Field(default_factory=list)
    program: str


class EvaluationTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["evaluate"] = "evaluate"
    verdict: EvaluationVerdict
    resolved: bool
    should_escalate: bool = False
    explanation: str


TargetContract = Annotated[
    ParseTarget | CodeTarget | EvaluationTarget,
    Field(discriminator="task"),
]


class DatasetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    task: TaskType
    input_mode: InputMode
    messages: list[Message]
    target: TargetContract
    metadata: dict[str, Any] = Field(default_factory=dict)


class StructuredTrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    example_id: str
    instruction: str
    target: TargetContract
    input_mode: InputMode = "hybrid"
    triples: list[Triple] = Field(default_factory=list)
    natural_language_context: list[str] = Field(default_factory=list)
    system_prompt: str = (
        "You are a customer support reasoning assistant. "
        "Use the provided facts and return only valid JSON that matches the requested schema."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def task(self) -> TaskType:
        return self.target.task

    def render_triples(self) -> str:
        return "\n".join(triple.as_line() for triple in self.triples)

    def render_text(self) -> str:
        if self.natural_language_context:
            return "\n".join(self.natural_language_context)
        return " ".join(triple.as_sentence() for triple in self.triples)

    def render_user_content(self) -> str:
        parts = [self.instruction.strip()]

        if self.input_mode == "triples":
            parts.append("Facts:\n" + self.render_triples())
        elif self.input_mode == "text":
            parts.append("Context:\n" + self.render_text())
        else:
            parts.append("Facts:\n" + self.render_triples())
            parts.append("Context:\n" + self.render_text())

        return "\n\n".join(part for part in parts if part.strip())

    def render_assistant_content(self) -> str:
        return json.dumps(self.target.model_dump(), ensure_ascii=True, sort_keys=True)

    def to_messages(self) -> list[Message]:
        return [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=self.render_user_content()),
            Message(role="assistant", content=self.render_assistant_content()),
        ]

    def to_record(self) -> DatasetRecord:
        return DatasetRecord(
            example_id=self.example_id,
            task=self.task,
            input_mode=self.input_mode,
            messages=self.to_messages(),
            target=self.target,
            metadata=self.metadata,
        )


class TrainingExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    instruction: str
    response: str
    triples: list[Triple] = Field(default_factory=list)
    natural_language_context: list[str] = Field(default_factory=list)
    system_prompt: str = (
        "You are a customer support reasoning assistant. "
        "Use the provided facts and return only the requested result."
    )

    def render_triples(self) -> str:
        return "\n".join(triple.as_line() for triple in self.triples)

    def render_text(self) -> str:
        if self.natural_language_context:
            return "\n".join(self.natural_language_context)
        return " ".join(triple.as_sentence() for triple in self.triples)

    def render_user_content(self, input_mode: InputMode) -> str:
        parts = [self.instruction.strip()]

        if input_mode == "triples":
            parts.append("Facts:\n" + self.render_triples())
        elif input_mode == "text":
            parts.append("Context:\n" + self.render_text())
        else:
            parts.append("Facts:\n" + self.render_triples())
            parts.append("Context:\n" + self.render_text())

        return "\n\n".join(part for part in parts if part.strip())

    def to_messages(self, input_mode: InputMode) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.render_user_content(input_mode)},
            {"role": "assistant", "content": self.response},
        ]


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: TaskType
    expected: TargetContract
    predicted: TargetContract


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    task: TaskType
    passed: bool
    score: float
    matched_fields: list[str] = Field(default_factory=list)
    mismatched_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    return " ".join(value.strip().lower().split())


def _evaluate_parse(case: EvaluationCase) -> EvaluationResult:
    expected = case.expected
    predicted = case.predicted
    assert isinstance(expected, ParseTarget)
    assert isinstance(predicted, ParseTarget)

    fields = ["device", "issue", "cause", "severity"]
    matched: list[str] = []
    mismatched: list[str] = []

    for field_name in fields:
        expected_value = getattr(expected, field_name)
        if expected_value is None:
            continue
        predicted_value = getattr(predicted, field_name)
        if _normalize_text(str(expected_value)) == _normalize_text(str(predicted_value)):
            matched.append(field_name)
        else:
            mismatched.append(field_name)

    total = len(matched) + len(mismatched)
    score = 1.0 if total == 0 else len(matched) / total
    return EvaluationResult(
        case_id=case.case_id,
        task=case.task,
        passed=not mismatched,
        score=score,
        matched_fields=matched,
        mismatched_fields=mismatched,
    )


def _evaluate_code(case: EvaluationCase) -> EvaluationResult:
    expected = case.expected
    predicted = case.predicted
    assert isinstance(expected, CodeTarget)
    assert isinstance(predicted, CodeTarget)

    matched: list[str] = []
    mismatched: list[str] = []
    notes: list[str] = []

    for field_name in ["language", "framework", "entrypoint", "query_variable"]:
        if _normalize_text(getattr(expected, field_name)) == _normalize_text(
            getattr(predicted, field_name)
        ):
            matched.append(field_name)
        else:
            mismatched.append(field_name)

    for statement in expected.required_statements:
        normalized_statement = _normalize_text(statement)
        normalized_program = _normalize_text(predicted.program) or ""
        if normalized_statement and normalized_statement in normalized_program:
            matched.append(f"program:{statement}")
        else:
            mismatched.append(f"program:{statement}")

    if not predicted.program.strip():
        mismatched.append("program_non_empty")
        notes.append("Predicted program was empty.")

    total = len(matched) + len(mismatched)
    score = 1.0 if total == 0 else len(matched) / total
    return EvaluationResult(
        case_id=case.case_id,
        task=case.task,
        passed=not mismatched,
        score=score,
        matched_fields=matched,
        mismatched_fields=mismatched,
        notes=notes,
    )


def _evaluate_final(case: EvaluationCase) -> EvaluationResult:
    expected = case.expected
    predicted = case.predicted
    assert isinstance(expected, EvaluationTarget)
    assert isinstance(predicted, EvaluationTarget)

    matched: list[str] = []
    mismatched: list[str] = []

    for field_name in ["verdict", "resolved", "should_escalate"]:
        if getattr(expected, field_name) == getattr(predicted, field_name):
            matched.append(field_name)
        else:
            mismatched.append(field_name)

    total = len(matched) + len(mismatched)
    score = 1.0 if total == 0 else len(matched) / total
    return EvaluationResult(
        case_id=case.case_id,
        task=case.task,
        passed=not mismatched,
        score=score,
        matched_fields=matched,
        mismatched_fields=mismatched,
    )


def evaluate_case(case: EvaluationCase) -> EvaluationResult:
    if case.task != case.expected.task or case.task != case.predicted.task:
        raise ValueError("Evaluation case task must match both expected and predicted task.")

    if case.task == "parse":
        return _evaluate_parse(case)
    if case.task == "code":
        return _evaluate_code(case)
    return _evaluate_final(case)


def write_jsonl_records(records: list[DatasetRecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")
