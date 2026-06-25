from __future__ import annotations

import json
from typing import Any

from msa_zria.data import CodeTarget, EvaluationTarget, ParseTarget
from msa_zria.dspy_modules import _call_llm, _extract_json_object


class ThinkingParseModule:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def prompt(self, text: str, kg_context: dict[str, Any] | None = None) -> str:
        context_suffix = ""
        if kg_context:
            context_suffix = f"\nKG context: {json.dumps(kg_context, sort_keys=True)}"
        return (
            "You are the specialist thinking branch. "
            "Reason carefully over domain rules, escalation boundaries, and evidence before deciding on the parse state. "
            "Internally verify that the extracted fields preserve the facts that drive downstream business logic. "
            "Return only valid JSON with keys task, device, issue, cause, severity. "
            "Set task to parse and use null when cause or severity is unknown.\n"
            f"Message: {text}{context_suffix}"
        )

    def __call__(self, text: str, kg_context: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = _extract_json_object(_call_llm(self.llm, self.prompt(text, kg_context=kg_context)))
        return {"parsed_result": ParseTarget.model_validate(payload).model_dump(mode="json")}


class ThinkingCodeGenModule:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def prompt(self, parsed: dict[str, Any], kg_context: dict[str, Any] | None = None) -> str:
        context_suffix = ""
        if kg_context:
            context_suffix = f"\nKG context: {json.dumps(kg_context, sort_keys=True)}"
        return (
            "You are the specialist thinking branch. "
            "Generate a Pyro-oriented reasoning program that encodes the decisive constraints, escalation boundaries, and edge conditions. "
            "Prefer robust logic over shallow pattern matching, and ensure the generated program can support verification. "
            "Return only valid JSON with keys task, language, framework, entrypoint, query_variable, required_statements, program. "
            "Set task to code."
            f"\nParsed state: {json.dumps(parsed, sort_keys=True)}{context_suffix}"
        )

    def __call__(self, parsed: dict[str, Any], kg_context: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = _extract_json_object(_call_llm(self.llm, self.prompt(parsed, kg_context=kg_context)))
        return {"code_str": CodeTarget.model_validate(payload).model_dump(mode="json")}


class ThinkingEvalModule:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def prompt(self, query: str, answer: Any, kg_context: dict[str, Any] | None = None) -> str:
        context_suffix = ""
        if kg_context:
            context_suffix = f"\nKG context: {json.dumps(kg_context, sort_keys=True)}"
        return (
            "You are the specialist thinking branch. "
            "Evaluate whether the proposed answer is defensible under the domain rules, verified claims, and escalation policy. "
            "Reject answers that are locally plausible but miss a decisive constraint. "
            "Return only valid JSON with keys task, verdict, resolved, should_escalate, explanation. "
            "Set task to evaluate."
            f"\nQuery: {query}\nAnswer: {json.dumps(answer, default=str, sort_keys=True)}{context_suffix}"
        )

    def __call__(self, query: str, answer: Any, kg_context: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = _extract_json_object(_call_llm(self.llm, self.prompt(query, answer, kg_context=kg_context)))
        return {"evaluation": EvaluationTarget.model_validate(payload).model_dump(mode="json")}
