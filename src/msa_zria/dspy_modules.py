from __future__ import annotations

import json
from typing import Any

from msa_zria.data import CodeTarget, EvaluationTarget, ParseTarget


def _call_llm(llm: Any, prompt: str) -> str:
    candidates = [
        lambda: llm(prompt),
        lambda: llm(prompt=prompt),
    ]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return _coerce_llm_text(candidate())
        except TypeError as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM did not return a usable response.")


def _coerce_llm_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, list) and response:
        return _coerce_llm_text(response[0])
    if isinstance(response, dict):
        for key in ("text", "content", "response"):
            if key in response:
                return _coerce_llm_text(response[key])
        if "choices" in response and response["choices"]:
            choice = response["choices"][0]
            if isinstance(choice, dict):
                if "text" in choice:
                    return _coerce_llm_text(choice["text"])
                if "message" in choice:
                    return _coerce_llm_text(choice["message"])
    if hasattr(response, "text"):
        return _coerce_llm_text(getattr(response, "text"))
    if hasattr(response, "content"):
        return _coerce_llm_text(getattr(response, "content"))
    raise RuntimeError(f"Unsupported LLM response type: {type(response).__name__}")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if "\n" in stripped:
            stripped = stripped.split("\n", 1)[1]
        if "\n```" in stripped:
            stripped = stripped.rsplit("\n```", 1)[0]

    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response did not contain a JSON object.")
        payload = json.loads(stripped[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("LLM response must decode to a JSON object.")
    return payload


class ParseModule:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def prompt(
        self,
        text: str,
        kg_context: dict[str, Any] | None = None,
        evidence_context: str | None = None,
    ) -> str:
        context_suffix = ""
        if kg_context:
            context_suffix = f"\nKG context: {json.dumps(kg_context, sort_keys=True)}"
        evidence_suffix = ""
        if evidence_context:
            evidence_suffix = f"\n{evidence_context}"
        return (
            "Extract the device, issue, cause, and severity from the customer message. "
            "Return only valid JSON with keys task, device, issue, cause, severity. "
            "Set task to parse and use null when cause or severity is unknown.\n"
            f"Message: {text}{context_suffix}{evidence_suffix}"
        )

    def __call__(
        self,
        text: str,
        kg_context: dict[str, Any] | None = None,
        evidence_context: str | None = None,
    ) -> dict[str, Any]:
        payload = _extract_json_object(
            _call_llm(self.llm, self.prompt(text, kg_context=kg_context, evidence_context=evidence_context))
        )
        return {"parsed_result": ParseTarget.model_validate(payload).model_dump(mode="json")}


class CodeGenModule:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def prompt(
        self,
        parsed: dict[str, Any],
        kg_context: dict[str, Any] | None = None,
        evidence_context: str | None = None,
    ) -> str:
        context_suffix = ""
        if kg_context:
            context_suffix = f"\nKG context: {json.dumps(kg_context, sort_keys=True)}"
        evidence_suffix = ""
        if evidence_context:
            evidence_suffix = f"\n{evidence_context}"
        return (
            "Generate a Pyro-oriented reasoning program from the parsed customer state. "
            "Return only valid JSON with keys task, language, framework, entrypoint, "
            "query_variable, required_statements, program. Set task to code."
            f"\nParsed state: {json.dumps(parsed, sort_keys=True)}{context_suffix}{evidence_suffix}"
        )

    def __call__(
        self,
        parsed: dict[str, Any],
        kg_context: dict[str, Any] | None = None,
        evidence_context: str | None = None,
    ) -> dict[str, Any]:
        payload = _extract_json_object(
            _call_llm(self.llm, self.prompt(parsed, kg_context=kg_context, evidence_context=evidence_context))
        )
        return {"code_str": CodeTarget.model_validate(payload).model_dump(mode="json")}


class EvalModule:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def prompt(
        self,
        query: str,
        answer: Any,
        kg_context: dict[str, Any] | None = None,
        evidence_context: str | None = None,
    ) -> str:
        context_suffix = ""
        if kg_context:
            context_suffix = f"\nKG context: {json.dumps(kg_context, sort_keys=True)}"
        evidence_suffix = ""
        if evidence_context:
            evidence_suffix = f"\n{evidence_context}"
        return (
            "Evaluate whether the proposed answer addresses the customer request. "
            "Return only valid JSON with keys task, verdict, resolved, should_escalate, explanation. "
            "Set task to evaluate."
            f"\nQuery: {query}\nAnswer: {json.dumps(answer, default=str, sort_keys=True)}{context_suffix}{evidence_suffix}"
        )

    def __call__(
        self,
        query: str,
        answer: Any,
        kg_context: dict[str, Any] | None = None,
        evidence_context: str | None = None,
    ) -> dict[str, Any]:
        payload = _extract_json_object(
            _call_llm(
                self.llm,
                self.prompt(query, answer, kg_context=kg_context, evidence_context=evidence_context),
            )
        )
        return {"evaluation": EvaluationTarget.model_validate(payload).model_dump(mode="json")}
