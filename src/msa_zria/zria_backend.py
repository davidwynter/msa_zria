from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib import error, request

from pydantic import BaseModel, ConfigDict, Field

from msa_zria.config import KGScope, ZRIAConfig
from msa_zria.data import EvaluationTarget, ParseTarget
from msa_zria.zria import (
    ZRIALearnedArtifact,
    load_artifact,
    predict_with_model,
)


class BaseZRIABackend(ABC):
    @abstractmethod
    def predict_with_trace(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> "ZRIAPredictionTrace":
        raise NotImplementedError

    def predict(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> EvaluationTarget:
        return self.predict_with_trace(query, parsed=parsed, kg_scope=kg_scope).prediction


class ZRIAPredictionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    configured_backend: str
    effective_backend: str
    fallback_fired: bool = False
    fallback_reason: str | None = None
    control_event_type: str | None = None
    confidence: float | None = None
    prediction: EvaluationTarget


class ZRIARule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    keywords: list[str] = Field(default_factory=list)
    device: str | None = None
    issue: str | None = None
    severity: str | None = None
    workspace: str | None = None
    branch: str | None = None
    commit: str | None = None
    outcome: EvaluationTarget


class ZRIABundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1"
    rules: list[ZRIARule]
    default_outcome: EvaluationTarget


class RuleBasedZRIABackend(BaseZRIABackend):
    def __init__(self, bundle: ZRIABundle) -> None:
        self.bundle = bundle

    @classmethod
    def from_path(cls, path: str | Path) -> "RuleBasedZRIABackend":
        bundle_path = Path(path)
        if not bundle_path.is_absolute() and not bundle_path.exists():
            bundle_path = Path(__file__).resolve().parents[2] / bundle_path
        with bundle_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(ZRIABundle.model_validate(payload))

    def predict_with_trace(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> ZRIAPredictionTrace:
        normalized_query = query.lower()
        best_rule: ZRIARule | None = None
        best_score = -1

        for rule in self.bundle.rules:
            score = self._score_rule(rule, normalized_query, parsed, kg_scope)
            if score > best_score:
                best_score = score
                best_rule = rule

        if best_rule is None or best_score <= 0:
            prediction = self.bundle.default_outcome
        else:
            prediction = best_rule.outcome
        return ZRIAPredictionTrace(
            configured_backend="rules",
            effective_backend="rules",
            prediction=prediction,
        )

    def _score_rule(
        self,
        rule: ZRIARule,
        normalized_query: str,
        parsed: ParseTarget | None,
        kg_scope: KGScope | None,
    ) -> int:
        score = 0

        if rule.workspace and (kg_scope is None or kg_scope.workspace != rule.workspace):
            return -1
        if rule.branch and (kg_scope is None or kg_scope.branch != rule.branch):
            return -1
        if rule.commit and (kg_scope is None or kg_scope.commit != rule.commit):
            return -1

        if rule.workspace:
            score += 4
        if rule.branch:
            score += 4
        if rule.commit:
            score += 4

        if rule.device:
            if parsed is None or parsed.device.lower() != rule.device.lower():
                return -1
            score += 3

        if rule.issue:
            if parsed is None or parsed.issue.lower() != rule.issue.lower():
                return -1
            score += 3

        if rule.severity:
            if parsed is None or parsed.severity is None or parsed.severity.lower() != rule.severity.lower():
                return -1
            score += 2

        if rule.keywords:
            if not all(keyword.lower() in normalized_query for keyword in rule.keywords):
                return -1
            score += len(rule.keywords)

        return score


class LearnedZRIABackend(BaseZRIABackend):
    def __init__(
        self,
        artifact: ZRIALearnedArtifact,
        model: Any,
        *,
        confidence_threshold: float = 0.6,
        fallback_backend: BaseZRIABackend | None = None,
    ) -> None:
        self.artifact = artifact
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.fallback_backend = fallback_backend

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        confidence_threshold: float = 0.6,
        fallback_backend: BaseZRIABackend | None = None,
    ) -> "LearnedZRIABackend":
        artifact, model = load_artifact(path)
        return cls(
            artifact,
            model,
            confidence_threshold=confidence_threshold,
            fallback_backend=fallback_backend,
        )

    def predict_with_trace(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> ZRIAPredictionTrace:
        if parsed is None:
            return self._fallback(query, parsed, kg_scope, "Parsed input was missing for learned ZRIA backend.")

        try:
            predicted, confidence = predict_with_model(
                self.artifact,
                self.model,
                query,
                parsed,
                kg_scope,
            )
        except Exception as exc:
            return self._fallback(query, parsed, kg_scope, f"Learned ZRIA backend failed: {exc}")

        if confidence < self.confidence_threshold:
            return self._fallback(
                query,
                parsed,
                kg_scope,
                f"Learned ZRIA backend confidence {confidence:.3f} was below threshold {self.confidence_threshold:.3f}.",
                confidence=confidence,
            )
        return ZRIAPredictionTrace(
            configured_backend="learned",
            effective_backend="learned",
            confidence=confidence,
            prediction=predicted,
        )

    def _fallback(
        self,
        query: str,
        parsed: ParseTarget | None,
        kg_scope: KGScope | None,
        explanation: str,
        confidence: float | None = None,
    ) -> ZRIAPredictionTrace:
        if self.fallback_backend is not None:
            fallback_trace = self.fallback_backend.predict_with_trace(query, parsed=parsed, kg_scope=kg_scope)
            return ZRIAPredictionTrace(
                configured_backend="learned",
                effective_backend=fallback_trace.effective_backend,
                fallback_fired=True,
                fallback_reason=explanation,
                control_event_type=(
                    "low_confidence_learned_fallback"
                    if "below threshold" in explanation
                    else "learned_backend_failure"
                ),
                confidence=confidence,
                prediction=fallback_trace.prediction,
            )
        return ZRIAPredictionTrace(
            configured_backend="learned",
            effective_backend="learned",
            fallback_fired=True,
            fallback_reason=explanation,
            control_event_type=(
                "low_confidence_learned_fallback"
                if "below threshold" in explanation
                else "learned_backend_failure"
            ),
            confidence=confidence,
            prediction=EvaluationTarget(
                verdict="insufficient_information",
                resolved=False,
                should_escalate=False,
                explanation=explanation,
            ),
        )


class RemoteZRIABackend(BaseZRIABackend):
    def __init__(
        self,
        *,
        remote_url: str,
        api_key: str | None = None,
        timeout_seconds: float = 10.0,
        retry_attempts: int = 1,
        retry_backoff_seconds: float = 0.25,
        fallback_backend: BaseZRIABackend | None = None,
    ) -> None:
        self.remote_url = remote_url
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.fallback_backend = fallback_backend

    def predict_with_trace(
        self,
        query: str,
        parsed: ParseTarget | None = None,
        kg_scope: KGScope | None = None,
    ) -> ZRIAPredictionTrace:
        payload = {
            "query": query,
            "parsed": None if parsed is None else parsed.model_dump(),
            "kg_scope": None if kg_scope is None else kg_scope.model_dump(exclude_none=True),
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.retry_attempts + 1):
            req = request.Request(self.remote_url, data=body, headers=headers, method="POST")
            try:
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                    parsed_payload = json.loads(raw) if raw.strip() else {}
                    return ZRIAPredictionTrace(
                        configured_backend="remote",
                        effective_backend="remote",
                        prediction=self._decode_response(parsed_payload),
                    )
            except Exception as exc:
                if attempt < self.retry_attempts:
                    delay = self.retry_backoff_seconds * (2**attempt)
                    if delay > 0:
                        time.sleep(delay)
                    continue
                return self._fallback(query, parsed, kg_scope, f"Remote ZRIA backend failed: {exc}")

        return self._fallback(query, parsed, kg_scope, "Remote ZRIA backend exhausted retries.")

    def _decode_response(self, payload: Any) -> EvaluationTarget:
        if isinstance(payload, dict):
            if "prediction" in payload:
                return EvaluationTarget.model_validate(payload["prediction"])
            if "evaluation" in payload:
                return EvaluationTarget.model_validate(payload["evaluation"])
            return EvaluationTarget.model_validate(payload)
        raise ValueError("Remote ZRIA backend returned a non-dict payload.")

    def _fallback(
        self,
        query: str,
        parsed: ParseTarget | None,
        kg_scope: KGScope | None,
        explanation: str,
    ) -> ZRIAPredictionTrace:
        if self.fallback_backend is not None:
            fallback_trace = self.fallback_backend.predict_with_trace(query, parsed=parsed, kg_scope=kg_scope)
            return ZRIAPredictionTrace(
                configured_backend="remote",
                effective_backend=fallback_trace.effective_backend,
                fallback_fired=True,
                fallback_reason=explanation,
                control_event_type="remote_backend_failure",
                prediction=fallback_trace.prediction,
            )
        return ZRIAPredictionTrace(
            configured_backend="remote",
            effective_backend="remote",
            fallback_fired=True,
            fallback_reason=explanation,
            control_event_type="remote_backend_failure",
            prediction=EvaluationTarget(
                verdict="insufficient_information",
                resolved=False,
                should_escalate=False,
                explanation=explanation,
            ),
        )


def _rules_fallback(config: ZRIAConfig) -> BaseZRIABackend | None:
    if not config.fallback_to_rules or config.backend == "rules":
        return None
    return RuleBasedZRIABackend.from_path(config.rules_path)


def load_zria_backend(config: ZRIAConfig) -> BaseZRIABackend:
    if config.backend == "rules":
        return RuleBasedZRIABackend.from_path(config.rules_path)
    if config.backend == "learned":
        return LearnedZRIABackend.from_path(
            config.learned_model_path,
            confidence_threshold=config.confidence_threshold,
            fallback_backend=_rules_fallback(config),
        )
    if config.backend == "remote":
        if not config.remote_url:
            raise ValueError("zria.remote_url must be set when backend='remote'.")
        return RemoteZRIABackend(
            remote_url=config.remote_url,
            api_key=config.remote_api_key,
            timeout_seconds=config.remote_timeout_seconds,
            retry_attempts=config.remote_retry_attempts,
            retry_backoff_seconds=config.remote_retry_backoff_seconds,
            fallback_backend=_rules_fallback(config),
        )
    raise ValueError(f"Unsupported ZRIA backend '{config.backend}'.")
