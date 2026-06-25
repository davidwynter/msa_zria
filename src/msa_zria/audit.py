from __future__ import annotations

import argparse
import hashlib
import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from msa_zria.config import AuditConfig, ExperimentConfig, KGScope
from msa_zria.data import EvaluationTarget, ParseTarget

LOGGER = logging.getLogger(__name__)


class AuditEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    timestamp: datetime
    source: str
    kg_scope: KGScope | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class DatasetLineagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_case_id: str
    workspace: str | None = None
    branch: str | None = None
    ingestion_timestamp: datetime
    input_file_hash: str
    output_record_ids: list[str]


class ModelLineagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_config_hash: str
    training_dataset_version: str
    model_artifact_path: str
    model_artifact_hash: str
    backend_type: str
    fallback_setting: bool
    confidence_threshold: float | None = None


class DecisionLineagePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    parsed_state: dict[str, Any]
    backend_used: str
    fallback_fired: bool
    final_evaluation: dict[str, Any]
    operator_override: dict[str, Any] | None = None


class ControlEventPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    control_type: str
    backend_type: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationEvidencePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_id: str
    evidence_type: str
    per_backend_scores: dict[str, dict[str, float | int]]
    artifact_path: str | None = None
    learned_vs_rules_report: dict[str, Any] | None = None


class BranchPromotionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_workspace: str
    source_branch: str
    production_workspace: str
    approver: str
    approved_at: datetime
    notes: str | None = None
    evidence_report_id: str | None = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))


def sha256_file(path: str | Path) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    dir_path = Path(path)
    digest = hashlib.sha256()
    for file_path in sorted(p for p in dir_path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(dir_path)).encode("utf-8"))
        digest.update(sha256_file(file_path).encode("utf-8"))
    return digest.hexdigest()


def stable_dataset_version(paths: list[str | Path]) -> str:
    manifest = []
    for path in paths:
        if path is None:
            continue
        file_path = Path(path)
        if file_path.exists():
            manifest.append({"path": str(file_path), "sha256": sha256_file(file_path)})
        else:
            manifest.append({"path": str(file_path), "sha256": "missing"})
    return sha256_json(manifest)


def event_report_id(payload: Any) -> str:
    return sha256_json(payload)


class AuditRecorder:
    def __init__(self, config: AuditConfig, default_scope: KGScope | None = None) -> None:
        self.config = config
        self.default_scope = default_scope

    @classmethod
    def from_experiment(cls, config: ExperimentConfig) -> "AuditRecorder | None":
        if not config.audit.enabled:
            return None
        default_scope = config.kg.effective_scope()
        return cls(config.audit, default_scope=default_scope)

    @classmethod
    def from_output_path(
        cls,
        output_path: str,
        *,
        wwkg_enabled: bool = False,
        source: str = "msa_zria",
        default_scope: KGScope | None = None,
        promotion_output_path: str | None = None,
    ) -> "AuditRecorder":
        return cls(
            AuditConfig(
                enabled=True,
                output_path=output_path,
                wwkg_enabled=wwkg_enabled,
                source=source,
                promotion_output_path=promotion_output_path or output_path,
            ),
            default_scope=default_scope,
        )

    def record_dataset_lineage(
        self,
        *,
        source_case_id: str,
        input_file_hash: str,
        output_record_ids: list[str],
        kg_scope: KGScope | None = None,
    ) -> AuditEvent:
        payload = DatasetLineagePayload(
            source_case_id=source_case_id,
            workspace=(kg_scope or self.default_scope).workspace if (kg_scope or self.default_scope) else None,
            branch=(kg_scope or self.default_scope).branch if (kg_scope or self.default_scope) else None,
            ingestion_timestamp=utc_now(),
            input_file_hash=input_file_hash,
            output_record_ids=output_record_ids,
        )
        return self.record_event("dataset_lineage", payload, kg_scope=kg_scope)

    def record_model_lineage(
        self,
        *,
        experiment_config_hash: str,
        training_dataset_version: str,
        model_artifact_path: str,
        model_artifact_hash: str,
        backend_type: str,
        fallback_setting: bool,
        confidence_threshold: float | None,
        kg_scope: KGScope | None = None,
    ) -> AuditEvent:
        payload = ModelLineagePayload(
            experiment_config_hash=experiment_config_hash,
            training_dataset_version=training_dataset_version,
            model_artifact_path=model_artifact_path,
            model_artifact_hash=model_artifact_hash,
            backend_type=backend_type,
            fallback_setting=fallback_setting,
            confidence_threshold=confidence_threshold,
        )
        return self.record_event("model_lineage", payload, kg_scope=kg_scope)

    def record_decision_lineage(
        self,
        *,
        query: str,
        parsed_state: ParseTarget | dict[str, Any],
        backend_used: str,
        fallback_fired: bool,
        final_evaluation: EvaluationTarget | dict[str, Any],
        operator_override: dict[str, Any] | None = None,
        kg_scope: KGScope | None = None,
    ) -> AuditEvent:
        payload = DecisionLineagePayload(
            query=query,
            parsed_state=parsed_state if isinstance(parsed_state, dict) else parsed_state.model_dump(),
            backend_used=backend_used,
            fallback_fired=fallback_fired,
            final_evaluation=(
                final_evaluation if isinstance(final_evaluation, dict) else final_evaluation.model_dump()
            ),
            operator_override=operator_override,
        )
        return self.record_event("decision_lineage", payload, kg_scope=kg_scope)

    def record_control_event(
        self,
        *,
        control_type: str,
        backend_type: str | None = None,
        details: dict[str, Any] | None = None,
        kg_scope: KGScope | None = None,
    ) -> AuditEvent:
        payload = ControlEventPayload(
            control_type=control_type,
            backend_type=backend_type,
            details=details or {},
        )
        return self.record_event("control_event", payload, kg_scope=kg_scope)

    def record_validation_evidence(
        self,
        *,
        report_id: str,
        evidence_type: str,
        per_backend_scores: dict[str, dict[str, float | int]],
        artifact_path: str | None = None,
        learned_vs_rules_report: dict[str, Any] | None = None,
        kg_scope: KGScope | None = None,
    ) -> AuditEvent:
        payload = ValidationEvidencePayload(
            report_id=report_id,
            evidence_type=evidence_type,
            per_backend_scores=per_backend_scores,
            artifact_path=artifact_path,
            learned_vs_rules_report=learned_vs_rules_report,
        )
        return self.record_event("validation_evidence", payload, kg_scope=kg_scope)

    def record_branch_promotion(
        self,
        *,
        source_workspace: str,
        source_branch: str,
        production_workspace: str,
        approver: str,
        notes: str | None = None,
        evidence_report_id: str | None = None,
    ) -> AuditEvent:
        payload = BranchPromotionPayload(
            source_workspace=source_workspace,
            source_branch=source_branch,
            production_workspace=production_workspace,
            approver=approver,
            approved_at=utc_now(),
            notes=notes,
            evidence_report_id=evidence_report_id,
        )
        return self.record_event(
            "branch_promotion",
            payload,
            kg_scope=KGScope(workspace=production_workspace, branch=source_branch),
            output_path_override=self.config.promotion_output_path,
        )

    def record_event(
        self,
        event_type: str,
        payload: BaseModel | dict[str, Any],
        *,
        kg_scope: KGScope | None = None,
        output_path_override: str | None = None,
    ) -> AuditEvent:
        resolved_scope = kg_scope or self.default_scope
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=utc_now(),
            source=self.config.source,
            kg_scope=resolved_scope,
            payload=payload.model_dump(mode="json") if isinstance(payload, BaseModel) else payload,
        )
        self._write_local(event, output_path_override=output_path_override)
        if self.config.wwkg_enabled:
            self._write_wwkg(event)
        return event

    def _write_local(self, event: AuditEvent, *, output_path_override: str | None = None) -> None:
        output_path = Path(output_path_override or self.config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(event.model_dump_json() + "\n")

    def _write_wwkg(self, event: AuditEvent) -> None:
        try:
            from contextkg import ContextMemoryStore, MemoryItem, WWKGClient, WWKGConfig
        except ModuleNotFoundError:
            LOGGER.warning("WWKG audit mirror was enabled but contextkg is not available.")
            return

        try:
            config = WWKGConfig.from_env()
            scoped_config = config.scoped(
                workspace=event.kg_scope.workspace if event.kg_scope else None,
                branch=event.kg_scope.branch if event.kg_scope else None,
                commit=event.kg_scope.commit if event.kg_scope else None,
                as_of=event.kg_scope.as_of if event.kg_scope else None,
            )
            store = ContextMemoryStore(WWKGClient(scoped_config))
            item = MemoryItem(
                memory_id=event.event_id,
                subject=f"audit:{event.event_type}",
                predicate="recorded",
                obj=json.dumps(event.model_dump(mode="json"), sort_keys=True),
                source=self.config.source,
                created_at=event.timestamp,
                tags=("msa_zria", event.event_type),
            )
            store.put_memory(item)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to mirror audit event to WWKG: %s", exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record msa_zria audit events.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--output", default="outputs/audit/audit.jsonl")
    promote_parser.add_argument("--source-workspace", required=True)
    promote_parser.add_argument("--source-branch", required=True)
    promote_parser.add_argument("--production-workspace", required=True)
    promote_parser.add_argument("--approver", required=True)
    promote_parser.add_argument("--notes")
    promote_parser.add_argument("--evidence-report-id")
    promote_parser.add_argument("--wwkg-enabled", action="store_true")

    args = parser.parse_args()

    if args.command == "promote":
        recorder = AuditRecorder.from_output_path(
            args.output,
            wwkg_enabled=args.wwkg_enabled,
        )
        event = recorder.record_branch_promotion(
            source_workspace=args.source_workspace,
            source_branch=args.source_branch,
            production_workspace=args.production_workspace,
            approver=args.approver,
            notes=args.notes,
            evidence_report_id=args.evidence_report_id,
        )
        print(event.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
