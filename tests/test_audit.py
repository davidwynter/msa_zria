from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.audit import AuditRecorder
from msa_zria.config import KGScope
from msa_zria.data import EvaluationTarget, ParseTarget


def _read_events(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class AuditRecorderTest(unittest.TestCase):
    def test_dataset_lineage_records_scope_hash_and_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.jsonl"
            recorder = AuditRecorder.from_output_path(str(audit_path))

            recorder.record_dataset_lineage(
                source_case_id="case-123",
                input_file_hash="abc123",
                output_record_ids=["case-123-parse-hybrid", "case-123-code-hybrid"],
                kg_scope=KGScope(workspace="urn:wwkg:workspace:test", branch="support-hotfix"),
            )

            events = _read_events(audit_path)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "dataset_lineage")
            self.assertEqual(events[0]["kg_scope"]["workspace"], "urn:wwkg:workspace:test")
            self.assertEqual(events[0]["payload"]["source_case_id"], "case-123")
            self.assertEqual(events[0]["payload"]["input_file_hash"], "abc123")
            self.assertEqual(
                events[0]["payload"]["output_record_ids"],
                ["case-123-parse-hybrid", "case-123-code-hybrid"],
            )

    def test_model_and_validation_events_capture_lineage_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.jsonl"
            recorder = AuditRecorder.from_output_path(str(audit_path))

            recorder.record_model_lineage(
                experiment_config_hash="cfg-hash",
                training_dataset_version="dataset-v1",
                model_artifact_path="outputs/model",
                model_artifact_hash="model-hash",
                backend_type="learned",
                fallback_setting=True,
                confidence_threshold=0.72,
            )
            recorder.record_validation_evidence(
                report_id="report-123",
                evidence_type="learned_vs_rules_comparison",
                per_backend_scores={
                    "rules": {"accuracy": 0.5, "passed_cases": 1, "total_cases": 2, "average_score": 0.5},
                    "learned": {"accuracy": 1.0, "passed_cases": 2, "total_cases": 2, "average_score": 1.0},
                },
                artifact_path="outputs/report.json",
                learned_vs_rules_report={"winner": "learned"},
            )

            events = _read_events(audit_path)
            self.assertEqual([event["event_type"] for event in events], ["model_lineage", "validation_evidence"])
            self.assertEqual(events[0]["payload"]["backend_type"], "learned")
            self.assertTrue(events[0]["payload"]["fallback_setting"])
            self.assertEqual(events[0]["payload"]["confidence_threshold"], 0.72)
            self.assertEqual(events[1]["payload"]["report_id"], "report-123")
            self.assertEqual(events[1]["payload"]["learned_vs_rules_report"]["winner"], "learned")

    def test_branch_promotion_uses_requested_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            promotion_path = Path(tmpdir) / "promotions.jsonl"
            recorder = AuditRecorder.from_output_path(str(promotion_path))

            recorder.record_branch_promotion(
                source_workspace="urn:wwkg:workspace:staging",
                source_branch="candidate-a",
                production_workspace="urn:wwkg:workspace:prod",
                approver="compliance@example.com",
                evidence_report_id="report-123",
            )

            events = _read_events(promotion_path)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["event_type"], "branch_promotion")
            self.assertEqual(events[0]["payload"]["production_workspace"], "urn:wwkg:workspace:prod")
            self.assertEqual(events[0]["payload"]["evidence_report_id"], "report-123")


if __name__ == "__main__":
    unittest.main()
