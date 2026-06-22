from pathlib import Path
import sys
import json
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    from fastapi.testclient import TestClient
except ModuleNotFoundError:  # pragma: no cover
    TestClient = None

from msa_zria.audit import AuditRecorder
from msa_zria.config import KGScope
from msa_zria.data import Triple
from msa_zria.zria_adapter import RuleBasedZRIAAdapter

if TestClient is not None:
    from msa_zria.main import create_app
else:  # pragma: no cover
    create_app = None


class _ParseModule:
    def __call__(self, text: str, kg_context=None):
        return {
            "parsed_result": {
                "task": "parse",
                "device": "monitor",
                "issue": "hardware failure",
                "cause": None,
                "severity": "medium",
            }
        }


class _CodeModule:
    def __call__(self, parsed, kg_context=None):
        return {
            "code_str": {
                "task": "code",
                "language": "python",
                "framework": "pyro",
                "entrypoint": "run_inference",
                "query_variable": "approve_refund",
                "required_statements": ["def run_inference"],
                "program": "def run_inference():\n    return {'resolved': True, 'should_escalate': False}\n",
            }
        }


class _EvalModule:
    def __call__(self, query, answer, kg_context=None):
        return {
            "evaluation": {
                "task": "evaluate",
                "verdict": "resolved",
                "resolved": True,
                "should_escalate": False,
                "explanation": str(answer),
            }
        }


class _UnsafeCodeModule:
    def __call__(self, parsed, kg_context=None):
        return {
            "code_str": {
                "task": "code",
                "language": "python",
                "framework": "pyro",
                "entrypoint": "run_inference",
                "query_variable": "approve_refund",
                "required_statements": ["def run_inference"],
                "program": "import os\n\ndef run_inference():\n    return os.listdir('.')\n",
            }
        }


def _read_audit_events(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@unittest.skipIf(TestClient is None, "fastapi test client dependencies are not installed")
class APIIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        rules_path = Path(__file__).resolve().parents[1] / "examples" / "zria_rules.json"
        app = create_app(
            parse_module=_ParseModule(),
            code_module=_CodeModule(),
            eval_module=_EvalModule(),
            zria_adapter=RuleBasedZRIAAdapter.from_rules_path(rules_path),
            initialize_ray=False,
        )
        self.client = TestClient(app)

    def test_infer_zria_returns_backend_answer(self) -> None:
        response = self.client.post(
            "/infer",
            json={
                "query": "The box was opened, but the monitor failed on day 12.",
                "mode": "zria",
                "kg_scope": {
                    "workspace": "urn:wwkg:workspace:example",
                    "branch": "policy-review",
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"]["verdict"], "resolved")
        self.assertEqual(payload["kg_scope"]["kg_branch"], "policy-review")

    def test_run_pyro_uses_controlled_runner(self) -> None:
        response = self.client.post(
            "/run_pyro",
            json={
                "parsed": {
                    "task": "parse",
                    "device": "monitor",
                    "issue": "hardware failure",
                    "cause": None,
                    "severity": "medium",
                }
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["pyro_result"]["success"])
        self.assertTrue(payload["pyro_result"]["answer"]["resolved"])

    def test_produce_dataset_writes_branch_aware_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("msa_zria.main.load_triples", return_value=[Triple(subject="A", predicate="rel", object="B")]):
                response = self.client.post(
                    "/produce_dataset",
                    json={
                        "output_path": tmpdir,
                        "format": "hybrid",
                        "kg": {
                            "backend": "wwkg",
                            "workspace": "urn:wwkg:workspace:example",
                            "branch": "main",
                        },
                    },
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["kg_metadata"]["kg_branch"], "main")
            triples_rows = Path(tmpdir, "triples.jsonl").read_text(encoding="utf-8")
            self.assertIn('"kg_branch": "main"', triples_rows)

    def test_infer_zria_writes_decision_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.jsonl"
            app = create_app(
                parse_module=_ParseModule(),
                code_module=_CodeModule(),
                eval_module=_EvalModule(),
                zria_adapter=RuleBasedZRIAAdapter.from_rules_path(
                    Path(__file__).resolve().parents[1] / "examples" / "zria_rules.json"
                ),
                audit_recorder=AuditRecorder.from_output_path(str(audit_path)),
                initialize_ray=False,
            )
            client = TestClient(app)

            response = client.post(
                "/infer",
                json={
                    "query": "The box was opened, but the monitor failed on day 12.",
                    "mode": "zria",
                    "kg_scope": {
                        "workspace": "urn:wwkg:workspace:example",
                        "branch": "policy-review",
                    },
                    "operator_override": {"reviewer": "alice"},
                },
            )

            self.assertEqual(response.status_code, 200)
            events = _read_audit_events(audit_path)
            self.assertEqual([event["event_type"] for event in events], ["decision_lineage"])
            self.assertEqual(events[0]["payload"]["backend_used"], "rules")
            self.assertEqual(events[0]["payload"]["operator_override"]["reviewer"], "alice")
            self.assertEqual(events[0]["payload"]["final_evaluation"]["verdict"], "resolved")

    def test_run_pyro_rejection_writes_control_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audit_path = Path(tmpdir) / "audit.jsonl"
            app = create_app(
                parse_module=_ParseModule(),
                code_module=_UnsafeCodeModule(),
                eval_module=_EvalModule(),
                zria_adapter=RuleBasedZRIAAdapter.from_rules_path(
                    Path(__file__).resolve().parents[1] / "examples" / "zria_rules.json"
                ),
                audit_recorder=AuditRecorder.from_output_path(str(audit_path)),
                initialize_ray=False,
            )
            client = TestClient(app)

            response = client.post(
                "/run_pyro",
                json={
                    "parsed": {
                        "task": "parse",
                        "device": "monitor",
                        "issue": "hardware failure",
                        "cause": None,
                        "severity": "medium",
                    }
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertFalse(response.json()["pyro_result"]["success"])
            events = _read_audit_events(audit_path)
            self.assertEqual([event["event_type"] for event in events], ["control_event"])
            self.assertEqual(events[0]["payload"]["control_type"], "disallowed_import")
            self.assertEqual(events[0]["payload"]["backend_type"], "pyro")


if __name__ == "__main__":
    unittest.main()
