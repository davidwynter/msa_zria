from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
from urllib import error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.config import KGScope, ZRIAConfig
from msa_zria.data import ParseTarget
from msa_zria.zria_backend import load_zria_backend


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class ZRIARemoteBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ZRIAConfig(
            backend="remote",
            remote_url="http://zria.local/predict",
            rules_path=str(Path(__file__).resolve().parents[1] / "examples" / "zria_rules.json"),
        )
        self.parsed = ParseTarget(
            task="parse",
            device="Router123",
            issue="Overheating",
            cause="Blocked ventilation",
            severity="high",
        )
        self.scope = KGScope(workspace="urn:wwkg:workspace:example", branch="support-hotfix")

    def test_remote_backend_uses_service_response(self) -> None:
        backend = load_zria_backend(self.config)
        response_payload = {
            "prediction": {
                "task": "evaluate",
                "verdict": "resolved",
                "resolved": True,
                "should_escalate": False,
                "explanation": "Remote backend resolved the case.",
            }
        }
        with patch("msa_zria.zria_backend.request.urlopen", return_value=_FakeResponse(response_payload)):
            prediction = backend.predict("query", parsed=self.parsed, kg_scope=self.scope)
        self.assertEqual(prediction.verdict, "resolved")

    def test_remote_backend_falls_back_to_rules_on_failure(self) -> None:
        backend = load_zria_backend(self.config)
        with patch("msa_zria.zria_backend.request.urlopen", side_effect=error.URLError("offline")):
            trace = backend.predict_with_trace("query", parsed=self.parsed, kg_scope=self.scope)
        self.assertEqual(trace.prediction.verdict, "escalate")
        self.assertEqual(trace.effective_backend, "rules")
        self.assertTrue(trace.fallback_fired)
        self.assertEqual(trace.control_event_type, "remote_backend_failure")


if __name__ == "__main__":
    unittest.main()
