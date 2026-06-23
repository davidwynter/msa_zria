from pathlib import Path
import os
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT.parent / "contextkg" / "src"))

from msa_zria.audit import AuditRecorder
from msa_zria.config import KGConfig, KGScope
from msa_zria.kg import load_triples

try:
    from contextkg import ContextMemoryStore, WWKGClient, WWKGConfig
except ModuleNotFoundError:  # pragma: no cover
    ContextMemoryStore = None
    WWKGClient = None
    WWKGConfig = None


@unittest.skipUnless(os.getenv("MSA_ZRIA_RUN_WWKG_LIVE") == "1", "live WWKG checks are disabled")
@unittest.skipIf(ContextMemoryStore is None or WWKGClient is None or WWKGConfig is None, "contextkg is not available")
class WWKGLiveIntegrationTest(unittest.TestCase):
    def test_load_triples_from_live_wwkg(self) -> None:
        config = KGConfig(
            backend="wwkg",
            base_url=os.getenv("WWKG_BASE_URL", "http://127.0.0.1:4242"),
            api_key=os.getenv("WWKG_API_KEY"),
            workspace=os.getenv("WWKG_WORKSPACE"),
            branch=os.getenv("WWKG_BRANCH"),
            commit=os.getenv("WWKG_COMMIT"),
            as_of=os.getenv("WWKG_AS_OF"),
            sparql_query="SELECT ?subject ?predicate ?object WHERE { ?subject ?predicate ?object } LIMIT 1",
        )
        triples = load_triples(config)
        self.assertIsInstance(triples, list)

    def test_audit_event_is_mirrored_to_live_wwkg(self) -> None:
        scope = KGScope(
            workspace=os.getenv("WWKG_WORKSPACE"),
            branch=os.getenv("WWKG_BRANCH"),
            commit=os.getenv("WWKG_COMMIT"),
            as_of=os.getenv("WWKG_AS_OF"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = AuditRecorder.from_output_path(
                str(Path(tmpdir) / "audit.jsonl"),
                wwkg_enabled=True,
                default_scope=scope,
            )
            event = recorder.record_control_event(
                control_type="wwkg_live_probe",
                backend_type="wwkg",
                details={"probe": "audit_roundtrip"},
                kg_scope=scope,
            )

            store = ContextMemoryStore(
                WWKGClient(
                    WWKGConfig.from_env().scoped(
                        workspace=scope.workspace,
                        branch=scope.branch,
                        commit=scope.commit,
                        as_of=scope.as_of,
                    )
                )
            )
            mirrored = store.get_memory(event.event_id)
            self.assertIsNotNone(mirrored)
            self.assertEqual(mirrored.memory_id, event.event_id)
            self.assertEqual(mirrored.subject, "audit:control_event")


if __name__ == "__main__":
    unittest.main()
