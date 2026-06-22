from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.config import ZRIAConfig
from msa_zria.zria import compare_backends, load_zria_examples, train_model
from msa_zria.zria_backend import LearnedZRIABackend, load_zria_backend

try:
    import torch  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ZRIALearnedBackendTest(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        self.train_examples = load_zria_examples(repo_root / "examples" / "zria_examples_train.jsonl")
        self.eval_examples = load_zria_examples(repo_root / "examples" / "zria_examples_eval.jsonl")
        self.rules_path = repo_root / "examples" / "zria_rules.json"

    def test_train_and_load_learned_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_model(
                self.train_examples,
                tmpdir,
                eval_examples=self.eval_examples,
                epochs=30,
                batch_size=2,
                embedding_dim=16,
                hidden_dim=16,
            )
            self.assertGreaterEqual(metrics["train_examples"], 1)

            backend = LearnedZRIABackend.from_path(tmpdir, confidence_threshold=0.0)
            prediction = backend.predict(
                self.eval_examples[0].query,
                parsed=self.eval_examples[0].parsed,
                kg_scope=self.eval_examples[0].kg_scope,
            )
            self.assertEqual(prediction.verdict, "escalate")

    def test_load_zria_backend_supports_learned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_model(
                self.train_examples,
                tmpdir,
                epochs=30,
                batch_size=2,
                embedding_dim=16,
                hidden_dim=16,
            )
            config = ZRIAConfig(
                backend="learned",
                learned_model_path=tmpdir,
                confidence_threshold=0.0,
                rules_path=str(self.rules_path),
            )
            backend = load_zria_backend(config)
            prediction = backend.predict(
                self.eval_examples[1].query,
                parsed=self.eval_examples[1].parsed,
                kg_scope=self.eval_examples[1].kg_scope,
            )
            self.assertEqual(prediction.verdict, "resolved")

    def test_compare_backends_reports_rules_and_learned(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_model(
                self.train_examples,
                tmpdir,
                epochs=30,
                batch_size=2,
                embedding_dim=16,
                hidden_dim=16,
            )
            report = compare_backends(
                learned_model_path=tmpdir,
                rules_path=self.rules_path,
                examples=self.eval_examples,
                confidence_threshold=0.0,
            )
            self.assertEqual([summary.backend for summary in report.summaries], ["rules", "learned"])
            self.assertEqual(len(report.details), 6)

    def test_learned_backend_reports_low_confidence_fallback_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_model(
                self.train_examples,
                tmpdir,
                epochs=30,
                batch_size=2,
                embedding_dim=16,
                hidden_dim=16,
            )
            backend = LearnedZRIABackend.from_path(
                tmpdir,
                confidence_threshold=1.1,
                fallback_backend=load_zria_backend(
                    ZRIAConfig(
                        backend="rules",
                        rules_path=str(self.rules_path),
                    )
                ),
            )
            trace = backend.predict_with_trace(
                self.eval_examples[0].query,
                parsed=self.eval_examples[0].parsed,
                kg_scope=self.eval_examples[0].kg_scope,
            )
            self.assertTrue(trace.fallback_fired)
            self.assertEqual(trace.effective_backend, "rules")
            self.assertEqual(trace.control_event_type, "low_confidence_learned_fallback")
            self.assertIsNotNone(trace.confidence)


if __name__ == "__main__":
    unittest.main()
