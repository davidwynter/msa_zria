from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.config import KGConfig, ZRIAConfig
from msa_zria.data import Triple
from msa_zria.zria import (
    compare_backends,
    load_artifact,
    load_zria_examples,
    run_sweep,
    train_graph_model,
    train_model,
)
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
        for example in self.train_examples + self.eval_examples:
            example.neighborhood = [
                Triple(subject=example.parsed.device, predicate="hasIssue", object=example.parsed.issue),
                Triple(subject=example.parsed.device, predicate="severity", object=example.parsed.severity or "unknown"),
            ]

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
            self.assertGreaterEqual(metrics["memory_labels_persisted"], 1)

            backend = LearnedZRIABackend.from_path(tmpdir, confidence_threshold=0.0)
            prediction = backend.predict(
                self.eval_examples[0].query,
                parsed=self.eval_examples[0].parsed,
                kg_scope=self.eval_examples[0].kg_scope,
            )
            self.assertEqual(prediction.verdict, "escalate")

            artifact, model = load_artifact(tmpdir)
            self.assertTrue(artifact.memory_enabled)
            self.assertEqual(len(artifact.memory_vectors), len(artifact.labels))
            self.assertTrue(any(count > 0.0 for count in artifact.memory_counts))
            self.assertGreater(float(model.label_memory_counts.sum().item()), 0.0)
            self.assertGreaterEqual(artifact.calibration_temperature, 0.25)

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

    def test_train_model_reports_early_stopping_and_spectral_dropout_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_model(
                self.train_examples,
                tmpdir,
                eval_examples=self.eval_examples,
                epochs=30,
                batch_size=2,
                embedding_dim=16,
                spectral_k=8,
                hidden_dim=16,
                dropout=0.2,
                patience=3,
            )
            self.assertEqual(metrics["spectral_k"], 8)
            self.assertEqual(metrics["dropout"], 0.2)
            self.assertGreaterEqual(metrics["best_epoch"], 1)
            self.assertGreaterEqual(metrics["epochs_trained"], metrics["best_epoch"])
            self.assertIn("early_stopped", metrics)
            self.assertEqual(metrics["memory_momentum"], 0.9)

    def test_train_model_can_resume_persisted_memory_for_transfer(self) -> None:
        with tempfile.TemporaryDirectory() as base_tmpdir, tempfile.TemporaryDirectory() as resumed_tmpdir:
            train_model(
                self.train_examples,
                base_tmpdir,
                epochs=20,
                batch_size=2,
                embedding_dim=16,
                hidden_dim=16,
            )
            base_artifact, _ = load_artifact(base_tmpdir)
            self.assertTrue(any(count > 0.0 for count in base_artifact.memory_counts))

            metrics = train_model(
                self.train_examples,
                resumed_tmpdir,
                eval_examples=self.eval_examples,
                epochs=20,
                batch_size=2,
                embedding_dim=16,
                hidden_dim=16,
                resume_memory_from=base_tmpdir,
            )
            self.assertGreaterEqual(metrics["memory_labels_restored"], 1)

            resumed_artifact, model = load_artifact(resumed_tmpdir)
            self.assertEqual(len(resumed_artifact.memory_vectors), len(base_artifact.memory_vectors))
            self.assertGreater(float(model.label_memory_counts.sum().item()), 0.0)

    def test_run_sweep_reports_all_parameter_combinations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_sweep(
                train_examples=self.train_examples,
                eval_examples=self.eval_examples,
                output_dir=tmpdir,
                spectral_k_values=[4, 8],
                dropout_values=[0.0, 0.2],
                epochs=20,
                batch_size=2,
                embedding_dim=16,
                hidden_dim=16,
                patience=3,
            )
            self.assertEqual(len(report.records), 4)
            self.assertIn(report.best_record.spectral_k, {4, 8})
            self.assertIn(report.best_record.dropout, {0.0, 0.2})
            self.assertTrue(Path(report.best_record.artifact_path).exists())

    def test_train_and_load_learned_graph_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics = train_graph_model(
                self.train_examples,
                tmpdir,
                eval_examples=self.eval_examples,
                epochs=20,
                batch_size=2,
                embedding_dim=16,
                spectral_k=4,
                hidden_dim=16,
                dropout=0.2,
                patience=3,
            )
            self.assertEqual(metrics["backend_type"], "learned_graph")
            self.assertGreaterEqual(metrics["memory_labels_persisted"], 1)
            self.assertIn(metrics["calibration_source"], {"train", "eval"})
            config = ZRIAConfig(
                backend="learned_graph",
                learned_graph_model_path=tmpdir,
                confidence_threshold=0.0,
                rules_path=str(self.rules_path),
                graph_neighborhood_limit=8,
            )
            backend = load_zria_backend(
                config,
                kg_config=KGConfig(
                    backend="wwkg",
                    base_url="http://127.0.0.1:4242",
                    workspace="urn:wwkg:workspace:example",
                    branch="main",
                ),
            )
            with patch(
                "msa_zria.zria_backend.retrieve_neighborhood",
                return_value=self.eval_examples[0].neighborhood,
            ):
                prediction = backend.predict(
                    self.eval_examples[0].query,
                    parsed=self.eval_examples[0].parsed,
                    kg_scope=self.eval_examples[0].kg_scope,
                )
            self.assertIn(prediction.verdict, {"resolved", "escalate", "insufficient_information", "unresolved"})

            artifact, model = load_artifact(tmpdir)
            self.assertEqual(artifact.backend_type, "learned_graph")
            self.assertEqual(len(artifact.memory_vectors), len(artifact.labels))
            self.assertGreater(float(model.label_memory_counts.sum().item()), 0.0)
            self.assertTrue(artifact.relation_vocab)
            self.assertGreaterEqual(artifact.graph_max_nodes, 2)
            self.assertGreaterEqual(artifact.graph_max_edges, 2)


if __name__ == "__main__":
    unittest.main()
