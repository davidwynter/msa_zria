from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.config import KGConfig, KGScope, ZRIAConfig
from msa_zria.data import EvaluationTarget, ParseTarget, Triple
from msa_zria.zria_adapter import RuleBasedZRIAAdapter
from msa_zria.zria_backend import LearnedGraphZRIABackend, load_zria_backend


class ZRIABackendTest(unittest.TestCase):
    def setUp(self) -> None:
        rules_path = Path(__file__).resolve().parents[1] / "examples" / "zria_rules.json"
        self.adapter = RuleBasedZRIAAdapter.from_rules_path(rules_path)

    def test_branch_specific_rule_matches(self) -> None:
        result = self.adapter.predict(
            "Router keeps overheating",
            parsed=ParseTarget(task="parse", device="Router123", issue="Overheating", cause=None, severity="high"),
            kg_scope=KGScope(workspace="urn:wwkg:workspace:example", branch="support-hotfix"),
        )
        self.assertEqual(result.verdict, "escalate")
        self.assertTrue(result.should_escalate)

    def test_default_outcome_when_scope_does_not_match(self) -> None:
        result = self.adapter.predict(
            "Router keeps overheating",
            parsed=ParseTarget(task="parse", device="Router123", issue="Overheating", cause=None, severity="high"),
            kg_scope=KGScope(workspace="urn:wwkg:workspace:example", branch="main"),
        )
        self.assertEqual(result.verdict, "insufficient_information")
        self.assertFalse(result.should_escalate)

    def test_learned_graph_backend_uses_retrieved_neighborhood(self) -> None:
        backend = LearnedGraphZRIABackend(
            object(),
            object(),
            kg_config=KGConfig(
                backend="wwkg",
                base_url="http://127.0.0.1:4242",
                workspace="urn:wwkg:workspace:example",
                branch="main",
            ),
            neighborhood_limit=8,
            confidence_threshold=0.2,
        )
        parsed = ParseTarget(task="parse", device="Router123", issue="Overheating", cause=None, severity="high")
        with (
            patch(
                "msa_zria.zria_backend.retrieve_neighborhood",
                return_value=[Triple(subject="Router123", predicate="hasIssue", object="Overheating")],
            ) as retrieve_mock,
            patch(
                "msa_zria.zria_backend.predict_with_model",
                return_value=(
                    EvaluationTarget(
                        verdict="escalate",
                        resolved=False,
                        should_escalate=True,
                        explanation="graph match",
                    ),
                    0.9,
                ),
            ) as predict_mock,
        ):
            trace = backend.predict_with_trace(
                "Router keeps overheating",
                parsed=parsed,
                kg_scope=KGScope(workspace="urn:wwkg:workspace:example", branch="support-hotfix"),
            )
        self.assertEqual(trace.effective_backend, "learned_graph")
        self.assertFalse(trace.fallback_fired)
        self.assertEqual(trace.prediction.verdict, "escalate")
        retrieve_mock.assert_called_once()
        self.assertEqual(predict_mock.call_args.kwargs["neighborhood"][0].predicate, "hasIssue")

    def test_learned_graph_backend_surfaces_calibration_and_graph_explanation(self) -> None:
        backend = LearnedGraphZRIABackend(
            object(),
            object(),
            kg_config=KGConfig(
                backend="wwkg",
                base_url="http://127.0.0.1:4242",
                workspace="urn:wwkg:workspace:example",
                branch="main",
            ),
            neighborhood_limit=8,
            confidence_threshold=0.2,
        )
        parsed = ParseTarget(task="parse", device="Router123", issue="Overheating", cause=None, severity="high")
        with (
            patch(
                "msa_zria.zria_backend.retrieve_neighborhood",
                return_value=[Triple(subject="Router123", predicate="hasIssue", object="Overheating")],
            ),
            patch(
                "msa_zria.zria_backend.predict_with_model",
                return_value=(
                    EvaluationTarget(
                        verdict="escalate",
                        resolved=False,
                        should_escalate=True,
                        explanation="graph match",
                    ),
                    0.82,
                    {
                        "raw_confidence": 0.91,
                        "temperature": 1.5,
                        "graph_explanation": [
                            {
                                "subject": "Router123",
                                "predicate": "hasIssue",
                                "object": "Overheating",
                                "score": 1.0,
                            }
                        ],
                    },
                ),
            ),
        ):
            trace = backend.predict_with_trace(
                "Router keeps overheating",
                parsed=parsed,
                kg_scope=KGScope(workspace="urn:wwkg:workspace:example", branch="support-hotfix"),
            )
        self.assertEqual(trace.confidence, 0.82)
        self.assertEqual(trace.raw_confidence, 0.91)
        self.assertEqual(trace.calibration_temperature, 1.5)
        self.assertEqual(len(trace.graph_explanation), 1)
        self.assertEqual(trace.graph_explanation[0].predicate, "hasIssue")

    def test_load_zria_backend_requires_kg_config_for_learned_graph(self) -> None:
        with self.assertRaises(ValueError):
            load_zria_backend(
                ZRIAConfig(
                    backend="learned_graph",
                    learned_graph_model_path="outputs/zria_graph_learned",
                )
            )


if __name__ == "__main__":
    unittest.main()
