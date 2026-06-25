from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.config import KGScope
from msa_zria.evidence import EvidenceSnippet
from msa_zria.reasoning_pipeline import ReasoningPipeline
from msa_zria.zria_adapter import RuleBasedZRIAAdapter


class _ParseModule:
    def __call__(self, text: str, kg_context=None, evidence_context=None):
        return {
            "parsed_result": {
                "task": "parse",
                "device": "Router123",
                "issue": "Overheating",
                "cause": "Blocked ventilation",
                "severity": "high",
            }
        }


class _CodeModule:
    def __call__(self, parsed, kg_context=None, evidence_context=None):
        return {
            "code_str": {
                "task": "code",
                "language": "python",
                "framework": "pyro",
                "entrypoint": "run_inference",
                "query_variable": "failure",
                "required_statements": ["def run_inference"],
                "program": (
                    "def run_inference():\n"
                    "    return {'recommended_action': 'restart', 'resolved': True, 'should_escalate': False}\n"
                ),
            }
        }


class _EvalModule:
    def __call__(self, query, answer, kg_context=None, evidence_context=None):
        resolved = isinstance(answer, dict) and bool(answer.get("resolved"))
        return {
            "evaluation": {
                "task": "evaluate",
                "verdict": "resolved" if resolved else "insufficient_information",
                "resolved": resolved,
                "should_escalate": False,
                "explanation": str(answer),
            }
        }


class _EvidenceRetriever:
    def retrieve(self, query: str, *, parsed=None, kg_scope=None):
        del parsed, kg_scope
        return [EvidenceSnippet(text=f"{query} evidence", score=3.0)]


class _EvidenceAwareParseModule:
    def __init__(self) -> None:
        self.last_evidence_context = None

    def __call__(self, text: str, kg_context=None, evidence_context=None):
        self.last_evidence_context = evidence_context
        return {
            "parsed_result": {
                "task": "parse",
                "device": "Router123",
                "issue": "Overheating",
                "cause": "Blocked ventilation",
                "severity": "high",
            }
        }


class PipelineIntegrationTest(unittest.TestCase):
    def test_pipeline_runs_end_to_end_with_real_zria_backend(self) -> None:
        rules_path = Path(__file__).resolve().parents[1] / "examples" / "zria_rules.json"
        pipeline = ReasoningPipeline(
            _ParseModule(),
            _CodeModule(),
            _EvalModule(),
            zria_adapter=RuleBasedZRIAAdapter.from_rules_path(rules_path),
        )

        result = pipeline.run(
            "Router keeps overheating",
            kg_scope=KGScope(workspace="urn:wwkg:workspace:example", branch="support-hotfix"),
        )

        self.assertTrue(result.pyro.success)
        self.assertEqual(result.evaluation.verdict, "resolved")
        self.assertEqual(result.kg_scope.branch, "support-hotfix")

    def test_pipeline_injects_retrieved_evidence_into_parse_stage(self) -> None:
        rules_path = Path(__file__).resolve().parents[1] / "examples" / "zria_rules.json"
        parse_module = _EvidenceAwareParseModule()
        pipeline = ReasoningPipeline(
            parse_module,
            _CodeModule(),
            _EvalModule(),
            zria_adapter=RuleBasedZRIAAdapter.from_rules_path(rules_path),
            evidence_retriever=_EvidenceRetriever(),
        )

        pipeline.run("Router keeps overheating")

        self.assertIsNotNone(parse_module.last_evidence_context)
        self.assertIn("Retrieved evidence", parse_module.last_evidence_context)


if __name__ == "__main__":
    unittest.main()
