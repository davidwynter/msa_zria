from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.cli import main
from msa_zria.runtime import build_runtime_dependencies
from msa_zria.zria_adapter import RuleBasedZRIAAdapter


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
                "program": "def run_inference():\n    return {'branch': 'non_thinking', 'resolved': True, 'should_escalate': False}\n",
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


class _ThinkingParseModule:
    def __call__(self, text: str, kg_context=None):
        return {
            "parsed_result": {
                "task": "parse",
                "device": "thinking-monitor",
                "issue": "policy exception review",
                "cause": "multi_step_reasoning",
                "severity": "high",
            }
        }


class _ThinkingCodeModule:
    def __call__(self, parsed, kg_context=None):
        return {
            "code_str": {
                "task": "code",
                "language": "python",
                "framework": "pyro",
                "entrypoint": "run_inference",
                "query_variable": "thinking_path",
                "required_statements": ["def run_inference"],
                "program": "def run_inference():\n    return {'branch': 'thinking', 'resolved': True, 'should_escalate': False}\n",
            }
        }


class _ThinkingEvalModule:
    def __call__(self, query, answer, kg_context=None):
        return {
            "evaluation": {
                "task": "evaluate",
                "verdict": "resolved",
                "resolved": True,
                "should_escalate": False,
                "explanation": f"thinking:{answer}",
            }
        }


class CLITest(unittest.TestCase):
    def setUp(self) -> None:
        rules_path = Path(__file__).resolve().parents[1] / "examples" / "zria_rules.json"
        self.runtime = build_runtime_dependencies(
            parse_module=_ParseModule(),
            code_module=_CodeModule(),
            eval_module=_EvalModule(),
            thinking_parse_module=_ThinkingParseModule(),
            thinking_code_module=_ThinkingCodeModule(),
            thinking_eval_module=_ThinkingEvalModule(),
            zria_adapter=RuleBasedZRIAAdapter.from_rules_path(rules_path),
            audit_recorder=None,
        )

    def test_branches_lists_configured_reasoning_branches(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(["branches"], runtime=self.runtime)

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["available"], ["non_thinking", "thinking"])

    def test_infer_uses_requested_reasoning_branch(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = main(
                [
                    "infer",
                    "--query",
                    "Need the deeper reasoning path.",
                    "--mode",
                    "pyro",
                    "--reasoning-branch",
                    "thinking",
                ],
                runtime=self.runtime,
            )

        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["reasoning_branch"], "thinking")
        self.assertEqual(payload["pyro_result"]["answer"]["branch"], "thinking")

    def test_thinking_ingest_builds_specialist_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thinking_records.jsonl"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "thinking-ingest",
                        "--input",
                        str(Path(__file__).resolve().parents[1] / "examples" / "thinking_cases_eval.jsonl"),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["records_written"], 3)
            self.assertTrue(output_path.exists())

    def test_synthetic_ingest_builds_clara_style_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "synthetic_records.jsonl"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "synthetic-ingest",
                        "--input",
                        str(Path(__file__).resolve().parents[1] / "examples" / "thinking_cases_eval.jsonl"),
                        "--output",
                        str(output_path),
                        "--case-type",
                        "thinking",
                    ]
                )

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["records_written"], 6)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
