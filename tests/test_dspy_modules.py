from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.dspy_modules import CodeGenModule, EvalModule, ParseModule


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No more fake responses were configured.")
        return self.responses.pop(0)


class DSPyModuleContractTest(unittest.TestCase):
    def test_parse_module_validates_json_contract(self) -> None:
        llm = _FakeLLM(
            [
                '{"task":"parse","device":"Router123","issue":"Overheating","cause":null,"severity":"high"}'
            ]
        )
        module = ParseModule(
            llm
        )
        result = module("Router123 is overheating", evidence_context="Retrieved evidence:\n1. Router123 has overheating incidents.")
        self.assertEqual(result["parsed_result"]["task"], "parse")
        self.assertEqual(result["parsed_result"]["device"], "Router123")
        self.assertIn("Retrieved evidence", llm.prompts[0])

    def test_code_module_validates_json_contract(self) -> None:
        module = CodeGenModule(
            _FakeLLM(
                [
                    (
                        '{"task":"code","language":"python","framework":"pyro",'
                        '"entrypoint":"run_inference","query_variable":"failure",'
                        '"required_statements":["def run_inference"],'
                        '"program":"def run_inference():\\n    return {\\"resolved\\": true, \\"should_escalate\\": false}\\n"}'
                    )
                ]
            )
        )
        result = module({"task": "parse", "device": "Router123", "issue": "Overheating"})
        self.assertEqual(result["code_str"]["framework"], "pyro")
        self.assertIn("def run_inference", result["code_str"]["program"])

    def test_eval_module_validates_json_contract(self) -> None:
        module = EvalModule(
            _FakeLLM(
                [
                    '{"task":"evaluate","verdict":"resolved","resolved":true,"should_escalate":false,"explanation":"ok"}'
                ]
            )
        )
        result = module("query", {"resolved": True})
        self.assertEqual(result["evaluation"]["task"], "evaluate")
        self.assertTrue(result["evaluation"]["resolved"])


if __name__ == "__main__":
    unittest.main()
