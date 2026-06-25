from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.thinking_dspy_modules import ThinkingCodeGenModule, ThinkingEvalModule, ThinkingParseModule


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No more fake responses were configured.")
        return self.responses.pop(0)


class ThinkingDSPyModuleContractTest(unittest.TestCase):
    def test_parse_module_uses_specialist_prompt_and_validates_json_contract(self) -> None:
        llm = _FakeLLM(
            [
                '{"task":"parse","device":"Router123","issue":"Overheating","cause":"Blocked ventilation","severity":"high"}'
            ]
        )
        module = ThinkingParseModule(llm)
        result = module("Router123 is overheating")
        self.assertEqual(result["parsed_result"]["task"], "parse")
        self.assertIn("specialist thinking branch", llm.prompts[0].lower())

    def test_code_module_validates_json_contract(self) -> None:
        module = ThinkingCodeGenModule(
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

    def test_eval_module_validates_json_contract(self) -> None:
        module = ThinkingEvalModule(
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
