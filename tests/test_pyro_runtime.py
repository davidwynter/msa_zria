from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.pyro_runtime import execute_pyro_program


class PyroRuntimeTest(unittest.TestCase):
    def test_executes_safe_program_in_controlled_runner(self) -> None:
        result = execute_pyro_program(
            "def run_inference():\n"
            "    payload = {'recommended_action': 'restart', 'resolved': True}\n"
            "    print('ok')\n"
            "    return payload\n"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.answer["recommended_action"], "restart")
        self.assertEqual(result.stdout.strip(), "ok")

    def test_rejects_disallowed_imports(self) -> None:
        result = execute_pyro_program(
            "import os\n"
            "def run_inference():\n"
            "    return os.listdir('.')\n"
        )
        self.assertFalse(result.success)
        self.assertIn("Disallowed import", result.error or "")
        self.assertEqual(result.control_events[0]["control_type"], "disallowed_import")

    def test_rejects_unbounded_loop_syntax(self) -> None:
        result = execute_pyro_program(
            "def run_inference():\n"
            "    while True:\n"
            "        pass\n"
        )
        self.assertFalse(result.success)
        self.assertIn("Disallowed syntax", result.error or "")
        self.assertEqual(result.control_events[0]["control_type"], "disallowed_ast_syntax")


if __name__ == "__main__":
    unittest.main()
