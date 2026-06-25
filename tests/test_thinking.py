from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.thinking import (
    build_thinking_records_for_case,
    ingest_thinking_cases,
    load_thinking_cases,
)


class ThinkingIngestTest(unittest.TestCase):
    def test_build_thinking_records_marks_branch_and_uses_specialist_prompt(self) -> None:
        cases = load_thinking_cases(
            Path(__file__).resolve().parents[1] / "examples" / "thinking_cases_train.jsonl"
        )
        records = build_thinking_records_for_case(cases[0])

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0].metadata["reasoning_branch"], "thinking")
        self.assertEqual(records[0].metadata["reasoning_style"], "specialist")
        self.assertIn("specialist thinking branch", records[0].messages[0].content.lower())
        self.assertIn("Reasoning goal:", records[0].messages[1].content)

    def test_ingest_thinking_cases_writes_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "thinking_records.jsonl"
            count = ingest_thinking_cases(
                Path(__file__).resolve().parents[1] / "examples" / "thinking_cases_eval.jsonl",
                output_path,
            )

            self.assertEqual(count, 3)
            lines = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn('"reasoning_branch":"thinking"', lines[0])


if __name__ == "__main__":
    unittest.main()
