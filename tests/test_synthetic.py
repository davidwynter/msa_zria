from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.synthetic import build_synthetic_records, write_synthetic_records


class SyntheticRecordTest(unittest.TestCase):
    def test_build_synthetic_records_for_thinking_cases(self) -> None:
        records = build_synthetic_records(
            Path(__file__).resolve().parents[1] / "examples" / "thinking_cases_eval.jsonl",
            case_type="thinking",
        )
        self.assertEqual(len(records), 6)
        variants = {record.metadata["augmentation_variant"] for record in records}
        self.assertEqual(variants, {"paraphrase", "qa"})

    def test_write_synthetic_records_for_support_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "synthetic.jsonl"
            count = write_synthetic_records(
                Path(__file__).resolve().parents[1] / "examples" / "customer_support_cases.jsonl",
                output_path,
                case_type="support",
            )
            self.assertGreater(count, 0)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
