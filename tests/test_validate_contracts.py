from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.ingest import load_customer_support_cases
from msa_zria.validate_contracts import validate_contracts


class ContractValidationTest(unittest.TestCase):
    def test_baseline_contract_validation_runs_end_to_end(self) -> None:
        cases = load_customer_support_cases(
            Path(__file__).resolve().parents[1] / "examples" / "customer_support_cases.jsonl"
        )
        report = validate_contracts(cases, module_source="baseline", timeout_seconds=2.0)
        self.assertEqual(report.summaries.total_cases, len(cases))
        self.assertGreater(report.summaries.parse_valid_cases, 0)
        self.assertGreater(report.summaries.code_valid_cases, 0)
        self.assertGreater(report.summaries.evaluation_valid_cases, 0)


if __name__ == "__main__":
    unittest.main()
