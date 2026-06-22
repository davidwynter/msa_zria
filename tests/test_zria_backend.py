from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.config import KGScope
from msa_zria.data import ParseTarget
from msa_zria.zria_adapter import RuleBasedZRIAAdapter


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


if __name__ == "__main__":
    unittest.main()
