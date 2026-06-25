from pathlib import Path
import sys
import unittest

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from msa_zria.config import KGConfig


class KGConfigTest(unittest.TestCase):
    def test_oxigraph_rejects_wwkg_scope_fields(self) -> None:
        with self.assertRaises(ValidationError):
            KGConfig(
                backend="oxigraph",
                graph_path="data/domain_graph.nq",
                workspace="urn:wwkg:workspace:test",
                branch="main",
            )

    def test_oxigraph_metadata_excludes_wwkg_scope(self) -> None:
        config = KGConfig(
            backend="oxigraph",
            graph_path="data/domain_graph.nq",
        )
        self.assertIsNone(config.effective_scope())
        self.assertEqual(config.to_metadata(), {"kg_backend": "oxigraph"})

    def test_wwkg_rejects_graph_path(self) -> None:
        with self.assertRaises(ValidationError):
            KGConfig(
                backend="wwkg",
                base_url="http://127.0.0.1:4242",
                graph_path="data/domain_graph.nq",
            )

    def test_wwkg_scope_is_retained(self) -> None:
        config = KGConfig(
            backend="wwkg",
            base_url="http://127.0.0.1:4242",
            workspace="urn:wwkg:workspace:test",
            branch="main",
        )
        self.assertEqual(
            config.to_metadata(),
            {
                "kg_workspace": "urn:wwkg:workspace:test",
                "kg_branch": "main",
                "kg_backend": "wwkg",
            },
        )


if __name__ == "__main__":
    unittest.main()
