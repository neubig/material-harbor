import hashlib
import tempfile
import unittest
from pathlib import Path

from main import generate


class GenerationTests(unittest.TestCase):
    def test_real_capsule_excludes_executed_notebook(self):
        root = Path(__file__).resolve().parents[2] / "research-bixbench-continuation"
        capsule = root / "probe-capsule.zip"
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            generate(
                root / "BixBench.jsonl",
                capsule,
                Path(tmp),
                ["bix-18-q1"],
                hashlib.sha256(capsule.read_bytes()).hexdigest(),
            )
            task = Path(tmp) / "bix-18-q1"
            self.assertEqual(
                sorted(p.name for p in (task / "environment/data").iterdir()),
                ["Swarm_1.csv"],
            )
            self.assertFalse(list((task / "environment").rglob("*.ipynb")))
            self.assertFalse(list((task / "environment").rglob("gold.json")))
            self.assertTrue((task / "tests/gold.json").exists())

    def test_wrong_hash_rejected(self):
        with self.assertRaises(ValueError):
            generate(Path(__file__), Path(__file__), Path("/unused"), [], "wrong")


if __name__ == "__main__":
    unittest.main()
