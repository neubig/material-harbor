import json
import unittest
from pathlib import Path

from bioreason.verify import grade, load_ontology

ROOT = Path(__file__).resolve().parents[1]


class BioReasonGoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = sorted((ROOT / "bioreason/tasks-go100").iterdir())[0]
        cls.gold = json.loads((cls.task / "tests/gold.json").read_text())
        cls.ontology = cls.task / "tests/go-basic.obo"
        cls.parents, cls.aliases, cls.obsolete = load_ontology(cls.ontology)

    def test_oracle_and_equivalent_leaf_closure(self):
        expected = set().union(*(set(value) for value in self.gold["aspects"].values()))
        self.assertEqual(grade({"go_ids": sorted(expected)}, self.gold, self.ontology)["reward"], 1)
        ancestors = set().union(*(self.parents.get(term, set()) for term in expected))
        leaves = expected - ancestors
        self.assertEqual(grade({"go_ids": sorted(leaves)}, self.gold, self.ontology)["reward"], 1)

    def test_malformed_unknown_missing_and_extra_fail(self):
        expected = set().union(*(set(value) for value in self.gold["aspects"].values()))
        self.assertEqual(grade("not json", self.gold, self.ontology)["reward"], 0)
        self.assertEqual(grade({"go_ids": ["GO:9999999"]}, self.gold, self.ontology)["reward"], 0)
        leaf = next(term for term in expected if not any(term in self.parents.get(other, ()) for other in expected))
        self.assertEqual(grade({"go_ids": sorted(expected - {leaf})}, self.gold, self.ontology)["reward"], 0)
        roots = {"bp": "GO:0008150", "mf": "GO:0003674", "cc": "GO:0005575"}
        labeled_roots = {root for aspect, root in roots.items() if self.gold["aspects"][aspect]}
        extra = next(term for term in self.parents if term not in expected and any(root in self._closure(term) for root in labeled_roots))
        self.assertEqual(grade({"go_ids": sorted(expected | {extra})}, self.gold, self.ontology)["reward"], 0)

    def _closure(self, term):
        out, stack = set(), [term]
        while stack:
            current = stack.pop()
            if current not in out:
                out.add(current)
                stack.extend(self.parents.get(current, ()))
        return out

    def test_manifest_deduplication_and_label_isolation(self):
        manifest = json.loads((ROOT / "qualification/bioreason-go-manifest.json").read_text())
        self.assertEqual(manifest["count"], 100)
        self.assertGreater(manifest["deduplicated_clean_population"], 50)
        self.assertEqual(len({task["sequence_sha256"] for task in manifest["tasks"]}), 100)
        for item in manifest["tasks"]:
            task = ROOT / "bioreason/tasks-go100" / item["task_id"]
            case = json.loads((task / "environment/data/case.json").read_text())
            self.assertEqual(set(case), set(manifest["input_allowlist"]))
            self.assertFalse(set(case) & set(manifest["verifier_only"]))
            self.assertNotIn(item["source_protein_id_sha256"], json.dumps(case))


if __name__ == "__main__":
    unittest.main()
