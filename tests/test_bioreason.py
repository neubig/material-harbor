import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "bioreason-vep"
sys.path.insert(0, str(ROOT))
from answer_schema import excluded_rows, parse_source_answer

spec = importlib.util.spec_from_file_location(
    "bioreason_verify", ROOT / "task-template/tests/verify.py"
)
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class BioReasonTests(unittest.TestCase):
    def test_source_formats(self):
        self.assertEqual(
            parse_source_answer(
                "Pathogenic; Microcephaly 1, primary, autosomal recessive"
            )["diseases"],
            ["Microcephaly 1, primary, autosomal recessive"],
        )
        self.assertEqual(
            parse_source_answer("pathogenic; ['Disorder_A', 'Disorder_B']")["diseases"],
            ["Disorder_A", "Disorder_B"],
        )
        self.assertEqual(
            parse_source_answer("Benign"), {"pathogenicity": "benign", "diseases": []}
        )

    def test_missing_annotations(self):
        for raw in [
            "pathogenic",
            "Pathogenic; not provided",
            "Pathogenic; not specified",
        ]:
            self.assertIsNone(parse_source_answer(raw)["diseases"])

    def test_invalid_sources(self):
        for raw in [
            "pathogenic-ish",
            "uncertain",
            "pathogenic; [1]",
            "Benign; disease",
            "pathogenic; []",
        ]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_source_answer(raw)

    def test_all_components_required(self):
        gold = parse_source_answer("pathogenic; ['Disease_A', 'Disease_B']")
        self.assertEqual(
            verify.evaluate(
                {"pathogenicity": "pathogenic", "diseases": ["disease b", "Disease A"]},
                gold,
            ),
            {"reward": 1.0, "format": 1.0, "pathogenicity": 1.0, "diseases": 1.0},
        )
        for diseases in [[], ["Disease A"], ["Disease A", "Disease B", "Disease C"]]:
            result = verify.evaluate(
                {"pathogenicity": "pathogenic", "diseases": diseases}, gold
            )
            self.assertEqual(result["reward"], 0)
            self.assertEqual(result["pathogenicity"], 1)
            self.assertEqual(result["diseases"], 0)

    def test_strict_schema(self):
        for prediction in [
            "benign",
            {"pathogenicity": "not pathogenic", "diseases": []},
            {"pathogenicity": "benign"},
            {"pathogenicity": "benign", "diseases": "none"},
            {"pathogenicity": "benign", "diseases": [], "extra": True},
        ]:
            self.assertEqual(
                verify.evaluate(prediction, parse_source_answer("Benign"))["reward"], 0
            )

    def test_subtype_mismatch(self):
        gold = parse_source_answer("Pathogenic; Disorder type 2")
        self.assertEqual(
            verify.evaluate(
                {"pathogenicity": "pathogenic", "diseases": ["Disorder type 1"]}, gold
            )["reward"],
            0,
        )

    def test_conflicting_duplicate_and_missing_rows_excluded(self):
        rows = [
            {"reference_sequence": "A", "alt": "C", "answer": "Pathogenic; A"},
            {"reference_sequence": "A", "alt": "C", "answer": "Pathogenic; B"},
            {"reference_sequence": "T", "alt": "G", "answer": "Benign"},
            {"reference_sequence": "T", "alt": "G", "answer": "Benign"},
            {"reference_sequence": "G", "alt": "C", "answer": "pathogenic"},
            {"reference_sequence": "C", "alt": "A", "answer": "pathogenic; [, 'X']"},
        ]
        excluded = excluded_rows(rows, "alt")
        self.assertEqual(set(excluded), {0, 1, 3, 4, 5})
        self.assertIn("conflicting", excluded[0])
        self.assertIn("duplicate", excluded[3])

    def test_verifier_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            answer, info, logs = root / "answer.json", root / "info.json", root / "logs"
            info.write_text(json.dumps({"answer": parse_source_answer("Benign")}))
            command = [
                sys.executable,
                str(ROOT / "task-template/tests/verify.py"),
                "--answer-path",
                str(answer),
                "--info-path",
                str(info),
                "--logs-dir",
                str(logs),
            ]
            for content, expected in [
                (None, 0),
                ("not json", 0),
                ('{"pathogenicity":"benign","diseases":[]}', 1),
                (
                    '{"pathogenicity":"pathogenic","pathogenicity":"benign","diseases":[]}',
                    0,
                ),
            ]:
                if content is not None:
                    answer.write_text(content)
                subprocess.run(command, check=True)
                rewards = json.loads((logs / "reward.json").read_text())
                self.assertEqual(rewards["reward"], expected)
                self.assertEqual(float((logs / "reward.txt").read_text()), expected)
                self.assertTrue(all(type(v) in (int, float) for v in rewards.values()))
                self.assertTrue((logs / "details.json").exists())


if __name__ == "__main__":
    unittest.main()
