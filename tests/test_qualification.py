import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load("qualification_audit", ROOT / "qualification/audit_scientific_mcqa.py")
accuracy = load("qualification_accuracy", ROOT / "qualification/summarize_accuracy.py")



class QualificationTests(unittest.TestCase):
    def test_csmbench_frozen_cohort_is_image_complete_and_label_isolated(self):
        manifest = json.loads((ROOT / "qualification/csmbench-qualification-manifest.json").read_text())
        tasks = ROOT / "csmbench/tasks/qualification100"
        self.assertEqual(manifest["count"], 100)
        self.assertEqual(len(manifest["indices"]), 100)
        self.assertEqual(len(set(manifest["indices"])), 100)
        image_hashes = set()
        question_hashes = set()
        for index in manifest["indices"]:
            task = tasks / f"csmbench-mcqa-{index:04d}"
            self.assertTrue((task / "environment/data/image.jpg").is_file())
            image_hashes.add(hashlib.sha256((task / "environment/data/image.jpg").read_bytes()).hexdigest())
            question_hashes.add(hashlib.sha256((task / "environment/data/question.json").read_bytes()).hexdigest())
            self.assertTrue((task / "tests/label.json").is_file())
            self.assertNotIn("correct_answer", (task / "instruction.md").read_text())
            self.assertFalse((task / "environment/data/label.json").exists())
        self.assertEqual(len(image_hashes), 100)
        self.assertEqual(len(question_hashes), 100)

    def test_manifest_commits_to_but_does_not_reveal_labels(self):
        for name in ("matcha", "csmbench"):
            raw = (ROOT / f"qualification/{name}-scientific-audit-manifest.json").read_text()
            manifest = json.loads(raw)
            self.assertNotIn('"gold"', raw)
            self.assertNotIn('"answer"', raw)
            self.assertEqual(len(manifest["task_ids"]), 30)
            self.assertTrue(all("gold_commitment" in value for value in manifest["task_hashes"].values()))

    def test_fp_and_fn_are_separate_option_denominators(self):
        rows = [{
            "task_id": "x",
            "valid_options": ["B", "C"],
            "ambiguous": False,
            "confidence": "high",
            "gold": "B",
            "options": list("ABCD"),
        }]
        result = audit.report(rows, 1)
        self.assertEqual(result["false_positive"]["denominator"], 2)
        self.assertEqual(result["false_positive"]["errors"], 0)
        self.assertEqual(result["false_negative"]["denominator"], 2)
        self.assertEqual(result["false_negative"]["errors"], 1)

    def test_accuracy_summary_keeps_missing_attempts_as_zero(self):
        rows = json.loads((ROOT / "csmbench/manifest.json").read_text())["rows"][:100]
        rewards = {}
        for position, row in enumerate(rows):
            task_id = f"csmbench-mcqa-{row['index']:04d}"
            rewards[task_id] = {
                "task_id": f"/tasks/{task_id}",
                "reward": None if position < 10 else 1,
            }
        report = {
            "staged_tasks": 100,
            "model": "openai/deepseek-v4.1-flash",
            "rewards": rewards,
            "transport": {},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text(json.dumps(report))
            result = accuracy.summarize("csmbench", path)
        self.assertEqual(result["attempted"], 100)
        self.assertEqual(result["missing_rewards_counted_zero"], 10)
        self.assertEqual(result["successes"], 90)
        self.assertEqual(result["point_estimate"], 0.9)
        self.assertEqual(result["classification"], "fail_outside_band")



if __name__ == "__main__":
    unittest.main()
