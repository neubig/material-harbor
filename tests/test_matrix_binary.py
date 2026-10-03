import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MatrixBinaryTests(unittest.TestCase):
    def test_full_credit_only_binary_mapping(self):
        verifier = load(ROOT / "matrix/verify_binary.py", "matrix_verify_binary")
        for score in (0, 0.25, 0.5, 0.75):
            result = verifier.binaryize({"reward": score})
            self.assertEqual(result, {"reward": 0, "graded_reward": score})
        self.assertEqual(verifier.binaryize({"reward": 1}), {"reward": 1, "graded_reward": 1})
        self.assertEqual(verifier.binaryize({"reward": None}), {"reward": None})

    def test_frozen_cohort_is_image_complete_and_excludes_pilot(self):
        manifest = json.loads((ROOT / "qualification/matrix-binary-vision-manifest.json").read_text())
        self.assertEqual(len(manifest["tasks"]), 100)
        qids = [row["qid"] for row in manifest["tasks"]]
        self.assertEqual(len(set(qids)), 100)
        self.assertFalse(set(qids) & set(manifest["excluded_prior_pilot_qids"]))
        self.assertEqual(manifest["train_validation_exact_prompt_overlap"], 0)
        for item in manifest["tasks"]:
            task = ROOT / "matrix/tasks-binary-vision100" / item["task_id"]
            image = task / "environment/data/image.png"
            self.assertEqual(hashlib.sha256(image.read_bytes()).hexdigest(), item["image_sha256"])
            self.assertIn("matrix-full-credit-binary-v1", (task / "task.toml").read_text())
            self.assertEqual([path.name for path in (task / "environment/data").iterdir()], ["image.png"])


if __name__ == "__main__":
    unittest.main()
