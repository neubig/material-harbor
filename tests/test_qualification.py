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

    def test_csmbench_fails_accuracy_not_foundation_model_gate(self):
        verdict = json.loads((ROOT / "qualification/csmbench-qualification-verdict.json").read_text())
        self.assertEqual(verdict["gates"]["relevant_domain_foundation_model"]["verdict"], "pass_with_task_fit_caveat")
        self.assertEqual(verdict["gates"]["deepseek_v4_1_flash_binary_accuracy"]["verdict"], "fail")
        self.assertEqual(verdict["overall"], "fail_deepseek_binary_accuracy_gate")

    def test_matmech_failure_does_not_assert_foundation_model_absence(self):
        report = json.loads((ROOT / "qualification/structural-disqualifications.json").read_text())
        matmech = report["candidates"]["MatMech"]
        self.assertEqual(matmech["verdict"], "fail")
        self.assertEqual(matmech["countable_tasks"], 0)
        self.assertNotIn("foundation model", matmech["basis"].lower())




    def test_matcha_resume_preserves_frozen_cohort_without_reruns(self):
        original = set(json.loads((ROOT / "matcha/manifest100.json").read_text())["task_ids"])
        resume = json.loads((ROOT / "qualification/matcha-resume-manifest.json").read_text())
        completed = set(resume["retained_completed_task_ids"])
        remaining = set(resume["remaining_task_ids"])
        self.assertEqual(len(completed), 23)
        self.assertEqual(len(remaining), 77)
        self.assertFalse(completed & remaining)
        self.assertEqual(completed | remaining, original)


    def test_matcha_final_accuracy_uses_all_scheduled_attempts(self):
        report = json.loads((ROOT / "qualification/matcha-accuracy-report.json").read_text())
        self.assertEqual(report["scheduled"], 100)
        self.assertEqual(report["attempted"], 100)
        self.assertEqual(report["successes"], 70)
        self.assertEqual(report["failures_including_missing"], 30)
        self.assertEqual(report["missing_rewards"], 0)
        self.assertEqual(report["accuracy"], 0.70)
        self.assertEqual(report["strict_gate"]["observed_sample_point"], "not_met_at_excluded_upper_boundary")
        self.assertEqual(report["strict_gate"]["population_verdict"], "unresolved_expand_to_precommitted_n300")
        self.assertLess(report["intervals"]["wilson"][0], 0.70)
        self.assertGreater(report["intervals"]["wilson"][1], 0.70)
        self.assertEqual(len({row["task_id"] for row in report["rows"]}), 100)

    def test_matcha_n300_expansion_is_frozen_and_disjoint(self):
        plan = json.loads((ROOT / "qualification/matcha-n300-manifest.json").read_text())
        first = plan["first100_task_ids"]
        additional = plan["next200_task_ids"]
        cumulative = plan["cumulative_n300_task_ids"]
        self.assertEqual(first, json.loads((ROOT / "matcha/manifest100.json").read_text())["task_ids"])
        self.assertEqual(len(first), 100)
        self.assertEqual(len(additional), 200)
        self.assertFalse(set(first) & set(additional))
        self.assertEqual(cumulative, first + additional)
        self.assertEqual(len(set(cumulative)), 300)

    def test_omnimat_full_population_audit_uses_task_denominator(self):
        report = json.loads((ROOT / "qualification/omnimat-full-population-fn-audit-report.json").read_text())
        self.assertEqual(report["population"], 142)
        self.assertIn("task-equivalence/adversarial robustness", report["population_scope"])
        self.assertEqual(len(set(report["false_negative"]["task_ids"])), 24)
        self.assertEqual(report["false_negative"]["errors"], 24)
        self.assertGreater(report["false_negative"]["rate"], 0.15)
        self.assertEqual(report["false_negative"]["finite_population_gate"], "fail")
        self.assertEqual(report["false_positive"]["errors"], 0)

    def test_matcha_verifier_audit_keeps_full_cohort_denominator(self):
        report = json.loads((ROOT / "qualification/matcha-scientific-audit-full-cohort-report.json").read_text())
        self.assertEqual(report["completed"], 100)
        self.assertEqual(report["usable_high_confidence_unambiguous"], 94)
        self.assertEqual(report["unresolved_counted_non_error"], 6)
        self.assertEqual(report["false_positive"]["denominator"], 100)
        self.assertEqual(report["false_negative"]["denominator"], 100)
        self.assertEqual(report["false_positive"]["gate"], "preliminary_single_provider_unresolved")
        self.assertEqual(report["false_negative"]["gate"], "preliminary_single_provider_unresolved")
        self.assertEqual(report["overall"], "unresolved_pending_cross_provider_corroboration")

    def test_matcha_cross_provider_audit_remains_inconclusive(self):
        report = json.loads((ROOT / "qualification/matcha-scientific-cross-provider-corroboration-report.json").read_text())
        self.assertEqual(report["false_positive"]["denominator"], 100)
        self.assertEqual(report["false_negative"]["denominator"], 100)
        self.assertEqual(report["false_positive"]["corroborated_errors"], 20)
        self.assertEqual(report["false_negative"]["corroborated_errors"], 17)
        self.assertLess(report["false_positive"]["wilson95"][0], 0.15)
        self.assertGreater(report["false_positive"]["wilson95"][1], 0.15)
        self.assertEqual(report["overall"], "unresolved")

    def test_matcha_cumulative_cross_provider_fp_failure_is_bounded(self):
        report = json.loads((ROOT / "qualification/matcha-scientific-audit-cumulative300-report.json").read_text())
        self.assertEqual(report["false_positive"]["denominator"], 300)
        self.assertEqual(report["false_positive"]["corroborated_errors"], 61)
        self.assertGreater(report["false_positive"]["wilson97_5_two_look"][0], 0.15)
        self.assertGreater(report["false_positive"]["paper_group_bootstrap97_5"][0], 0.15)
        self.assertEqual(report["overall"], "fail_verifier_false_positive_gate")


    def test_bioreason_accuracy_uses_all_scheduled_attempts(self):
        report = json.loads((ROOT / "qualification/bioreason-go-accuracy-report.json").read_text())
        self.assertEqual(report["scheduled"], 100)
        self.assertEqual(report["attempted"], 100)
        self.assertEqual(report["successes"], 0)
        self.assertLess(report["wilson98_333"][1], 0.10)
        self.assertEqual(report["decision"], "fail_below_strict_10_percent_floor")

    def test_matrix_replay_is_containerized_and_missing_is_zero(self):
        report = json.loads((ROOT / "qualification/matrix-accuracy-report.json").read_text())
        self.assertIn("inside its task Dockerfile image", report["execution"])
        self.assertEqual(report["scheduled"], 100)
        self.assertEqual(report["successes"] + report["failures_including_missing"], 100)
        self.assertEqual(report["retained_answer_artifacts"], 86)
        self.assertEqual(report["missing_or_infrastructure_counted_zero"], 14)

    def test_matrix_cross_provider_audit_uses_actual_images(self):
        report = json.loads((ROOT / "qualification/matrix-verifier-cross-provider-report.json").read_text())
        self.assertEqual(report["actual_images_used"], 90)
        self.assertEqual(report["false_negative"]["errors"], 0)
        self.assertEqual(report["false_positive"]["errors"], 0)
        self.assertEqual(report["gate"], "pass")


    def test_matrix_bounded240_fails_exact_image_transport(self):
        report = json.loads((ROOT / "qualification/matrix-bounded240-accuracy-report.json").read_text())
        transport = json.loads((ROOT / "qualification/matrix-transport-failure-audit-report.json").read_text())
        verdict = json.loads((ROOT / "qualification/matrix-qualification-verdict.json").read_text())
        self.assertEqual(report["scheduled"], 240)
        self.assertEqual(report["attempted"], 240)
        self.assertEqual(report["successes"], 84)
        self.assertEqual(report["failures_including_missing"], 156)
        self.assertEqual(report["missing_or_infrastructure_counted_zero"], 98)
        self.assertEqual(report["strict_accuracy_gate"]["decision"], "pass_with_infrastructure_sensitivity")
        self.assertLess(report["retained_answer_sensitivity"]["simultaneous_wilson98_333"][1], 0.70)
        self.assertEqual(transport["scheduled"], 240)
        self.assertEqual(transport["transport_counts"]["exact_original_bytes_in_model_request"], 158)
        self.assertEqual(transport["transport_failures"], 82)
        self.assertEqual(transport["failed_transport_by_missing_cause"], {"pre_agent_infrastructure_failure": 82})
        self.assertEqual(transport["transport_gate"], "fail")
        self.assertEqual(verdict["overall"], "fail_image_transport_gate")
        self.assertEqual(verdict["gates"]["harbor_containerization"]["decision"], "fail")
        self.assertNotIn("rows", json.loads((ROOT / "qualification/matrix-accuracy-report.json").read_text()))
        self.assertNotIn("rows", json.loads((ROOT / "qualification/matrix-accuracy-additional140-report.json").read_text()))









if __name__ == "__main__":
    unittest.main()
