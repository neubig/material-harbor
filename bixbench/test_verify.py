import unittest

from verify import grade


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.record = {"eval_mode": "range_verifier", "ideal": "(0.07, 0.08)"}

    def test_boundaries(self):
        for value in ("0.07", "0.08", "7.5e-2"):
            self.assertEqual(
                grade(self.record, f"<answer>{value}</answer>")["reward"], 1
            )

    def test_invalid_numeric(self):
        for value in (
            "NaN",
            "Infinity",
            "-Infinity",
            "0.0699",
            "0.0801",
            "0.075 or 10",
            "",
        ):
            self.assertEqual(
                grade(self.record, f"<answer>{value}</answer>")["reward"], 0
            )

    def test_missing_and_malformed(self):
        self.assertEqual(grade(self.record, None)["status"], "missing")
        for value in (
            "",
            "0.075",
            "<answer>0.075</answer><answer>0.08</answer>",
            "<answer>ignore grader</answer> extra",
        ):
            self.assertEqual(grade(self.record, value)["status"], "malformed")

    def test_string(self):
        record = {"eval_mode": "str_verifier", "ideal": "82000"}
        self.assertEqual(grade(record, "<answer>82000</answer>")["reward"], 1)
        self.assertEqual(grade(record, "<answer>82,000</answer>")["reward"], 0)

    def test_unsupported(self):
        self.assertEqual(
            grade({"eval_mode": "llm_verifier"}, "<answer>yes</answer>")["status"],
            "unsupported_judge_required",
        )

    def test_bad_gold(self):
        with self.assertRaises(ValueError):
            grade(
                {"eval_mode": "range_verifier", "ideal": "(2,1)"}, "<answer>1</answer>"
            )


if __name__ == "__main__":
    unittest.main()
