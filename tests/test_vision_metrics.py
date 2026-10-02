import unittest

from harbor_vision.run_benchmark import summarize_rewards


class VisionMetricsTests(unittest.TestCase):
    def test_missing_answers_count_zero(self):
        result = summarize_rewards({'a': {'reward': 1}, 'b': {'reward': None}}, 2)
        self.assertEqual(result['accuracy_percent'], 50)
        self.assertEqual(result['missing_rewards'], 1)

    def test_fractional_scores_are_not_accuracy(self):
        result = summarize_rewards({'a': {'reward': 1}, 'b': {'reward': .75}}, 2)
        self.assertIsNone(result['accuracy_percent'])
        self.assertEqual(result['mean_reward'], .875)

    def test_unobserved_tasks_not_silently_scored(self):
        result = summarize_rewards({'a': {'reward': 1}}, 2)
        self.assertIsNone(result['accuracy_percent'])
        self.assertEqual(result['unobserved_tasks'], 1)

    def test_empty_run(self):
        self.assertIsNone(summarize_rewards({}, 2)['accuracy_percent'])
