import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import numpy as np

from scripts.summarize_accuracy import bootstrap_ci, classify, summarize


# Extracted fields from a real Harbor SDK DeepSeek result.json, not a mocked runner.
REAL_RESULT = {
    "id": "9f839e41-98a4-49d7-8fec-7d8354334ae1",
    "task_name": "omnimatbench/cal-10-022",
    "task_id": {"path": "/workspace/task"},
    "task_checksum": "0249b11c64483501a62293ef312a85e1a91a3e46dca46731e8ff2175ae373ec1",
    "config": {
        "task": {"path": "/workspace/task"}, "trial_name": "smoke",
        "trials_dir": "/workspace/trials", "timeout_multiplier": 1.0,
        "agent": {"name": "openhands-sdk", "model_name": "openai/deepseek-v4-flash",
                  "kwargs": {"load_skills": False, "max_iterations": 20, "temperature": 0}},
        "environment": {"type": "modal"}, "verifier": {"disable": False},
    },
    "agent_info": {"name": "openhands-sdk", "version": "1.50.1",
                   "model_info": {"name": "deepseek-v4-flash", "provider": "openai"}},
    "verifier_result": {"rewards": {"reward": 0.0}},
    "exception_info": None, "finished_at": "2026-10-01T14:55:41.688851Z",
}


class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self, rewards):
        tasks = []
        for i, reward in enumerate(rewards):
            result = copy.deepcopy(REAL_RESULT)
            result.update(id=f"trial-{i}", task_name=f"task-{i}")
            result["verifier_result"]["rewards"]["reward"] = reward
            (self.root / f"{i}.json").write_text(json.dumps(result))
            tasks.append({"task_id": f"task-{i}", "result_path": f"{i}.json", "paper_id": f"paper-{i % 3}"})
        return {"sampling_plan": "Fixed sample, no outcome-dependent stopping",
                "expected_config": REAL_RESULT["config"],
                "expected_agent_info": REAL_RESULT["agent_info"], "tasks": tasks}

    def change(self, i, edit):
        path = self.root / f"{i}.json"
        data = json.loads(path.read_text())
        edit(data)
        path.write_text(json.dumps(data))

    def test_deterministic_complete(self):
        m = self.manifest([0, 1] * 100)
        a = summarize(m, self.root)
        self.assertEqual(a, summarize(m, self.root))
        self.assertEqual(a["classification"], "within")
        self.assertEqual(a["point_estimate"], .5)
        self.assertEqual(a["denominators"]["expected"], 200)
        self.assertEqual(a["protocol"]["resamples"], 100000)

    def test_missing_exception_and_unfinished_never_omitted(self):
        m = self.manifest([0, 1, 0, 1])
        (self.root / '0.json').unlink()
        self.change(1, lambda r: r.update(exception_info={"exception_type": "RuntimeError", "exception_message": "Decimal verifier failure"}))
        self.change(2, lambda r: r.update(finished_at=None))
        r = summarize(m, self.root)
        self.assertEqual(r["classification"], "incomplete")
        self.assertEqual(r["denominators"], {"expected": 4, "observed_binary": 1, "successes": 1, "reward_failures": 0, "unresolved": 3})
        self.assertIsNone(r["point_estimate"])
        self.assertIsNone(r["bootstrap_ci"])
        self.assertEqual(len(r["trials"]), 4)

    def test_duplicates_error(self):
        m = self.manifest([0, 1])
        for field in ("task_id", "result_path"):
            bad = copy.deepcopy(m)
            bad["tasks"][1][field] = bad["tasks"][0][field]
            with self.assertRaises(ValueError):
                summarize(bad, self.root)
        self.change(1, lambda r: r.update(id="trial-0"))
        with self.assertRaises(ValueError):
            summarize(m, self.root)

    def test_configuration_identity_mismatches(self):
        for edit in (lambda r: r["config"]["agent"].update(model_name="other"),
                     lambda r: r["config"]["agent"]["kwargs"].update(temperature=1),
                     lambda r: r["agent_info"].update(name="oracle"),
                     lambda r: r.update(task_name="wrong")):
            m = self.manifest([0, 1])
            self.change(1, edit)
            r = summarize(m, self.root)
            self.assertEqual(r["classification"], "incomplete")
            self.assertTrue(r["trials"][1]["issues"])

    def test_invalid_rewards_and_json(self):
        for value in (.5, None, True, "1"):
            self.assertEqual(summarize(self.manifest([0, value]), self.root)["classification"], "incomplete")
        m = self.manifest([0])
        (self.root / '0.json').write_text('{')
        self.assertEqual(summarize(m, self.root)["classification"], "incomplete")

    def test_strict_boundaries(self):
        for ci, expected in [((.1, .6), 'inconclusive'), ((.2, .7), 'inconclusive'),
                             ((0, .1), 'inconclusive'), ((.7, 1), 'inconclusive'),
                             ((.11, .69), 'within'), ((0, .09), 'outside'), ((.71, 1), 'outside')]:
            self.assertEqual(classify(ci), expected)

    def test_degenerate_guard(self):
        for reward in (0, 1):
            r = summarize(self.manifest([reward]), self.root)
            self.assertEqual(r["bootstrap_ci"], [reward, reward])
            self.assertEqual(r["classification"], "inconclusive")
            self.assertTrue(r["warnings"])

    def test_cluster_ratio_not_mean(self):
        rewards = [1] + [0] * 9 + [1] * 20
        papers = ['a'] + ['b'] * 9 + ['c'] * 20
        rng = np.random.default_rng(5)
        counts = rng.multinomial(3, [1/3]*3, size=100000)
        expected = np.quantile((counts @ [1, 0, 20]) / (counts @ [1, 9, 20]), [.025, .975]).tolist()
        self.assertEqual(bootstrap_ci(rewards, papers, seed=5), expected)
        m = self.manifest([0, 1] * 10)
        self.assertEqual(summarize(m, self.root, scope='paper')["protocol"]["scope"], 'paper')
        del m['tasks'][0]['paper_id']
        with self.assertRaises(ValueError):
            summarize(m, self.root, scope='paper')

    def test_external_native_validation_blocks_reward(self):
        m = self.manifest([0, 1])
        m['tasks'][0]['validation_issues'] = ['native_decimal_exception']
        r = summarize(m, self.root)
        self.assertEqual(r['classification'], 'incomplete')
        self.assertEqual(r['unknown_outcome_bounds'], [.5, 1.])
        self.assertEqual(r['denominators']['unresolved'], 1)

    def test_cli_reproducible_json(self):
        manifest = self.root / 'manifest.json'
        manifest.write_text(json.dumps(self.manifest([0, 1] * 10)))
        script = Path(__file__).resolve().parents[1] / 'scripts/summarize_accuracy.py'
        command = [sys.executable, str(script), str(manifest)]
        first = subprocess.check_output(command)
        self.assertEqual(first, subprocess.check_output(command))
        self.assertEqual(json.loads(first)['denominators']['expected'], 20)

    def test_empty_or_unfrozen_manifest_error(self):
        with self.assertRaises(ValueError):
            summarize(self.manifest([]), self.root)
        m = self.manifest([0])
        del m['expected_config']
        with self.assertRaises(ValueError):
            summarize(m, self.root)


if __name__ == '__main__':
    unittest.main()
