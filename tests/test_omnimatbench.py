import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'omnimatbench'


def load(name):
    spec = importlib.util.spec_from_file_location('omnimat_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verify = load('verify')

    def test_vendored_source_hashes(self):
        hashes = json.loads((ROOT / 'source/sha256.json').read_text())
        for name in ['LICENSE', 'scripts/cal/eval_cal_results.py', 'scripts/cal/omnimat_paths.py']:
            self.assertEqual(hashlib.sha256((ROOT / 'source' / name).read_bytes()).hexdigest(), hashes[name])

    def test_malformed_answers(self):
        for raw in ['', 'null', '{}', '[true]', '[null]', '[""]', 'NaN', '[NaN]', '[Infinity]', '<answer>["1"]</answer>', '```json\n["1"]\n```', '["1"] trailing']:
            self.assertEqual(self.verify.grade(raw, ['1'])['reward'], 0, raw)

    def test_native_caveats(self):
        native = self.verify.native
        self.assertEqual(native.numeric_equal_exact('100', '100.4'), 1)
        self.assertEqual(native.numeric_equal_exact('1.2e-3', '0.0012'), 0)
        self.assertEqual(native.parse_prediction_string(r'\boxed{1} ["1", "2"]'), ['1'])
        self.assertEqual(self.verify.grade('["0.0012"]', ['1.2e-3'])['reward'], 0)
        self.assertEqual(self.verify.grade('["1.2e-3"]', ['1.2e-3'])['reward'], 1)
        self.assertEqual(self.verify.grade('["1e999999"]', ['1'])['reward'], 0)


class OmniMatBenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load('main')
        cls.verify = load('verify')
        if not (ROOT / 'source/cal').exists():
            raise unittest.SkipTest('Fetch fixtures: python -m omnimatbench.fetch_source')

    def require_cohort(self):
        try:
            self.rows = self.adapter.load_cohort()[0]
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

    def test_offline_fetch_and_hash_rejection(self):
        from omnimatbench.fetch_source import fetch
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)
            fetch(destination, (ROOT / 'source').as_uri() + '/')
            self.assertEqual((destination / 'LICENSE').read_bytes(), (ROOT / 'source/LICENSE').read_bytes())
            (destination / 'LICENSE').write_text('corrupt')
            with self.assertRaises(ValueError):
                fetch(destination, destination.as_uri() + '/')

    def test_cohort_namespace_and_source(self):
        self.require_cohort()
        self.assertEqual(len(self.rows), 360)
        self.assertEqual(len({r['key'] for r in self.rows}), 360)
        self.assertTrue(all(not r['row']['image_url'] for r in self.rows))
        self.adapter.validate_sources()

    def test_pinned_answers_conjunctive(self):
        self.require_cohort()
        for key in ['cal/01/001', 'cal/13/003']:
            gold = next(r['row']['final_answer_list'] for r in self.rows if r['key'] == key)
            self.assertEqual(self.verify.grade(json.dumps(gold), gold)['reward'], 1)
            self.assertEqual(self.verify.grade(json.dumps(gold[:-1]), gold)['reward'], 0)
            self.assertEqual(self.verify.grade(json.dumps(gold + ['extra']), gold)['reward'], 0)
        for row in self.rows:
            gold = row['row']['final_answer_list']
            result = self.verify.grade(json.dumps(gold), gold)
            if row['key'] == 'cal/11/018':
                self.assertEqual(result, {'reward': 0, 'status': 'native_exception', 'exception': 'InvalidOperation'})
            else:
                self.assertEqual(result['reward'], 1, row['key'])

    @unittest.skipUnless(importlib.util.find_spec('harbor'), 'Install the test extra on Python 3.12+ for Harbor schema validation')
    def test_harbor_schema(self):
        from harbor.models.task.config import TaskConfig
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'tasks'
            self.adapter.generate(out, 1)
            config = next(out.glob('*/task.toml'))
            TaskConfig.model_validate(tomllib.loads(config.read_text()))

    def test_generation_and_verifier(self):
        self.require_cohort()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'pilot'
            manifest = self.adapter.generate(out, 10, 20260814)
            second = Path(tmp) / 'second'
            self.adapter.generate(second, 10, 20260814)
            self.assertEqual((out / 'manifest.json').read_bytes(), (second / 'manifest.json').read_bytes())
            self.assertEqual(len(manifest['cohort']), 360)
            tasks = sorted(out.glob('omnimatbench-*'))
            self.assertEqual(len(tasks), 10)
            for task in tasks:
                self.assertEqual(tomllib.loads((task / 'task.toml').read_text())['schema_version'], '1.0')
                self.assertEqual((task / 'tests/native/LICENSE').read_bytes(),
                                 (ROOT / 'source/LICENSE').read_bytes())
                case = json.loads((task / 'environment/case.json').read_text())
                self.assertEqual(set(case), {'question', 'final_answer_format'})
                self.assertEqual(sorted(p.name for p in (task / 'environment').iterdir()), ['Dockerfile', 'case.json'])
                gold = json.loads((task / 'tests/gold.json').read_text())
                answer = Path(tmp) / 'answer.json'
                reward = Path(tmp) / 'reward.txt'
                answer.write_text(json.dumps(gold))
                cmd = [sys.executable, str(task / 'tests/verify.py'), '--answer', str(answer), '--gold', str(task / 'tests/gold.json'), '--reward', str(reward)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(reward.read_text().strip(), '1')
                answer.unlink()
                subprocess.run(cmd, check=True, capture_output=True)
                self.assertEqual(reward.read_text().strip(), '0')
            with self.assertRaises(FileExistsError):
                self.adapter.generate(out, 10, 20260814)



class OmniMatBenchVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from omnimatbench import build_cal_vision
        cls.builder = build_cal_vision
        cls.tasks = ROOT / 'tasks/cal-vision-qualification100'
        if not cls.tasks.exists():
            raise unittest.SkipTest('Build fixtures: python -m omnimatbench.build_cal_vision')

    def test_frozen_image_cohort_is_complete_distinct_and_label_isolated(self):
        manifest = json.loads((ROOT.parent / 'qualification/omnimat-cal-vision-manifest.json').read_text())
        self.assertEqual(manifest['population'], 142)
        self.assertEqual(manifest['count'], 100)
        self.assertEqual(len({item['key'] for item in manifest['selected']}), 100)
        self.assertEqual(len({item['question_sha256'] for item in manifest['selected']}), 100)
        self.assertEqual(len({item['image_sha256'] for item in manifest['selected']}), 100)
        for item in manifest['selected']:
            task = self.tasks / ('omnimatbench-' + item['key'].replace('/', '-'))
            images = [path for path in (task / 'environment/data').iterdir() if path.name != 'case.json']
            self.assertEqual(len(images), 1)
            self.assertEqual(hashlib.sha256(images[0].read_bytes()).hexdigest(), item['image_sha256'])
            self.assertEqual(set(json.loads((task / 'environment/data/case.json').read_text())),
                             {'question', 'final_answer_format'})
            self.assertTrue((task / 'tests/gold.json').is_file())
            self.assertNotIn('final_answer_list', (task / 'instruction.md').read_text())

    @unittest.skipUnless(importlib.util.find_spec('harbor'), 'Install Harbor for schema validation')
    def test_all_frozen_image_tasks_validate_as_harbor(self):
        from harbor.models.task.config import TaskConfig
        for task in self.tasks.iterdir():
            TaskConfig.model_validate(tomllib.loads((task / 'task.toml').read_text()))

    def test_full_population_verifier_audit_rates_are_separate(self):
        native = json.loads((ROOT.parent / 'qualification/omnimat-verifier-audit-native-v1-report.json').read_text())
        self.assertEqual(native['false_positive']['denominator'], 693)
        self.assertEqual(native['false_positive']['errors'], 0)
        self.assertEqual(native['false_negative']['denominator'], 181)
        self.assertEqual(native['false_negative']['errors'], 22)
        repaired = json.loads((ROOT.parent / 'qualification/omnimat-verifier-audit-adapter-v2-report.json').read_text())
        self.assertFalse(repaired['official_exact'])
        self.assertIn('not an independent source-label', repaired['scope_caveat'])
        self.assertEqual(repaired['false_positive']['errors'], 0)
        self.assertEqual(repaired['false_negative']['errors'], 0)
        regrade = json.loads((ROOT.parent / 'qualification/omnimat-retained-dual-regrade-report.json').read_text())
        self.assertTrue(regrade['native_v1']['matches_original_harbor_rewards'])
        self.assertEqual(regrade['native_v1']['correct'], 26)
        self.assertEqual(regrade['adapter_v2']['correct'], 26)
        transport = json.loads((ROOT.parent / 'qualification/omnimat-transport-audit-report.json').read_text())
        self.assertEqual(transport['image_enabled_attempts'], 100)
        self.assertTrue(all(task['image_use_proven'] for task in transport['tasks']))
        self.assertFalse(transport['rerun_required'])



if __name__ == '__main__':
    unittest.main()
