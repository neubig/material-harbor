import importlib.util
import json
import hashlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'proteingym-stability'
spec = importlib.util.spec_from_file_location('pg_verify', ROOT / 'verify.py')
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


class VerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory()
        cls.base = Path(cls.workspace.name) / 'frozen'
        source = os.environ.get('PROTEINGYM_SOURCE_DIR')
        if source:
            subprocess.run([sys.executable, str(ROOT / 'main.py'), '--source-dir', source,
                            '--output-dir', str(cls.base)], check=True)
        else:
            spec = importlib.util.spec_from_file_location('pg_builder', ROOT / 'main.py')
            builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(builder)
            builder.OUT = cls.base
            expected = json.loads((ROOT / 'expected-frozen-sha256.json').read_text())
            for record in json.loads((ROOT / 'curation-manifest.json').read_text()):
                if 'tasks/' + record['public']['task_id'] + '/tests/gold.json' in expected:
                    builder.generate(record)

    @classmethod
    def tearDownClass(cls):
        cls.workspace.cleanup()

    def test_all_frozen_labels_and_adversarial_inputs(self):
        manifest = [{'gold': json.loads(p.read_text())} for p in sorted((self.base / 'tasks').glob('*/tests/gold.json'))]
        self.assertEqual(len(manifest), 57)
        checks = 0
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); answer = base / 'answer.json'; gold = base / 'gold.json'
            for record in manifest:
                label = record['gold']['answer']; other = 'B' if label == 'A' else 'A'
                gold.write_text(json.dumps({'answer': label}))
                cases = [(json.dumps({'answer': label}), 1), (json.dumps({'answer': other}), 0),
                         ('{"answer":"'+label+'","answer":"'+label+'"}',0),
                         ('{"answer":"'+label+'","extra":1}',0), ('{}',0), ('[]',0),
                         ('null',0), ('{"answer":true}',0), ('{"answer":["A","B"]}',0),
                         ('{"answer":"a"}',0), ('not json',0), ('x'*1025,0),
                         (json.dumps(label),0), ('{"answer":"A or B"}',0)]
                for text, expected in cases:
                    answer.write_text(text)
                    self.assertEqual(verify.grade(answer,gold),expected)
                    checks += 1
                answer.unlink()
                self.assertEqual(verify.grade(answer,gold),0); checks += 1
                answer.symlink_to(gold)
                self.assertEqual(verify.grade(answer,gold),0); checks += 1
                answer.unlink(); answer.mkdir()
                self.assertEqual(verify.grade(answer,gold),0); checks += 1
                answer.rmdir()
                answer.write_bytes(b'\xff\xfe')
                self.assertEqual(verify.grade(answer,gold),0); checks += 1
                answer.unlink()
        print(f'{checks} real-file verifier assertions across {len(manifest)} tasks')

    def test_no_gold_in_agent_inputs_and_frozen_hashes(self):
        base = self.base
        for name, digest in json.loads((ROOT/'expected-frozen-sha256.json').read_text()).items():
            if name.startswith('tasks/') or os.environ.get('PROTEINGYM_SOURCE_DIR'):
                self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(),digest, name)
        actual_tasks = {str(p.relative_to(base)) for p in (base / 'tasks').rglob('*') if p.is_file()}
        expected_tasks = {name for name in json.loads((ROOT / 'expected-frozen-sha256.json').read_text()) if name.startswith('tasks/')}
        self.assertEqual(actual_tasks, expected_tasks)
        for task in (base/'tasks').iterdir():
            config = tomllib.loads((task/'task.toml').read_text())
            self.assertEqual(config['verifier']['environment_mode'],'separate')
            self.assertEqual(config['environment']['allowed_hosts'],['llm-proxy.app.all-hands.dev'])
            case = json.loads((task/'environment/data/case.json').read_text())
            self.assertFalse({'gold','answer','score_A','score_B','raw_pair'} & set(case))


if __name__ == '__main__':
    unittest.main()
