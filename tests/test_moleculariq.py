import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'moleculariq'
spec = importlib.util.spec_from_file_location('moleculariq_verifier', ROOT / 'verifier.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


class MolecularIQTests(unittest.TestCase):
    def test_finite_exhaustive(self):
        for n in range(12):
            for gold in range(n + 1):
                ref = {'key': 'carbon_atom_count', 'value': gold, 'upper_bound': n}
                for candidate in range(-1, n + 2):
                    self.assertEqual(v.accepts(json.dumps({'carbon_atom_count': candidate}), ref), candidate == gold)

    def test_permissive_regressions(self):
        ref = {'key': 'k', 'value': 2, 'upper_bound': 4}
        for text in ['{"wrong":2}', '{"k":2,"extra":0}', '{"k":2.0}', '{"k":2.0000001}',
                     '{"k":true}', '{"k":"2"}', '{"k":NaN}', '{"k":Infinity}', '{"k":2,"k":2}',
                     '[2]', '2', 'null', '<answer>{"k":2}</answer>', '{"k":2} trailing', ' ' * 4097]:
            with self.subTest(text=text):
                self.assertFalse(v.accepts(text, ref))
        self.assertTrue(v.accepts(' \n{"k":2}\n', ref))

    def test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'answer'
            p.write_text('{"k":2}')
            self.assertEqual(v.read_answer(p), '{"k":2}')
            link = Path(tmp) / 'link'
            link.symlink_to(p)
            with self.assertRaises(OSError):
                v.read_answer(link)
            p.write_bytes(b'x' * 4097)
            with self.assertRaises(ValueError):
                v.read_answer(p)
            fifo = Path(tmp) / 'fifo'
            os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                v.read_answer(fifo)

    def test_frozen_isolation_and_domain(self):
        root = ROOT / 'frozen'
        if not root.exists():
            self.skipTest('cohort not generated')
        sample = json.loads((root / 'sample.json').read_text())
        self.assertEqual(len(sample['uids']), 100)
        self.assertEqual(len(set(sample['uids'])), 100)
        tasks_root = Path(os.environ.get('MOLECULARIQ_TASKS_DIR', root / 'tasks'))
        if not tasks_root.exists():
            self.skipTest('generated tasks not available')
        for uid in sample['uids']:
            task = tasks_root / uid
            self.assertEqual([p.name for p in (task / 'environment').iterdir()], ['Dockerfile'])
            ref = json.loads((task / 'tests/reference.json').read_text())
            for value in range(ref['upper_bound'] + 1):
                self.assertEqual(v.accepts(json.dumps({ref['key']: value}), ref), value == ref['value'])
            config = (task / 'task.toml').read_text()
            self.assertIn('environment_mode = "separate"', config)
            self.assertIn('allowed_hosts = ["llm-proxy.app.all-hands.dev"]', config)



class FrozenManifestTests(unittest.TestCase):
    def test_reproducible_selection(self):
        universe = json.loads((ROOT / 'frozen/universe.json').read_text())
        sample = json.loads((ROOT / 'frozen/sample.json').read_text())
        self.assertEqual(sample['uids'], [r['uid'] for r in random.Random(sample['seed']).sample(universe, 100)])
        self.assertEqual(sample['universe_sha256'], hashlib.sha256((ROOT/'frozen/universe.json').read_bytes()).hexdigest())
        self.assertEqual(sample['universe_size'], len(universe))
        self.assertTrue(all('target' not in r for r in universe))

    def test_blind_audit_context(self):
        audit_root = Path(os.environ.get('MOLECULARIQ_AUDIT_TASKS_DIR', ROOT / 'audit-tasks'))
        if not audit_root.exists():
            self.skipTest('generated audit tasks not available')
        for task in audit_root.iterdir():
            self.assertEqual([p.name for p in (task/'environment').iterdir()], ['Dockerfile'])
            self.assertFalse((task/'tests/reference.json').exists())
            self.assertEqual((task/'solution/solve.sh').read_text(), '#!/bin/sh\nexit 1\n')


if __name__ == '__main__':
    unittest.main()
