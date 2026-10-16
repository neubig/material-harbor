import importlib.util
import json
import os
from pathlib import Path
import tempfile
import tomllib
import unittest

ROOT = Path(__file__).resolve().parents[1] / 'matbench-discovery'


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load('verify')
adapter = load('adapter')


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.answer = self.root / 'answer.json'

    def test_boolean_truth_table(self):
        for gold in (False, True):
            for answer in (False, True):
                self.answer.write_text(json.dumps({'stable': answer}))
                self.assertEqual(verify.evaluate(self.root, gold)['reward'], int(gold == answer))

    def test_reject_bad_payloads(self):
        for raw in ['{}', '[]', 'true', 'null', '{"stable":1}', '{"stable":0}',
                    '{"stable":"true"}', '{"stable":null}', '{"stable":NaN}',
                    '{"stable":true,"extra":0}', '{"stable":true,"stable":false}',
                    '{"stable":true} trailing', 'x'*1025, '['*1000, '']:
            self.answer.write_text(raw)
            self.assertEqual(verify.evaluate(self.root, True)['status'], 'invalid', raw[:50])

    def test_missing_and_non_utf8(self):
        self.assertEqual(verify.evaluate(self.root, False)['status'], 'invalid')
        self.answer.write_bytes(b'\xff')
        self.assertEqual(verify.evaluate(self.root, True)['status'], 'invalid')

    def test_symlink_hardlink_directory_fifo(self):
        target = self.root / 'target'
        target.write_text('{"stable":true}')
        self.answer.symlink_to(target)
        self.assertEqual(verify.evaluate(self.root, True)['status'], 'invalid')
        self.answer.unlink()
        os.link(target, self.answer)
        self.assertEqual(verify.evaluate(self.root, True)['status'], 'invalid')
        self.answer.unlink()
        self.answer.mkdir()
        self.assertEqual(verify.evaluate(self.root, True)['status'], 'invalid')
        self.answer.rmdir()
        os.mkfifo(self.answer)
        self.assertEqual(verify.evaluate(self.root, True)['status'], 'invalid')

    def test_symlink_output_directory(self):
        link = self.root / 'linked'
        link.symlink_to(self.root, target_is_directory=True)
        self.answer.write_text('{"stable":true}')
        self.assertEqual(verify.evaluate(link, True)['status'], 'invalid')

    def test_frozen_sample_truth_table(self):
        gold_path = ROOT / 'private/gold.json'
        if not gold_path.exists():
            self.skipTest('local frozen artifacts unavailable')
        gold = json.loads(gold_path.read_text())
        self.assertEqual(len(gold), 100)
        self.assertEqual(sum(x['stable'] for x in gold.values()), 50)
        for info in gold.values():
            self.answer.write_text(json.dumps({'stable': info['stable']}))
            self.assertEqual(verify.evaluate(self.root, info['stable'])['reward'], 1)
            self.answer.write_text(json.dumps({'stable': not info['stable']}))
            self.assertEqual(verify.evaluate(self.root, info['stable'])['reward'], 0)


class AdapterTests(unittest.TestCase):
    def test_sanitization(self):
        source = {'lattice': {'matrix': [[1,0,0],[0,1,0],[0,0,1]], 'leak': 2},
                  'sites': [{'species': [{'element': 'Si', 'occu': 1}], 'abc': [0,0,0], 'properties': {'energy': -5}, 'label': 'secret'}], 'energy': -5}
        clean = adapter.sanitize(source)
        self.assertNotIn('energy', json.dumps(clean))
        self.assertNotIn('secret', json.dumps(clean))
        source['lattice']['matrix'][0][0] = 0
        with self.assertRaises(ValueError):
            adapter.sanitize(source)

    def test_sample_manifest_and_no_resample(self):
        path = ROOT / 'private/sample.json'
        if not path.exists():
            self.skipTest('local frozen artifacts unavailable')
        sample = json.loads(path.read_text())
        manifest = json.loads((ROOT / 'sample-manifest.json').read_text())
        self.assertEqual(len({r['material_id'] for r in sample}), 100)
        self.assertEqual(adapter.digest(path), manifest['sample_sha256'])
        self.assertEqual(adapter.digest(ROOT / 'protocol.json'), manifest['protocol_sha256'])
        with self.assertRaises(FileExistsError):
            adapter.prepare(Path('/nonexistent'))


class GeneratedTaskTests(unittest.TestCase):
    def test_private_build_boundary_and_task_hashes(self):
        manifest_path = ROOT / 'task-manifest.json'
        if not manifest_path.exists():
            self.skipTest('generated tasks unavailable')
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(len(manifest), 100)
        tasks_root = Path(os.environ.get('MATBENCH_TASKS_DIR', ROOT / 'tasks'))
        if not tasks_root.exists():
            self.skipTest('generated tasks unavailable')
        for task_id, hashes in manifest.items():
            task = tasks_root / task_id
            for relative, expected in hashes.items():
                self.assertEqual(adapter.digest(task / relative), expected)
            config = tomllib.loads((task / 'task.toml').read_text())
            self.assertEqual(config['verifier']['environment_mode'], 'separate')
            self.assertEqual(config['verifier']['network_mode'], 'no-network')
            self.assertEqual(config['agent']['user'], 'agent')
            self.assertEqual(config['environment']['allowed_hosts'], ['llm-proxy.app.all-hands.dev'])
            self.assertEqual(config['artifacts'], ['/output'])
            paths = {str(p.relative_to(task / 'environment')) for p in (task / 'environment').rglob('*') if p.is_file()}
            self.assertEqual(paths, {'Dockerfile','requirements.lock','sdk-requirements.lock','predict.py','checkpoint.json','chgnet_0.3.0.pth.tar','data/structure.json','data/reference.json'})
            for name in ['data/structure.json','data/reference.json']:
                text = (task / 'environment' / name).read_text()
                self.assertNotIn('wbm-', text)
                self.assertNotIn('stable', text)
                self.assertNotIn('distance', text)


class ScientificEvidenceTests(unittest.TestCase):
    def test_independent_audit_frozen_and_complete(self):
        path = ROOT / 'science/audit.json'
        if not path.exists():
            self.skipTest('independent reconstruction unavailable')
        audit = json.loads(path.read_text())
        self.assertEqual(audit['state'], 'frozen_before_gold_comparison')
        self.assertEqual(len(audit['records']), 100)
        self.assertEqual(adapter.digest(path), (ROOT / 'science/audit.sha256').read_text().split()[0])
        self.assertEqual(audit['mp_entry_count'], 154718)
        for row in audit['records']:
            self.assertEqual(row['stable'], row['signed_distance_ev_per_atom'] <= 0)


if __name__ == '__main__':
    unittest.main()
