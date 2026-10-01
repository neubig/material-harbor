import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from harbor.models.task.task import Task

ROOT = Path(__file__).resolve().parents[1] / 'matcha'


def load(name):
    spec = importlib.util.spec_from_file_location('matcha_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load('adapter')
verifier = load('verify')


class MatchaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all((ROOT / p).exists() for p in ("source/utils.py", "source/images.zip", "tasks/full", "tasks/pilot10", "manifest.json")):
            raise unittest.SkipTest("Generate MATCHA pilot10 and full cohorts before running real-image audit")
        cls.assert_source = adapter.digest(ROOT / 'source/utils.py')
        if cls.assert_source != adapter.UPSTREAM_HASH:
            raise ValueError('Unverified upstream reference code')
        native_spec = importlib.util.spec_from_file_location('matcha_native', ROOT / 'source/utils.py')
        cls.native = importlib.util.module_from_spec(native_spec)
        sys.modules[native_spec.name] = cls.native
        native_spec.loader.exec_module(cls.native)
        cls.records = adapter.flatten(ROOT / 'source/matcha_vqa_inputs.jsonl')

    def test_manifest_and_all_labels(self):
        plan = adapter.manifest(self.records)
        self.assertEqual(plan, json.loads((ROOT / 'manifest.json').read_text()))
        self.assertEqual(len(set(r['task_id'] for r in self.records)), 1500)
        for record in self.records:
            qa = record['qa']
            self.assertEqual(verifier.grade(qa['answer'], qa['answer'], qa['options']), 1)
            for option in qa['options']:
                self.assertEqual(verifier.grade(option, qa['answer'], qa['options']), int(option == qa['answer']))
            for bad in [None, '', qa['answer'].lower(), qa['answer'] + '.', 'Answer: ' + qa['answer'], json.dumps({'answer': qa['answer']}), 'A B C D', [], 1]:
                self.assertEqual(verifier.grade(bad, qa['answer'], qa['options']), 0)

    def test_all_real_images_native_pixels(self):
        with zipfile.ZipFile(ROOT / 'source/images.zip') as archive:
            for record in self.records:
                for spec in record['images']:
                    actual = adapter.crop_image(archive, spec)
                    with tempfile.TemporaryDirectory() as temp:
                        path = Path(temp) / spec['image_path']
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(archive.read(adapter.safe_member(archive, spec['image_path'])))
                        expected = self.native.crop_subfigure(str(path), spec.get('geometry'))
                        expected.load()
                    self.assertEqual(actual.mode, expected.mode)
                    self.assertEqual(actual.size, expected.size)
                    self.assertEqual(actual.tobytes(), expected.tobytes())
                    index = record['images'].index(spec)
                    with Image.open(ROOT / 'tasks/full' / record['task_id'] / f'environment/data/image-{index:02d}.png') as saved:
                        self.assertEqual(saved.mode, expected.mode)
                        self.assertEqual(saved.size, expected.size)
                        self.assertEqual(saved.tobytes(), expected.tobytes())
                    expected.close()
                    actual.close()

    def test_generated_pilot_and_cli_verifier(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            for task_id in adapter.manifest(self.records)['pilot_ids']:
                task = ROOT / 'tasks/pilot10' / task_id
                Task(task)
                public = json.loads((task / 'environment/data/input.json').read_text())
                self.assertEqual(set(public), {'system', 'prompt', 'question', 'options', 'images'})
                record = next(r for r in self.records if r['task_id'] == task_id)
                self.assertEqual(public['question'], record['qa']['question'])
                self.assertEqual(public['options'], record['qa']['options'])
                reference = task / 'tests/reference.json'
                answer, reward = temp / 'answer.txt', temp / 'reward.txt'
                for payload in [record['qa']['answer'], 'Z', 'A B C D', '', None, '\xff']:
                    answer.unlink(missing_ok=True)
                    if payload is not None:
                        answer.write_bytes(payload.encode('latin1'))
                    subprocess.run([sys.executable, str(task / 'tests/verify.py'), '--answer', str(answer), '--reference', str(reference), '--reward', str(reward)], check=True)
                    self.assertEqual(reward.read_text().strip(), '1' if payload == record['qa']['answer'] else '0')

    def test_prospective_selection_and_no_overwrite(self):
        plan = adapter.selection100(self.records)
        self.assertEqual(len(set(plan['task_ids'])), 100)
        self.assertEqual(plan, adapter.selection100(self.records))
        self.assertEqual(plan, json.loads((ROOT / 'manifest100.json').read_text()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'manifest.json'
            adapter.save_manifest(path, plan)
            original = path.read_bytes()
            adapter.save_manifest(path, plan)
            with self.assertRaises(ValueError):
                adapter.save_manifest(path, {})
            self.assertEqual(path.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            adapter.generate([self.records[0]], ROOT / 'source/images.zip', ROOT / 'tasks/full')

    def test_full_tasks_and_gold_isolation(self):
        for record in self.records:
            task = ROOT / 'tasks/full' / record['task_id']
            Task(task)
            public = json.loads((task / 'environment/data/input.json').read_text())
            self.assertEqual(set(public), {'system', 'prompt', 'question', 'options', 'images'})
            self.assertEqual(public['question'], record['qa']['question'])
            self.assertEqual(public['options'], record['qa']['options'])
            self.assertEqual(public['images'], [f'/app/data/image-{i:02d}.png' for i in range(len(record['images']))])
            self.assertEqual({p.name for p in (task / 'environment').iterdir()}, {'data', 'Dockerfile'})
            self.assertEqual({p.name for p in (task / 'environment/data').iterdir()},
                             {'input.json'} | {f'image-{i:02d}.png' for i in range(len(record['images']))})
            self.assertEqual((task / 'environment/Dockerfile').read_text(),
                             'FROM python:3.12-slim\nWORKDIR /app\nCOPY data /app/data\n')
            reference = json.loads((task / 'tests/reference.json').read_text())
            self.assertEqual(reference, {'answer': record['qa']['answer'], 'options': list(record['qa']['options'])})

    def test_safe_paths_and_multi_image_order(self):
        with zipfile.ZipFile(ROOT / 'source/images.zip') as archive:
            for bad in ['../outside', '/etc/passwd', 'x/../../outside', 'x\\y', './x']:
                with self.assertRaises(ValueError):
                    adapter.safe_member(archive, bad)
        record = copy.deepcopy(self.records[0])
        record['images'].append(self.records[1]['images'][0])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)
            adapter.generate([record], ROOT / 'source/images.zip', output)
            with zipfile.ZipFile(ROOT / 'source/images.zip') as archive:
                for index, spec in enumerate(record['images']):
                    with Image.open(output / record['task_id'] / f'environment/data/image-{index:02d}.png') as image:
                        self.assertEqual(image.tobytes(), adapter.crop_image(archive, spec).tobytes())


if __name__ == '__main__':
    unittest.main()
