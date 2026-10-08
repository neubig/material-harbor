import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import tomllib
import unittest
import zipfile
from pathlib import Path

from PIL import Image
from harbor.models.task.config import TaskConfig

ADAPTER = Path(__file__).resolve().parents[1] / 'matcha'


def load_adapter():
    spec = importlib.util.spec_from_file_location('matcha_adapter', ADAPTER / 'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VerifierTests(unittest.TestCase):
    def test_real_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold = root / 'gold.json'
            gold.write_text(json.dumps({'answer': 'B', 'letters': ['A', 'B', 'C']}))
            answer = root / 'answer.txt'
            cases = [(b'B', 1), (b'B\n', 1), (b'A', 0), (b'b', 0),
                     (b' B', 0), (b'B\n\n', 0), (b'Answer: B', 0),
                     (b'{"answer":"B"}', 0), (b'Z', 0), (b'', 0),
                     (b'B' * 1025, 0), (b'\xff', 0), (None, 0)]
            for content, expected in cases:
                if answer.exists():
                    answer.unlink()
                if content is not None:
                    answer.write_bytes(content)
                self.check_score(root, gold, answer, expected)
            target = root / 'target'
            target.write_text('B')
            answer.symlink_to(target)
            self.check_score(root, gold, answer, 0)
            answer.unlink()
            os.mkfifo(answer)
            self.check_score(root, gold, answer, 0)
            answer.unlink()
            answer.mkdir()
            self.check_score(root, gold, answer, 0)
            child = root / 'child'
            child.mkdir()
            (child / 'answer').write_text('B')
            link = root / 'link'
            link.symlink_to(child, target_is_directory=True)
            self.check_score(root, gold, link / 'answer', 0)

    def check_score(self, root, gold, answer, expected):
        subprocess.run([sys.executable, str(ADAPTER / 'verify.py'),
                        '--answer-path', str(answer), '--gold-path', str(gold),
                        '--logs-dir', str(root / 'logs')], check=True, timeout=5)
        self.assertEqual(float((root / 'logs/reward.txt').read_text()), expected)


class GenerationTests(unittest.TestCase):
    def test_preservation_selection_and_config(self):
        adapter = load_adapter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.tif'
            Image.new('RGB', (8, 8), 'blue').save(image)
            archive = root / 'images.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.write(image, 'images/a.tif')
                z.write(image, 'images/b.tif')
            qa = {'question': 'Original? (A) first (B) second',
                  'options': {'A': 'first', 'B': 'second'}, 'answer': 'B', 'topic': 'test'}
            rows = [{'id': 'original-id', 'vqa': [qa, qa],
                     'images': [{'image_path': 'a.tif', 'geometry': [{'x': 1, 'y': 2}]},
                                {'image_path': 'b.tif', 'geometry': []}],
                     'article_info': {'license': 'example'}}]
            records = adapter.records(rows)
            self.assertEqual(adapter.select(records), records)
            self.assertEqual(len(records), 2)
            self.assertNotEqual(records[0]['id'], records[1]['id'])
            self.assertEqual(adapter.select(records, limit=1), records[:1])
            self.assertEqual(adapter.select(records, ids=[records[1]['id']]), records[1:])
            for kwargs in [{'ids': ['missing']}, {'limit': 0}]:
                with self.assertRaises(ValueError):
                    adapter.select(records, **kwargs)
            output = root / 'tasks'
            adapter.generate(records, archive, output)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(len(manifest['tasks']), 2)
            self.assertEqual(manifest['tasks'][0]['question_group'], manifest['tasks'][1]['question_group'])
            for entry in manifest['tasks']:
                task = output / entry['task_id']
                public = json.loads((task / 'environment/data/question.json').read_text())
                self.assertEqual(public['question'], qa['question'])
                self.assertEqual(public['options'], qa['options'])
                self.assertEqual(public['images'], rows[0]['images'])
                self.assertNotIn('answer', public)
                self.assertEqual(entry['source_id'], 'original-id')
                self.assertEqual(len(entry['assets']), 2)
                for asset in entry['assets']:
                    self.assertEqual((task / 'environment/data' / asset['original']).read_bytes(), image.read_bytes())
                    self.assertTrue((task / 'environment/data' / asset['view']).exists())
                    self.assertIn('/app/data/' + asset['view'], (task / 'instruction.md').read_text())
                TaskConfig.model_validate(tomllib.loads((task / 'task.toml').read_text()))
            repeated = root / 'repeated'
            adapter.generate(records, archive, repeated)
            self.assertEqual((output / 'manifest.json').read_bytes(),
                             (repeated / 'manifest.json').read_bytes())
            with self.assertRaises(FileExistsError):
                adapter.generate(records, archive, output)


if __name__ == '__main__':
    unittest.main()
