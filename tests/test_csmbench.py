import ast
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'csmbench'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adapter = load_module('csmbench_adapter', ROOT / 'main.py')
verifier = load_module('csmbench_verifier', ROOT / 'verify.py')


@unittest.skipUnless(all((ROOT / p).exists() for p in ("source/metadata.jsonl", "source/utils.py", "manifest.json", "tasks/pilot10")), "Generate CSMBench pilot and fetch pinned upstream audit fixtures first")
class CSMBenchTests(unittest.TestCase):
    def test_full_actual_source_and_frozen_selection(self):
        rows = adapter.load_rows()
        manifest = json.loads((ROOT / 'manifest.json').read_text())
        self.assertEqual(len(rows), 1041)
        self.assertEqual(len(manifest['rows']), 1041)
        self.assertEqual(len({r['paper_folder_name'] for r in rows}), 322)
        expected = sorted(rows, key=lambda r: adapter.sha256(
            f"{adapter.SEED}:{r['file_name']}".encode()))[:10]
        self.assertEqual(manifest['pilot_indices_in_selection_order'], [r['index'] for r in expected])
        for row, frozen in zip(rows, manifest['rows']):
            for key in ('index', 'file_name', 'paper_folder_name', 'scale', 'source', 'hybrid'):
                self.assertEqual(frozen[key], row[key])
        self.assertEqual(sum(r['pilot'] for r in manifest['rows']), 10)

    def test_prompt_matches_authors_on_every_actual_row(self):
        tree = ast.parse((ROOT / 'source/utils.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'build_mcqa_prompt')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<official prompt>', 'exec'), namespace)
        for row in adapter.load_rows():
            options = json.loads(row['options'])
            self.assertEqual(adapter.build_prompt(options), namespace['build_mcqa_prompt'](options))

    def test_images_paper_groups_and_task_isolation(self):
        rows = {r['index']: r for r in adapter.load_rows()}
        audit = json.loads((ROOT / 'image-audit.json').read_text())
        self.assertEqual(audit['paper_group_sizes_full'], dict(Counter(
            r['paper_folder_name'] for r in rows.values())))
        tasks = sorted((ROOT / 'tasks/pilot10').iterdir())
        self.assertEqual(len(tasks), 10)
        for task in tasks:
            index = int(task.name.rsplit('-', 1)[1])
            row = rows[index]
            entry = next(e for e in audit['images'] if e['index'] == index)
            for key, value in adapter.validate_image(task / 'environment/data/image.jpg').items():
                self.assertEqual(entry[key], value)
            self.assertEqual((ROOT / 'source/images' / row['file_name']).read_bytes(),
                             (task / 'environment/data/image.jpg').read_bytes())
            question = json.loads((task / 'environment/data/question.json').read_text())
            self.assertEqual(set(question), {'prompt', 'options', 'image'})
            self.assertEqual(question['options'], row['options'])
            self.assertTrue((task / 'instruction.md').read_text().startswith(question['prompt']))
            self.assertEqual({str(p.relative_to(task / 'environment')) for p in
                              (task / 'environment').rglob('*') if p.is_file()},
                             {'Dockerfile', 'data/question.json', 'data/image.jpg'})
            self.assertNotIn('correct_answer', (task / 'instruction.md').read_text())
            self.assertNotIn(row['paper_folder_name'], (task / 'instruction.md').read_text())
            self.assertEqual(json.loads((task / 'tests/label.json').read_text()),
                             {'correct_answer': row['correct_answer']})
            self.assertEqual((task / 'tests/verify.py').read_bytes(), (ROOT / 'verify.py').read_bytes())

    def test_download_real_file_bytes_and_cache_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.jsonl'
            source.write_bytes(b'{"hello":"world"}\n')
            target = root / 'downloaded.jsonl'
            expected = adapter.sha256(source.read_bytes())
            adapter.download(source.as_uri(), target, expected)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertFalse(target.with_suffix('.jsonl.partial').exists())
            target.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                adapter.download(source.as_uri(), target, expected)

    def test_all_labels_and_malformed_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            answer = Path(directory) / 'answer.txt'
            for correct in 'ABCD':
                self.assertEqual(verifier.grade(answer, correct), 0)
                for letter in 'ABCD':
                    for suffix in ('', '\n'):
                        answer.write_bytes((letter + suffix).encode())
                        self.assertEqual(verifier.grade(answer, correct), float(letter == correct))
                malformed = ['', correct.lower(), ' ' + correct, correct + ' ', correct + '\r\n',
                             correct + '\n\n', correct + '. ', f'"{correct}"',
                             f'{{"answer":"{correct}","is_correct":true}}',
                             f'The answer is {correct}', f'```\n{correct}\n```', correct * 10000,
                             correct + '\x00', '\ufeff' + correct]
                for text in malformed:
                    answer.write_bytes(text.encode())
                    self.assertEqual(verifier.grade(answer, correct), 0, repr(text))
                answer.write_bytes(b'\xff')
                self.assertEqual(verifier.grade(answer, correct), 0)
                answer.unlink()
            answer.mkdir()
            self.assertEqual(verifier.grade(answer, 'A'), 0)
            answer.rmdir()
            target = Path(directory) / 'target'
            target.write_text('A')
            answer.symlink_to(target)
            self.assertEqual(verifier.grade(answer, 'A'), 0)
            with self.assertRaises(ValueError):
                verifier.grade(target, 'Z')

    def test_generated_verifier_cli_correct_wrong_missing_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            answer = Path(directory) / 'answer.txt'
            reward = Path(directory) / 'logs/reward.txt'
            for task in (ROOT / 'tasks/pilot10').iterdir():
                correct = json.loads((task / 'tests/label.json').read_text())['correct_answer']
                wrong = next(x for x in 'ABCD' if x != correct)
                for content, score in [(correct + '\n', 1), (wrong, 0), ('malformed', 0), (None, 0)]:
                    if content is None:
                        answer.unlink()
                    else:
                        answer.write_text(content)
                    subprocess.run([sys.executable, str(task / 'tests/verify.py'),
                                    '--answer', str(answer), '--label', str(task / 'tests/label.json'),
                                    '--reward', str(reward)], check=True)
                    self.assertEqual(float(reward.read_text()), score)
                subprocess.run(['sh', '-n', str(task / 'tests/test.sh')], check=True)
                subprocess.run(['sh', '-n', str(task / 'solution/solve.sh')], check=True)

    def test_invalid_image_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'bad.jpg'
            bad.write_bytes(b'not an image')
            with self.assertRaises(OSError):
                adapter.validate_image(bad)
        with self.assertRaises(FileExistsError):
            adapter.generate(ROOT / 'tasks/pilot10')


if __name__ == '__main__':
    unittest.main()
