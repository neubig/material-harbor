import collections
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / 'lab-bench-seqqa'
sys.path.insert(0, str(ADAPTER))
import main as adapter
import validation

spec = importlib.util.spec_from_file_location('seqqa_verify', ADAPTER / 'task-template/tests/verify.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
SOURCE = Path(os.environ.get('SEQQA_SOURCE_DIR', str(ADAPTER / 'sources')))


class SeqQATests(unittest.TestCase):
    def test_orf_conventions(self):
        self.assertEqual(validation.proteins('ATGATGTAA'), ['MM*', 'M*'])
        self.assertEqual(validation.proteins('ATGAAA'), [])
        self.assertEqual(validation.proteins('ATGAAATAAATGCCCTAG'), ['MK*', 'MP*'])
        self.assertEqual(validation.proteins('AATGAAATAA'), ['MK*'])
        row = {'question': 'ORFs greater than 1 in ATGATGTAACCCCCCC', 'ideal': '1', 'distractors': ['0', '2', '3']}
        self.assertTrue(validation.validate(row, 'ORF-seq-numlen')['valid'])

    def test_pcr_orientation_and_inclusive_length(self):
        sequence = 'AAACCCGGGTTT'
        self.assertEqual(validation.reverse_complement('ATGC'), 'GCAT')
        self.assertEqual(validation.amplicons(sequence, 'AAAC, AAAC'), {sequence})
        self.assertNotIn(sequence, validation.amplicons(sequence, 'AAAC, GTTT'))
        self.assertEqual(validation.amplicons(sequence, 'ATAT, TATA'), set())

    def test_strict_format_and_whole_option(self):
        for answer in 'ABCDEF':
            for gold in 'ABCDEF':
                result = verifier.evaluate(json.dumps({'answer': answer}), gold, list('ABCDEF'))
                self.assertEqual(result, {'format': 1.0, 'answer': float(answer == gold), 'reward': float(answer == gold)})
        for malformed in ['', '{}', 'null', '[]', '"A"', '{"answer":true}', '{"answer":1}',
                          '{"answer":"a"}', '{"answer":" A"}', '{"answer":"AA"}',
                          '{"answer":"G"}', '{"answer":"A","reason":"x"}',
                          '{"answer":"A","answer":"A"}', '```json\n{"answer":"A"}\n```']:
            self.assertEqual(verifier.evaluate(malformed, 'A', list('ABCDEF')),
                             {'format': 0.0, 'answer': 0.0, 'reward': 0.0})

    def test_reject_unsafe_answer_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / 'target.json'
            target.write_text('{"answer":"A"}')
            self.assertEqual(verifier.read_answer(target), '{"answer":"A"}')
            link = root / 'link.json'
            link.symlink_to(target)
            with self.assertRaises(OSError):
                verifier.read_answer(link)
            target.write_text(' ' * 1025)
            with self.assertRaises(ValueError):
                verifier.read_answer(target)
            fifo = root / 'fifo'
            os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                verifier.read_answer(fifo)
            with self.assertRaises((ValueError, OSError)):
                verifier.read_answer(root)
            child = root / 'child'
            child.mkdir()
            (child / 'answer.json').write_text('{"answer":"A"}')
            parent_link = root / 'parent-link'
            parent_link.symlink_to(child, target_is_directory=True)
            with self.assertRaises(ValueError):
                verifier.read_answer(parent_link / 'answer.json')

    def test_grouping_is_transitive_and_reverse_complement_invariant(self):
        sequences = ['ACGTACCCCCGGGGTT', 'AAACGTACCCCCGGGGTTAAA', 'AACCCCGGGGGTACGT', 'TTTTTTTTTTTTTTT']
        records = [{'id': str(i), 'sequence': s} for i, s in enumerate(sequences)]
        groups = adapter.group_candidates(records)
        self.assertEqual(sorted(map(len, groups)), [1, 3])

    @unittest.skipUnless(SOURCE.exists(), 'Pinned source checkout is needed for full-corpus tests')
    def test_all_source_answers_and_frozen_sampling(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'frozen'
            subprocess.run([sys.executable, str(ADAPTER / 'main.py'), '--source', str(SOURCE),
                            '--output', str(output)], check=True, stdout=subprocess.DEVNULL)
            expected = json.loads((ADAPTER / 'expected-frozen-sha256.json').read_text())
            actual = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in output.rglob('*') if p.is_file() and p.name != 'frozen-sha256.json'}
            self.assertEqual(actual, expected)
        candidates, excluded, audit = adapter.load_candidates(SOURCE)
        self.assertEqual(len(candidates), 238)
        self.assertEqual(len(audit), 240)
        self.assertEqual(len(excluded), 362)
        self.assertEqual({r['id'] for r in excluded if r['reason'] != 'outside_predeclared_six_family_scope'},
                         {'0576d42c-8bfa-4851-a88a-b2bb295d3e59', 'a1758c94-728d-4d22-9526-c5291177c54c'})
        sample, groups, quotas = adapter.select_representatives(candidates)
        repeated, _, _ = adapter.select_representatives(list(reversed(candidates)))
        self.assertEqual([r['id'] for r in sample], [r['id'] for r in repeated])
        self.assertEqual(len(sample), 80)
        self.assertEqual(len({r['group_id'] for r in sample}), 80)
        self.assertEqual(set(quotas), set(validation.FAMILIES))
        self.assertTrue(all(quotas.values()))
        for record in candidates:
            row = record['row']
            options, answer = adapter.choices(row)
            self.assertEqual(options[answer], row['ideal'])
            self.assertEqual(collections.Counter(options.values()),
                             collections.Counter([row['ideal'], adapter.REFUSAL, *row['distractors']]))
            self.assertEqual(adapter.choices(row), adapter.choices(row))
            for letter in options:
                score = verifier.evaluate(json.dumps({'answer': letter}), answer, list(options))
                self.assertEqual(score['reward'], float(letter == answer))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for family in validation.FAMILIES:
                record = next(r for r in sample if r['family'] == family)
                task = adapter.generate_task(record, path)
                case = json.loads((task / 'environment/data/case.json').read_text())
                self.assertEqual(case['question'], record['row']['question'])
                self.assertEqual(set(case), {'question', 'options', 'conventions'})
                self.assertEqual({p.name for p in (task / 'environment/data').iterdir()}, {'case.json', 'ATTRIBUTION.txt', 'LICENSE'})
                self.assertNotIn('COPY tests', (task / 'environment/Dockerfile').read_text())
                self.assertIn('environment_mode = "separate"', (task / 'task.toml').read_text())
                self.assertIn('COPY . /tests', (task / 'tests/Dockerfile').read_text())
                self.assertIn('python -I', (task / 'tests/test.sh').read_text())
                gold = json.loads((task / 'tests/gold.json').read_text())
                answer_path = path / 'answer.json'
                args = [sys.executable, str(task / 'tests/verify.py'), '--answer-path', str(answer_path),
                        '--info-path', str(task / 'tests/gold.json'), '--logs-dir', str(path / 'logs')]
                for text, reward in [(json.dumps({'answer': gold['answer']}), 1), ('{}', 0), ('missing', 0)]:
                    if text == 'missing':
                        answer_path.unlink(missing_ok=True)
                    else:
                        answer_path.write_text(text)
                    subprocess.run(args, check=True)
                    self.assertEqual(float((path / 'logs/reward.txt').read_text()), reward)


if __name__ == '__main__':
    unittest.main()
