import base64
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from matqna.adapter import generate, parse_value, records_at_indices


class MatQnASelectedRowsTests(unittest.TestCase):
    def test_labeled_choices_allow_internal_apostrophes(self):
        value = "['A. unchanged'\n 'B. material's transition narrows'\n 'C. transition widens'\n 'D. disappears']"
        self.assertEqual(
            parse_value(value),
            ['A. unchanged', "B. material's transition narrows", 'C. transition widens', 'D. disappears'],
        )

    def make_source(self, root):
        rows = []
        for index in range(3):
            rows.append({
                'qa_type': 'objective',
                'images': base64.b64encode(f'image-{index}'.encode()).decode(),
                'question': f'Question {index}?',
                'type': 'test',
                'choices': "['A. first' 'B. second']",
                'scholar_reference_answer': 'AB'[index % 2],
                'category': 'xrd',
            })
        source = root / 'source.parquet'
        pd.DataFrame(rows).to_parquet(source, row_group_size=1)
        return source

    def test_selected_rows_preserve_source_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.make_source(root)
            output = root / 'tasks'
            self.assertEqual(generate(output, source, ids=['2', '0']), 2)
            self.assertEqual({path.name for path in output.iterdir()}, {'matqna-000000', 'matqna-000002'})
            self.assertEqual((output / 'matqna-000002/environment/data/image.png').read_bytes(), b'image-2')
            self.assertIn('Question 0?', (output / 'matqna-000000/instruction.md').read_text())

    def test_selected_rows_reject_invalid_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(Path(directory))
            with self.assertRaises(IndexError):
                records_at_indices(source, [-1])
            with self.assertRaises(IndexError):
                records_at_indices(source, [3])


if __name__ == '__main__':
    unittest.main()
