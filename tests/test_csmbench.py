import json
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

from PIL import Image
from harbor.models.task.config import TaskConfig

from csmbench.main import generate


class CSMBenchTests(unittest.TestCase):
    def make_source(self, root, duplicate=False):
        image_dir = root / 'images'; image_dir.mkdir()
        Image.new('RGB', (8, 8), 'red').save(image_dir / 'one.jpg')
        captions = ['first', 'first' if duplicate else 'second']
        row = {'file_name':'one.jpg','options':json.dumps([{'label':'A','caption':captions[0]},{'label':'B','caption':captions[1]}]),'correct_answer':'B','index':7,'scale':'Microscale','source':'mc','paper_folder_name':'paper','hybrid':False}
        metadata = root / 'metadata.jsonl';metadata.write_text(json.dumps(row)+'\n')
        return metadata,image_dir

    def test_generation_withholds_gold_and_parses_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);metadata,images=self.make_source(root);output=root/'tasks'
            self.assertEqual(generate(metadata,images,output,['7']),1)
            task=output/'csmbench-000007';self.assertTrue((task/'environment/data/image.jpg').exists())
            self.assertFalse(any('gold' in path.name or 'answer' in path.name for path in (task/'environment').rglob('*')))
            self.assertEqual(json.loads((task/'tests/gold.json').read_text())['answer'],'B')
            TaskConfig.model_validate(tomllib.loads((task/'task.toml').read_text()))

    def test_duplicate_captions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);metadata,images=self.make_source(root,True)
            with self.assertRaises(ValueError):generate(metadata,images,root/'tasks',['7'])

    def test_verifier_strict_outputs(self):
        verifier=Path(__file__).resolve().parents[1]/'csmbench/verify.py'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);gold=root/'gold.json';gold.write_text(json.dumps({'answer':'B','letters':['A','B']}));answer=root/'answer.txt'
            for content,expected in [(b'B',1),(b'B\n',1),(b'A',0),(b'b',0),(b'Answer: B',0),(b'',0),(None,0)]:
                if answer.exists():answer.unlink()
                if content is not None:answer.write_bytes(content)
                logs=root/f'logs-{len(list(root.glob("logs-*")))}'
                subprocess.run([sys.executable,str(verifier),'--answer-path',str(answer),'--gold-path',str(gold),'--logs-dir',str(logs)],check=True)
                self.assertEqual(float((logs/'reward.txt').read_text()),expected)


if __name__=='__main__':
    unittest.main()
