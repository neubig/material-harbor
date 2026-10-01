"""Generate a pinned, image-complete CSMBench MCQA pilot without model calls."""
import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parent
REVISION = '6dfc2ba8aea24dd04977801c68d2d6f24f76bc02'
BASE = f'https://huggingface.co/datasets/lututu/CSMBench/resolve/{REVISION}/multi_scale_mcq/train/'
METADATA_SHA256 = 'b676ba75003e2a89c7578ea981b1e8c36b12087ebf3fd43f4017cfdceda2cfec'
SEED = 'csmbench-pilot-v1-20260814'



def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def download(url, path, expected_sha256=None):
    """Only promote complete, validated downloads; retry transient source failures."""
    if path.exists():
        if expected_sha256 and sha256(path.read_bytes()) != expected_sha256:
            raise ValueError('Existing source checksum mismatch')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            with urlopen(url, timeout=120) as response:
                data = response.read()
            if expected_sha256 and sha256(data) != expected_sha256:
                raise ValueError('Downloaded source checksum mismatch')
            temporary.write_bytes(data)
            if path.suffix == '.jpg':
                validate_image(temporary)
            temporary.replace(path)
            return
        except (OSError, ValueError):
            temporary.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def load_rows():
    download(BASE + 'metadata.jsonl', ROOT / 'source/metadata.jsonl', METADATA_SHA256)
    data = (ROOT / 'source/metadata.jsonl').read_bytes()
    if sha256(data) != METADATA_SHA256:
        raise ValueError('Pinned metadata checksum mismatch')
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != 1041 or len({r['file_name'] for r in rows}) != 1041:
        raise ValueError('Unexpected count or duplicate images')
    if len({r['index'] for r in rows}) != 1041:
        raise ValueError('Duplicate source indices')
    for row in rows:
        options = json.loads(row['options'])
        if [o['label'] for o in options] != list('ABCD'):
            raise ValueError('Unexpected option labels')
        if any(not isinstance(o['caption'], str) or not o['caption'] for o in options):
            raise ValueError('Empty caption')
        if row['correct_answer'] not in list('ABCD') or not row['paper_folder_name']:
            raise ValueError('Invalid label or paper group')
        if not re.fullmatch(r'[0-9a-f]{64}\.jpg', row['file_name']):
            raise ValueError('Unsafe image filename')
    return rows


def build_prompt(options):
    options_text = '\n'.join(f"{opt['label']}. {opt['caption']}" for opt in options)
    return (
        'You are given an image and four candidate captions. Exactly one caption '
        'correctly describes the image.\n'
        'Return ONLY the letter of the correct caption (A, B, C, or D). '
        'Do not return any other text.\n\n'
        'Example response: C\n\n'
        f'Options:\n{options_text}\n\n'
        'IMPORTANT: Return only the letter (A, B, C, or D). No explanation.'
    )


def freeze_manifest(rows):
    selected = sorted(rows, key=lambda r: sha256(f"{SEED}:{r['file_name']}".encode()))[:10]
    selected_ids = {r['index'] for r in selected}
    manifest = {
        'dataset': 'lututu/CSMBench', 'revision': REVISION,
        'configuration': 'multi_scale_mcq', 'split': 'train',
        'metadata_sha256': METADATA_SHA256, 'seed': SEED,
        'selection': '10 lowest SHA256(seed + colon + file_name); before outcomes; no label stratification',
        'paper_count': len({r['paper_folder_name'] for r in rows}),
        'pilot_indices_in_selection_order': [r['index'] for r in selected],
        'rows': [{k: r[k] for k in ('index', 'file_name', 'paper_folder_name', 'scale', 'source', 'hybrid')}
                 | {'pilot': r['index'] in selected_ids} for r in rows],
    }
    path = ROOT / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Refusing to change frozen manifest')
    write_json(path, manifest)
    return selected


def validate_image(path):
    with Image.open(path) as image:
        if image.format != 'JPEG':
            raise ValueError('Expected JPEG')
        image.verify()
    with Image.open(path) as image:
        image.load()
        return {'width': image.width, 'height': image.height, 'mode': image.mode,
                'sha256': sha256(path.read_bytes()), 'bytes': path.stat().st_size}


def generate(output):
    rows = load_rows()
    selected = freeze_manifest(rows)
    if any((output / f"csmbench-mcqa-{r['index']:04d}").exists() for r in selected):
        raise FileExistsError('Refusing to overwrite existing tasks; use a fresh output directory')
    output.mkdir(parents=True, exist_ok=True)
    images = ROOT / 'source/images'
    images.mkdir(exist_ok=True)
    audit = []
    for row in selected:
        image = images / row['file_name']
        url = BASE + row['file_name']
        download(url, image)
        image_info = validate_image(image)
        task = output / f"csmbench-mcqa-{row['index']:04d}"
        (task / 'environment/data').mkdir(parents=True)
        (task / 'tests').mkdir()
        (task / 'solution').mkdir()
        shutil.copyfile(image, task / 'environment/data/image.jpg')
        prompt = build_prompt(json.loads(row['options']))
        write_json(task / 'environment/data/question.json', {
            'prompt': prompt, 'options': row['options'], 'image': '/app/data/image.jpg'})
        (task / 'instruction.md').write_text(prompt + '\n\nThe image is /app/data/image.jpg. '
            'The same prompt and original JSON-encoded options are in /app/data/question.json. '
            'Inspect the actual image; text alone is not the complete task. '
            'Write your final answer to /app/answer.txt as exactly one uppercase ASCII letter '
            '(A, B, C, or D), optionally followed by one LF newline. '
            'No spaces, JSON, markdown, or explanation in that file.\n')
        (task / 'environment/Dockerfile').write_text('FROM python:3.12-slim\nWORKDIR /app\nCOPY data /app/data\n')
        (task / 'task.toml').write_text('version = "1.0"\n\n[metadata]\n'
            'category = "materials-science"\ndataset = "CSMBench MCQA"\n'
            f'source_revision = "{REVISION}"\n\n'
            '[verifier]\ntimeout_sec = 60.0\n\n[agent]\ntimeout_sec = 900.0\n\n'
            '[environment]\nbuild_timeout_sec = 600.0\ncpus = 1\nmemory_mb = 2048\nstorage_mb = 4096\n')
        shutil.copyfile(ROOT / 'verify.py', task / 'tests/verify.py')
        write_json(task / 'tests/label.json', {'correct_answer': row['correct_answer']})
        (task / 'tests/test.sh').write_text('#!/bin/sh\nset -eu\npython /tests/verify.py\n')
        (task / 'solution/solve.sh').write_text('#!/bin/sh\nset -eu\n'
            f"printf '{row['correct_answer']}\\n' > /app/answer.txt\n")
        for path in (task / 'tests/test.sh', task / 'solution/solve.sh'):
            path.chmod(0o755)
        audit.append({'task': str(task), 'index': row['index'],
                      'paper_folder_name': row['paper_folder_name'], 'image_url': url, **image_info})
    write_json(ROOT / 'image-audit.json', {'images': audit,
        'pilot_paper_count': len({r['paper_folder_name'] for r in selected}),
        'paper_group_sizes_full': dict(Counter(r['paper_folder_name'] for r in rows))})
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tasks/pilot10')
    args = parser.parse_args()
    print(json.dumps(generate(args.output_dir), indent=2))


if __name__ == '__main__':
    main()
