from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import tifffile
from PIL import Image

ROOT = Path(__file__).resolve().parent
REVISION = 'dc52d6016053c67ebdf41719d6cfaa0603b154fa'
BASE = f'https://huggingface.co/datasets/FreedomIntelligence/MatCha/resolve/{REVISION}/data/'
HASHES = {'matcha_vqa_inputs.jsonl': 'd80382468646ab7d58ef638780754f5a602e6a2c2a98c0564dfd260ae89bd916',
          'images.zip': '1dea6b1065d33ca09266989fecd3543ab7dbfb171b923efd9584e01664021e35'}
SYSTEM = 'You are a helpfule material science assistant. Based on the figure, please answer the following question. The answer could be inferred from the figure and must be concise and clear. Answer directly without any explanation.'
SEED = 42
UPSTREAM_REVISION = 'fef900719ae08b7cdbd2df90b35331b3649f88d6'
UPSTREAM_HASH = '42dc371d3354a96d86dcb8cfb28cba282e150d23a74f7c9e2f96a4dcc3b3965e'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def fetch(source):
    source.mkdir(parents=True, exist_ok=True)
    sources = {name: (BASE + name, checksum) for name, checksum in HASHES.items()}
    sources['utils.py'] = (f'https://raw.githubusercontent.com/FreedomIntelligence/MatCha/{UPSTREAM_REVISION}/src/utils.py', UPSTREAM_HASH)
    for name, (url, expected) in sources.items():
        path = source / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + '.partial')
            urllib.request.urlretrieve(url, temporary)
            if digest(temporary) != expected:
                temporary.unlink()
                raise ValueError(f'Source checksum mismatch: {name}')
            temporary.replace(path)
        if digest(path) != expected:
            raise ValueError(f'Source checksum mismatch: {path}')


def flatten(source):
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    result = []
    for outer, row in enumerate(rows):
        for inner, qa in enumerate(row['vqa']):
            if qa['answer'] not in qa['options'] or any(len(k) != 1 or not k.isupper() for k in qa['options']):
                raise ValueError('Invalid native label/options')
            article = row.get('article_info', {})
            family = row['images'][0]['image_path'].split('-')[0]
            result.append({'task_id': f'matcha-{outer:04d}-{inner:02d}', 'outer_index': outer,
                           'vqa_index': inner, 'source_id': row['id'], 'qa': qa, 'images': row['images'],
                           'article_info': article, 'paper_group': article.get('article_name') or f'corpus:{family}',
                           'group_basis': 'article' if article else 'corpus (paper unavailable)'})
    if len(rows) != 1261 or len(result) != 1500:
        raise ValueError('Unexpected pinned release counts')
    return result


def manifest(records):
    pilot = random.Random(SEED).sample([r['task_id'] for r in records], 10)
    return {'revision': REVISION, 'seed': SEED, 'selection': 'random.Random(42).sample(source-order task IDs, 10), before outcomes',
            'pilot_ids': pilot, 'outer_records': 1261, 'question_count': len(records),
            'tasks': [{k: v for k, v in r.items() if k != 'qa'} | {'topic': r['qa']['topic']} for r in records]}


def selection100(records):
    return {'revision': REVISION, 'seed': SEED,
            'selection': 'random.Random(42).sample(source-order task IDs, 100), prospective; no model outcomes used',
            'task_ids': random.Random(SEED).sample([r['task_id'] for r in records], 100)}


def save_manifest(path, plan):
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(f'Refusing to change existing manifest: {path}')
    else:
        with path.open('x') as stream:
            stream.write(json.dumps(plan, indent=2, ensure_ascii=False))


def safe_member(archive, image_path):
    path = PurePosixPath(image_path)
    if path.is_absolute() or '..' in path.parts or '\\' in image_path or str(path) != image_path:
        raise ValueError(f'Unsafe image path: {image_path}')
    name = 'images/' + image_path
    entries = [i for i in archive.infolist() if i.filename == name]
    if len(entries) != 1:
        raise ValueError(f'Missing/duplicate archive member: {name}')
    entry = entries[0]
    if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size > 100_000_000:
        raise ValueError(f'Unsafe archive member: {name}')
    return entry


def crop_image(archive, spec):
    entry = safe_member(archive, spec['image_path'])
    content = io.BytesIO(archive.read(entry))
    if spec['image_path'].lower().endswith(('.tif', '.tiff')):
        data = tifffile.imread(content)
        # Preserve upstream dtype arithmetic and constant-image normalization.
        with np.errstate(divide='ignore', invalid='ignore'):
            image = Image.fromarray(((data - np.min(data)) / (np.max(data) - np.min(data)) * 255).astype(np.uint8))
    else:
        with Image.open(content) as opened:
            image = opened.copy()
    geometry = spec.get('geometry')
    if geometry is not None:
        xs, ys = [p['x'] for p in geometry], [p['y'] for p in geometry]
        box = max(min(xs), 0), max(min(ys), 0), min(max(xs), image.width), min(max(ys), image.height)
        image = image.crop(box)
    return image


def generate(records, archive_path, output):
    for record in records:
        task_id = record['task_id']
        if PurePosixPath(task_id).name != task_id or task_id in ('', '.', '..') or '\\' in task_id:
            raise ValueError(f'Unsafe task ID: {task_id}')
        if (output / task_id).exists():
            raise FileExistsError(output / task_id)
    output.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for record in records:
            task = output / record['task_id']
            if task.exists():
                raise FileExistsError(task)
            images = [crop_image(archive, spec) for spec in record['images']]
            (task / 'environment/data').mkdir(parents=True)
            (task / 'tests').mkdir()
            (task / 'solution').mkdir()
            paths = []
            for index, image in enumerate(images):
                name = f'image-{index:02d}.png'
                image.save(task / 'environment/data' / name)
                paths.append('/app/data/' + name)
            qa = record['qa']
            prompt = f"Question: {qa['question']}\nAnswer with the option's letter from the given choices directly:\n"
            (task / 'environment/data/input.json').write_text(json.dumps({'system': SYSTEM, 'prompt': prompt,
                'question': qa['question'], 'options': qa['options'], 'images': paths}, ensure_ascii=False, indent=2))
            (task / 'instruction.md').write_text(SYSTEM + '\n\n' + prompt + '\nImages in native order:\n' + '\n'.join(paths)
                + '\n\nWrite exactly one uppercase option letter to /app/answer.txt (surrounding whitespace allowed). No explanation, JSON, or punctuation.\n')
            (task / 'environment/Dockerfile').write_text('FROM python:3.12-slim\nWORKDIR /app\nCOPY data /app/data\n')
            (task / 'task.toml').write_text('schema_version = "1.0"\n[metadata]\ncategory = "matcha"\nsource_revision = "' + REVISION + '"\n[verifier]\ntimeout_sec = 60.0\n[agent]\ntimeout_sec = 900.0\n[environment]\nbuild_timeout_sec = 600.0\ncpus = 1\nmemory_mb = 2048\nstorage_mb = 4096\n')
            (task / 'tests/reference.json').write_text(json.dumps({'answer': qa['answer'], 'options': list(qa['options'])}))
            shutil.copy2(ROOT / 'verify.py', task / 'tests/verify.py')
            (task / 'tests/test.sh').write_text('#!/bin/sh\nset -eu\npython /tests/verify.py\n')
            (task / 'solution/solve.sh').write_text(f"#!/bin/sh\nset -eu\nprintf '%s\\n' '{qa['answer']}' > /app/answer.txt\n")
            for name in ['tests/test.sh', 'solution/solve.sh']:
                (task / name).chmod(0o755)
    return len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'source')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--selection', choices=['pilot10', 'sample100', 'full'], default='pilot10')
    args = parser.parse_args()
    fetch(args.source)
    records = flatten(args.source / 'matcha_vqa_inputs.jsonl')
    plan = manifest(records)
    save_manifest(ROOT / 'manifest.json', plan)
    by_id = {r['task_id']: r for r in records}
    selected = records if args.selection == 'full' else [by_id[i] for i in plan['pilot_ids']]
    if args.selection == 'sample100':
        sample = selection100(records)
        save_manifest(ROOT / 'manifest100.json', sample)
        selected = [by_id[i] for i in sample['task_ids']]
    print(generate(selected, args.source / 'images.zip', args.output or ROOT / 'tasks' / args.selection))


if __name__ == '__main__':
    main()
