"""Build the frozen MATRIX diagnostic cohort; no outcome-based filtering."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
REVISION = '80b39472f8a22c4e47ec40b6a9b78c7077af01eb'
SOURCE_SHA = 'fcc4b654a7d799ef3e93403939641ca7c51f8868237c8cb4a01001d2bbe58f10'
UNIVERSE_SHA = '9f768b1b6fdc9183899c1790d05d021e2f9e6b744d3ed51f52ee569ed2bde38d'
SAMPLE_SHA = '832d9ca5f5bf4f844096c8787ec556c597a6fbbbdc71c0aa905c8a2407fae592'

DOCKERFILE = '''FROM python:3.12-slim
RUN python -m venv /opt/openhands-sdk-venv && /opt/openhands-sdk-venv/bin/pip install --no-cache-dir openhands-sdk openhands-tools fastapi
WORKDIR /app
COPY data /app/data
'''
TASK_CONFIG = '''version = "1.0"

[metadata]
category = "materials-science"
benchmark = "MATRIX"
status = "diagnostic-only"
source_revision = "{revision}"
source_qid = "{qid}"

[agent]
timeout_sec = 1200.0

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 4096
storage_mb = 8192

[verifier]
environment_mode = "separate"
timeout_sec = 300.0

[verifier.env]
MATRIX_JUDGE_API_KEY = "${{LLM_API_KEY}}"
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path, expected):
    data = path.read_bytes()
    if digest(data) != expected:
        raise ValueError('Pinned input changed: ' + str(path))
    return data


def load_inputs(workspace=WORKSPACE):
    research = workspace / 'research-matrix-continuation'
    source = checked_bytes(workspace / 'research-mmm/matrix-test-test.jsonl', SOURCE_SHA)
    universe_bytes = checked_bytes(research / 'universe.json', UNIVERSE_SHA)
    sample_bytes = checked_bytes(research / 'sample.json', SAMPLE_SHA)
    rows = [json.loads(line) for line in source.splitlines()]
    universe = json.loads(universe_bytes)
    sample = json.loads(sample_bytes)
    records = universe['records']
    ids = [row['qid'] for row in records]
    selected = random.Random(20261008).sample(records, 100)
    if (len(rows) != 470 or len(ids) != 249 or len(set(ids)) != 249 or ids != sorted(ids)
            or sample['sample_qids_in_draw_order'] != [row['qid'] for row in selected]):
        raise ValueError('Frozen cohort mismatch')
    by_id = {row['qid']: row for row in rows}
    for entry in selected:
        row = by_id[entry['qid']]
        if row['type'] != 'vision' or not row['question'] or not row['answer']:
            raise ValueError('Invalid sampled source row; never substitute')
        checked_bytes(workspace / 'research-mmm/matrix-test-images' / Path(entry['image_path']).name,
                      entry['image_sha256'])
    return by_id, selected, universe_bytes, sample_bytes


def generate(output=ROOT / 'tasks', workspace=WORKSPACE):
    output = output.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite frozen generated tasks')
    by_id, selected, universe, sample = load_inputs(workspace)
    output.mkdir(parents=True)
    (ROOT / 'universe.json').write_bytes(universe)
    (ROOT / 'sample.json').write_bytes(sample)
    for entry in selected:
        row = by_id[entry['qid']]
        task = output / ('matrix-' + row['qid'])
        for directory in ('environment/data', 'tests', 'solution'):
            (task / directory).mkdir(parents=True)
        image = workspace / 'research-mmm/matrix-test-images' / Path(entry['image_path']).name
        shutil.copyfile(image, task / 'environment/data/image.png')
        shutil.copyfile(image, task / 'tests/image.png')
        (task / 'environment/Dockerfile').write_text(DOCKERFILE)
        (task / 'instruction.md').write_text(
            '# MATRIX materials-science explanation\n\n' + row['question'] + '\n\n'
            'The {image} marker refers to `/app/data/image.png`. Inspect the image. '
            'Preserve the full requested caption/explanation, including relevant scientific '
            'interpretation and evidence; a technique label alone is not a response. '
            'Use only the supplied image, context and scientific knowledge; do not retrieve '
            'benchmark answer keys. Clearly qualify inferences not established by the figure.\n\n'
            'Write your full answer as a regular UTF-8 file `/logs/artifacts/answer.txt` '
            '(nonempty, no NUL bytes, at most 65536 bytes). Do not create a symlink.\n\n'
            'You have a working budget of 40 turns.\n')
        (task / 'task.toml').write_text(TASK_CONFIG.format(revision=REVISION, qid=row['qid']))
        (task / 'tests/gold.json').write_text(json.dumps({'qid': row['qid'],
            'question': row['question'], 'answer': row['answer']}, ensure_ascii=False))
        shutil.copyfile(ROOT / 'verify.py', task / 'tests/verify.py')
        (task / 'tests/test.sh').write_text('#!/bin/sh\nset -eu\nexec /usr/local/bin/python -I /tests/verify.py\n')
        (task / 'tests/test.sh').chmod(0o755)
        manifest = {name: digest((task / 'tests' / name).read_bytes())
                    for name in ('gold.json', 'image.png', 'verify.py', 'test.sh')}
        (task / 'tests/manifest.json').write_text(json.dumps(manifest, indent=2))
        encoded = base64.b64encode(row['answer'].encode()).decode('ascii')
        (task / 'solution/solve.sh').write_text('#!/bin/sh\nset -eu\nmkdir -p /logs/artifacts\n'
            "printf '%s' '" + encoded + "' | base64 -d > /logs/artifacts/answer.txt\n")
        (task / 'solution/solve.sh').chmod(0o755)
    return len(selected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'tasks')
    args = parser.parse_args()
    print(generate(args.output))
