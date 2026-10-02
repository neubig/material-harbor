import argparse
import hashlib
import json
import re
import shutil
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = '80b39472f8a22c4e47ec40b6a9b78c7077af01eb'
SOURCE_SHA256 = 'fcc4b654a7d799ef3e93403939641ca7c51f8868237c8cb4a01001d2bbe58f10'
SEED = 'matrix-text-pilot-20260814-v1'
MEASUREMENT_SEED = 'matrix-text-fixed100-20260814-v2'
IMAGE = 'python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea'


SOURCE_URL = f'https://huggingface.co/datasets/radical-ai/MATRIX/resolve/{REVISION}/test/test.jsonl'
MAX_SOURCE_BYTES = 16 * 1024 * 1024


def validate_source(raw):
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('Pinned source checksum mismatch')
    rows = [json.loads(line) for line in raw.splitlines()]
    validate_rows(rows)
    return rows


def validate_rows(rows):
    if (len(rows) != 470 or sum(r['type'] == 'text' for r in rows) != 220
            or sum(r['type'] == 'vision' for r in rows) != 250
            or len({r['qid'] for r in rows}) != 470):
        raise ValueError('Unexpected MATRIX split counts or duplicate qids')
    for row in rows:
        if not re.fullmatch('[0-9a-f]{32}', row['qid']):
            raise ValueError('Invalid qid')
        if not all(isinstance(row[k], str) and row[k].strip()
                   for k in ('question', 'answer', 'kind')):
            raise ValueError('Empty or non-text question, answer, or kind')


def load_rows(source=None):
    return validate_source((source or ROOT / 'source/test.jsonl').read_bytes())


def fetch_source(destination=None, *, local_file=None):
    destination = destination or ROOT / 'source/test.jsonl'
    if destination.exists():
        load_rows(destination)
        return destination
    if local_file is not None:
        with local_file.open('rb') as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=90) as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError('Source exceeds download size limit')
    validate_source(raw)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only checksum-verified bytes; never overwrite an existing cache.
    with tempfile.NamedTemporaryFile(dir=destination.parent) as temporary:
        temporary.write(raw)
        temporary.flush()
        try:
            destination.hardlink_to(temporary.name)
        except FileExistsError:
            load_rows(destination)
    return destination


def select(rows, count=10, seed=SEED):
    return sorted((r for r in rows if r['type'] == 'text'),
                  key=lambda r: (hashlib.sha256((seed + ':' + r['qid']).encode()).hexdigest(), r['qid']))[:count]


def build(output, source=None):
    selected = select(load_rows(source), 100, MEASUREMENT_SEED)
    output.mkdir(parents=True, exist_ok=False)
    for row in selected:
        task = output / ('matrix-' + row['qid'])
        environment = task / 'environment'
        tests = task / 'tests'
        environment.mkdir(parents=True)
        tests.mkdir()
        (task / 'instruction.md').write_text(row['question'] + '\n\nWrite your complete explanatory answer as plain UTF-8 text to /logs/artifacts/answer.txt (at most 65536 bytes).\n')
        (tests / 'gold.json').write_text(json.dumps(row, indent=2))
        shutil.copyfile(ROOT / 'verify.py', tests / 'verify.py')
        (tests / 'test.sh').write_text('#!/bin/sh\nset -eu\npython -I /tests/verify.py\n')
        (tests / 'test.sh').chmod(0o755)
        (task / 'task.toml').write_text(f'''schema_version = "1.0"
[metadata]
dataset = "radical-ai/MATRIX"
source_revision = "{REVISION}"
protocol = "matrix-five-level-official-style-v2"
[agent]
timeout_sec = 600.0
[verifier]
timeout_sec = 120.0
environment_mode = "separate"
[verifier.environment]
docker_image = "{IMAGE}"
network_mode = "public"
cpus = 1
memory_mb = 512
[environment]
docker_image = "{IMAGE}"
workdir = "/tmp"
build_timeout_sec = 600.0
cpus = 1
memory_mb = 4096
storage_mb = 2048
network_mode = "public"
''')
    manifest = {'revision': REVISION, 'source_sha256': SOURCE_SHA256, 'seed': MEASUREMENT_SEED,
                'selection': 'lowest sha256(seed + colon + qid) among all 220 text tasks; no category filtering',
                'tasks': [{'qid': r['qid'], 'kind': r['kind']} for r in selected]}
    (output.parent / 'manifest100-v2.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=ROOT / 'source/test.jsonl')
    parser.add_argument('--fetch', action='store_true', help='Explicitly fetch the pinned source if absent')
    parser.add_argument('--source-file', type=Path, help='Import pinned bytes offline (requires --fetch)')
    parser.add_argument('--fetch-only', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/fixed100-v2/tasks')
    args = parser.parse_args()
    if (args.source_file or args.fetch_only) and not args.fetch:
        parser.error('--source-file and --fetch-only require --fetch')
    if args.fetch:
        fetch_source(args.source, local_file=args.source_file)
    if not args.fetch_only:
        build(args.output, args.source)
