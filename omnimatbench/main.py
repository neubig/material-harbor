from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
REVISION = 'f933f03c733bb378ce5dd85b96452d9d5683c17e'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def validate_sources():
    hashes = json.loads((SOURCE / 'sha256.json').read_text())
    if any(not (SOURCE / name).is_file() for name in hashes):
        raise FileNotFoundError('Run python -m omnimatbench.fetch_source to fetch pinned sources')
    for name, expected in hashes.items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Source hash mismatch: {name}')
    return hashes


def load_cohort():
    validate_sources()
    cohort, excluded = [], []
    for path in sorted((SOURCE / 'cal').rglob('*.jsonl')):
        category = path.relative_to(SOURCE).parts[1]
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            key = f"cal/{category}/{row['id']}"
            item = {'key': key, 'source': str(path.relative_to(SOURCE)), 'line': line_no,
                    'question_sha256': hashlib.sha256(row['question'].encode()).hexdigest()}
            if row.get('image_url'):
                excluded.append(dict(item, reason='nonempty image_url'))
            else:
                if row.get('multimodal') or not row.get('final_answer_list'):
                    raise ValueError(f'Unexpected text-only schema: {key}')
                cohort.append(dict(item, row=row))
    if len(cohort) != 360 or len(excluded) != 142 or len({x['key'] for x in cohort}) != 360:
        raise ValueError('Pinned cohort count or uniqueness mismatch')
    return cohort, excluded


def select(cohort, count, seed):
    if not 1 <= count <= len(cohort):
        raise ValueError('count outside eligible cohort')
    return random.Random(seed).sample(sorted(x['key'] for x in cohort), count)


def generate(output, count=10, seed=20260814):
    cohort, excluded = load_cohort()
    selected = select(cohort, count, seed)
    if output.exists():
        raise FileExistsError('Use a fresh output directory; frozen pilots are not overwritten')
    output.mkdir(parents=True)
    manifest = {'revision': REVISION, 'subset': 'CAL text-only (image_url absent/empty)',
                'seed': seed, 'selection': 'random.Random(seed).sample(sorted(namespaced_ids), count)',
                'selected': selected, 'source_sha256': validate_sources(),
                'cohort': [{k: v for k, v in x.items() if k != 'row'} for x in cohort],
                'excluded': excluded, 'model_outcomes': None}
    write_json(output / 'manifest.json', manifest)
    lookup = {x['key']: x for x in cohort}
    for key in selected:
        record = lookup[key]
        row = record['row']
        task = output / ('omnimatbench-' + key.replace('/', '-'))
        for sub in ['environment', 'tests/native', 'solution']:
            (task / sub).mkdir(parents=True)
        write_json(task / 'environment/case.json', {k: row[k] for k in ['question', 'final_answer_format']})
        (task / 'environment/Dockerfile').write_text('FROM python:3.12-slim\nWORKDIR /app\nCOPY case.json /app/case.json\n')
        (task / 'instruction.md').write_text(
            '# OmniMatBench CAL — text-only subset\n\n'
            'Solve the official question in `/app/case.json`. You may use Python and shell tools. '
            'The question is preserved verbatim. Follow its units and rounding requirements. '
            'Write your final answer to `/app/answer.json` as a valid JSON list, following '
            '`final_answer_format` grouping and slot order; fill each blank with a string or finite number. '
            'Do not include reasoning, Markdown fences, an object wrapper, or <answer> tags in that file. '
            'Any requested table or derivation may be saved separately.\n\n'
            'Scoring uses the pinned native exact numeric/text comparator, not a scientific tolerance: '
            'numeric predictions are rounded to the decimal-place count in the reference string; '
            'otherwise normalized text is compared. No unit conversion or general symbolic equivalence '
            'is provided. All flattened answer slots must match for reward 1. Missing, malformed, '
            'empty or extra/missing-slot answers receive 0.\n')
        (task / 'task.toml').write_text(f'''schema_version = "1.0"
[task]
name = "omnimatbench/{key.replace('/', '-')}"
[metadata]
dataset = "OmniMatBench CAL text-only subset"
source_revision = "{REVISION}"
source_id = "{key}"
source = "https://github.com/wanhaoliu/OmniMatBench"
grading = "native-exact-conjunctive; strict JSON-list transport"
[agent]
timeout_sec = 600.0
[verifier]
timeout_sec = 60.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 4096
''')
        write_json(task / 'tests/gold.json', row['final_answer_list'])
        shutil.copy2(ROOT / 'verify.py', task / 'tests/verify.py')
        shutil.copy2(SOURCE / 'LICENSE', task / 'tests/native/LICENSE')
        for name in ['eval_cal_results.py', 'omnimat_paths.py']:
            shutil.copy2(SOURCE / 'scripts/cal' / name, task / 'tests/native' / name)
        (task / 'tests/test.sh').write_text('#!/bin/sh\nset -eu\npython /tests/verify.py\n')
        (task / 'tests/test.sh').chmod(0o755)
        answer = json.dumps(row['final_answer_list'], ensure_ascii=False)
        (task / 'solution/solve.sh').write_text("#!/bin/sh\nset -eu\nmkdir -p /app\ncat > /app/answer.json <<'OMNIMAT_GOLD'\n" + answer + '\nOMNIMAT_GOLD\n')
        (task / 'solution/solve.sh').chmod(0o755)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20260814)
    args = parser.parse_args()
    manifest = generate(args.output_dir, args.count, args.seed)
    print(json.dumps({'tasks': len(manifest['selected']), 'cohort': len(manifest['cohort']), 'output': str(args.output_dir)}))


if __name__ == '__main__':
    main()
