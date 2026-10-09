"""Convert the pinned public CSMBench MCQ release into Harbor tasks."""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://huggingface.co/datasets/lututu/CSMBench'
REVISION = '6dfc2ba8aea24dd04977801c68d2d6f24f76bc02'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indices = [row['index'] for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate source index')
    return {row['index']: row for row in rows}


def validate_row(row):
    options = json.loads(row['options'])
    labels = [option['label'] for option in options]
    captions = [option['caption'] for option in options]
    if len(labels) < 2 or len(labels) != len(set(labels)) or any(not re.fullmatch('[A-Z]', label) for label in labels):
        raise ValueError(f"Invalid option labels for row {row['index']}")
    if len(captions) != len(set(captions)) or any(not isinstance(caption, str) or not caption.strip() for caption in captions):
        raise ValueError(f"Ambiguous or empty captions for row {row['index']}")
    if row['correct_answer'] not in labels:
        raise ValueError(f"Missing source key for row {row['index']}")
    return options


def generate(metadata, image_dir, output, ids):
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    rows = load_rows(metadata)
    selected = [int(value) for value in ids]
    if len(selected) != len(set(selected)) or any(index not in rows for index in selected):
        raise ValueError('Duplicate or unknown task ID')
    output.mkdir(parents=True)
    manifest = {'source': SOURCE, 'revision': REVISION, 'tasks': []}
    for index in selected:
        row = rows[index]
        options = validate_row(row)
        source_image = image_dir / row['file_name']
        with Image.open(source_image) as image:
            image.verify()
        task_id = f'csmbench-{index:06d}'
        task = output / task_id
        (task / 'environment/data').mkdir(parents=True)
        (task / 'tests').mkdir()
        (task / 'solution').mkdir()
        suffix = source_image.suffix.lower()
        image_name = f'image{suffix}'
        shutil.copy2(source_image, task / 'environment/data' / image_name)
        option_text = '\n'.join(f"{option['label']}. {option['caption']}" for option in options)
        instruction = f'''# CSMBench cross-scale materials image matching

Inspect `/app/data/{image_name}` with the image-viewing tool. Select the one caption that best and most specifically describes the supplied materials-science image.

Scale: {row['scale']}

Options:
{option_text}

Write exactly one uppercase option letter to `/app/answer.txt`, optionally followed by one LF newline. Do not include prose, JSON, spaces, or additional text. Missing, malformed, or incorrect output receives zero.
'''
        (task / 'instruction.md').write_text(instruction)
        shutil.copy2(ROOT / 'Dockerfile', task / 'environment/Dockerfile')
        shutil.copy2(ROOT / 'task.toml', task / 'task.toml')
        shutil.copy2(ROOT / 'verify.py', task / 'tests/verify.py')
        (task / 'tests/test.sh').write_text('#!/bin/sh\nset -eu\npython3 /tests/verify.py\n')
        (task / 'tests/test.sh').chmod(0o755)
        (task / 'tests/gold.json').write_text(json.dumps({'answer': row['correct_answer'], 'letters': [option['label'] for option in options]}))
        (task / 'solution/solve.sh').write_text(f"#!/bin/sh\nset -eu\nprintf '%s\\n' '{row['correct_answer']}' > /app/answer.txt\n")
        (task / 'solution/solve.sh').chmod(0o755)
        public_files = {str(path.relative_to(task)): digest(path) for path in sorted(task.rglob('*')) if path.is_file() and path.relative_to(task).parts[0] not in {'tests', 'solution'}}
        manifest['tasks'].append({'task_id': task_id, 'source_index': index, 'file_name': row['file_name'], 'scale': row['scale'], 'paper_folder_name': row['paper_folder_name'], 'files': public_files})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return len(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--image-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    args = parser.parse_args()
    print(generate(args.metadata, args.image_dir, args.output_dir, args.ids))


if __name__ == '__main__':
    main()
