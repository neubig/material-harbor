"""Convert the pinned public MatCha release into one Harbor task per VQA item."""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image, ImageSequence, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parent
SOURCE = 'https://huggingface.co/datasets/FreedomIntelligence/MatCha'
REVISION = 'dc52d6016053c67ebdf41719d6cfaa0603b154fa'
SOURCE_HASHES = {
    'data/matcha_vqa_inputs.jsonl': 'd80382468646ab7d58ef638780754f5a602e6a2c2a98c0564dfd260ae89bd916',
    'data/images.zip': '1dea6b1065d33ca09266989fecd3543ab7dbfb171b923efd9584e01664021e35',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def group(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def records(rows):
    result = []
    seen = set()
    for row_index, row in enumerate(rows):
        if row['id'] in seen:
            raise ValueError(f'Duplicate source record ID: {row["id"]}')
        seen.add(row['id'])
        if not row['images'] or not row['vqa']:
            raise ValueError('Empty images or VQA list')
        image_paths = [image['image_path'] for image in row['images']]
        for path in image_paths:
            parsed = PurePosixPath(path)
            if parsed.is_absolute() or '..' in parsed.parts or '\\' in path:
                raise ValueError(f'Unsafe image path: {path}')
        for index, qa in enumerate(row['vqa']):
            options = qa['options']
            if (not isinstance(qa['question'], str) or not isinstance(options, dict)
                    or not options or any(not re.fullmatch('[A-Z]', k) for k in options)
                    or any(not isinstance(v, str) for v in options.values())
                    or qa['answer'] not in options):
                raise ValueError('Invalid source question/options/answer')
            result.append({'id': f'{row["id"]}::vqa-{index:03d}',
                           'source_id': row['id'], 'row_index': row_index, 'vqa_index': index,
                           'question_group': group([qa['question'], options]),
                           'image_group': group(sorted(set(image_paths))),
                           'image_paths': image_paths, 'row': row, 'qa': qa})
    return result


def select(items, ids=None, limit=None):
    if ids is not None and limit is not None:
        raise ValueError('IDs and limit are mutually exclusive')
    if limit is not None and limit <= 0:
        raise ValueError('Limit must be positive')
    if ids is not None:
        requested = set(ids)
        if not requested or requested - {item['id'] for item in items}:
            raise ValueError('Unknown or empty VQA IDs (use source-id::vqa-000)')
        return [item for item in items if item['id'] in requested]
    return items[:limit] if limit is not None else items


def generate(items, archive, output, provenance=None):
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'adapter_version': 1, 'source': provenance or {'verified': False},
                'selection_order': 'source row order, then zero-based VQA index',
                'image_policy': 'Full original files; no cropping. TIFF frames get RGB PNG previews.',
                'pillow_version': PILLOW_VERSION, 'tasks': []}
    with zipfile.ZipFile(archive) as images:
        for item in items:
            task_id = 'matcha-' + group(item['id'])[:20]
            task = output / task_id
            data = task / 'environment/data'
            data.mkdir(parents=True)
            (task / 'tests').mkdir()
            (task / 'solution').mkdir()
            assets = []
            for index, image in enumerate(item['row']['images']):
                original = f'original-{index:03d}{PurePosixPath(image["image_path"]).suffix}'
                raw = images.read('images/' + image['image_path'])
                (data / original).write_bytes(raw)
                views = [original]
                if Path(original).suffix.lower() not in {'.jpg', '.jpeg', '.png', '.webp', '.gif'}:
                    views = []
                    with Image.open(data / original) as source_image:
                        for frame_index, frame in enumerate(ImageSequence.Iterator(source_image)):
                            view = f'view-{index:03d}-{frame_index:03d}.png'
                            frame.convert('RGB').save(data / view)
                            views.append(view)
                assets.append({'source_path': image['image_path'], 'original': original,
                               'view': views[0], 'views': views, 'sha256': digest(data / original),
                               'view_sha256': {view: digest(data / view) for view in views}})
            qa = item['qa']
            public = {key: qa[key] for key in ('question', 'options', 'topic') if key in qa}
            public.update({'source_id': item['source_id'], 'vqa_index': item['vqa_index'],
                           'images': item['row']['images'], 'assets': assets})
            write_json(data / 'question.json', public)
            paths = '\n'.join(f'- /app/data/{view}' for asset in assets for view in asset['views'])
            instruction = (
                '# MatCha\n\nInspect every supplied image with your image-viewing tool '
                '(OpenHands: use file_editor view on the paths below). Do not answer without inspecting them.\n\n'
                + paths + '\n\nOriginal question (unchanged):\n' + qa['question']
                + '\n\nOriginal options:\n' + json.dumps(qa['options'], ensure_ascii=False, indent=2)
                + '\n\n/app/data/question.json contains the original image-region geometry and '
                'image-to-file mapping. Images are full figures, not crops; use that geometry to locate '
                'the intended region. Original files are also available alongside previews.\n\n'
                'Write exactly one uppercase option letter to /app/answer.txt, optionally followed '
                'by one LF newline. No explanation, JSON, spaces, or additional text. '
                'Missing, invalid, or incorrect output receives zero.\n'
            )
            (task / 'instruction.md').write_text(instruction, encoding='utf-8')
            shutil.copyfile(ROOT / 'Dockerfile', task / 'environment/Dockerfile')
            shutil.copyfile(ROOT / 'task.toml', task / 'task.toml')
            shutil.copyfile(ROOT / 'verify.py', task / 'tests/verify.py')
            write_json(task / 'tests/gold.json', {'answer': qa['answer'], 'letters': list(qa['options'])})
            (task / 'tests/test.sh').write_text('#!/bin/sh\nset -eu\npython3 /tests/verify.py\n')
            (task / 'solution/solve.sh').write_text(
                f"#!/bin/sh\nset -eu\nprintf '%s\\n' '{qa['answer']}' > /app/answer.txt\n")
            for script in (task / 'tests/test.sh', task / 'solution/solve.sh'):
                script.chmod(0o755)
            entry = {key: value for key, value in item.items() if key not in {'row', 'qa'}}
            entry.update({'task_id': task_id, 'assets': assets,
                          'article_info': item['row'].get('article_info'),
                          'files': {str(p.relative_to(task)): digest(p)
                                    for p in sorted(task.rglob('*'))
                                    if p.is_file() and p.relative_to(task).parts[0] not in {'tests', 'solution'}}})
            manifest['tasks'].append(entry)
    write_json(output / 'manifest.json', manifest)
    return len(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-json', type=Path, required=True,
                        help=f'{SOURCE}/resolve/{REVISION}/data/matcha_vqa_inputs.jsonl')
    parser.add_argument('--images-zip', type=Path, required=True,
                        help=f'{SOURCE}/resolve/{REVISION}/data/images.zip')
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--ids', nargs='+', help='Exact source-id::vqa-000 identifiers')
    selection.add_argument('--limit', type=int, help='First N VQA items in pinned source order')
    selection.add_argument('--all', action='store_true', help='All 1500 VQA items, without deduplication')
    args = parser.parse_args()
    for name, path in [('data/matcha_vqa_inputs.jsonl', args.source_json), ('data/images.zip', args.images_zip)]:
        if digest(path) != SOURCE_HASHES[name]:
            parser.error(f'Pinned source checksum mismatch: {path}')
    rows = [json.loads(line) for line in args.source_json.read_text(encoding='utf-8').splitlines() if line.strip()]
    selected = select(records(rows), ids=args.ids, limit=args.limit)
    provenance = {'url': SOURCE, 'revision': REVISION, 'license': 'Apache-2.0',
                  'verified': True, 'sha256': SOURCE_HASHES, 'source_records': len(rows),
                  'source_vqa_items': sum(len(row['vqa']) for row in rows),
                  'selection': {'ids': args.ids, 'limit': args.limit, 'all': args.all}}
    print(generate(selected, args.images_zip, args.output_dir, provenance))


if __name__ == '__main__':
    main()
