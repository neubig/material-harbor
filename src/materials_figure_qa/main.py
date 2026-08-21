from __future__ import annotations
import argparse, base64, json, shutil
from pathlib import Path
from datasets import load_dataset
ROOT = Path(__file__).parent

def generate(output: Path, split: str, limit: int | None = None, overwrite: bool = False) -> int:
    ds = load_dataset('gneubig/materials-figure-qa', data_files={split: f'{split}.parquet'}, split=split)
    count = 0
    for i in range(min(len(ds), limit) if limit else len(ds)):
        row = ds[i]; task = output / f'materials-figure-qa-{split}-{i:06d}'
        if task.exists() and not overwrite: continue
        if task.exists(): shutil.rmtree(task)
        (task / 'environment/data').mkdir(parents=True); (task / 'solution').mkdir(); (task / 'tests/data').mkdir(parents=True)
        row['image'].save(task / 'environment/data/image.png')
        (task / 'instruction.md').write_text((ROOT / 'task-template/instruction.md').read_text().replace('{{ question }}', row['question']))
        (task / 'task.toml').write_text((ROOT / 'task-template/task.toml').read_text().replace('{{ task_id }}', f'{split}-{i:06d}'))
        encoded = base64.b64encode(row['answer'].encode()).decode()
        (task / 'solution/solve.sh').write_text((ROOT / 'task-template/solution/solve.sh').read_text().replace('{{ answer }}', encoded)); (task / 'solution/solve.sh').chmod(0o755)
        for name in ('test.sh', 'verify.py'): shutil.copy2(ROOT / 'task-template/tests' / name, task / 'tests' / name)
        (task / 'tests/test.sh').chmod(0o755)
        (task / 'tests/data/info.json').write_text(json.dumps({'question': row['question'], 'answer': row['answer'], 'candidate_id': row['candidate_id'], 'difficulty': row['difficulty']}))
        count += 1
    return count

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output-dir', type=Path, default=Path('datasets/materials-figure-qa')); ap.add_argument('--split', choices=['validation', 'test', 'both'], default='both'); ap.add_argument('--limit', type=int); ap.add_argument('--overwrite', action='store_true'); args = ap.parse_args()
    splits = [args.split] if args.split != 'both' else ['validation', 'test']
    print(sum(generate(args.output_dir / s, s, args.limit, args.overwrite) for s in splits))
if __name__ == '__main__': main()
