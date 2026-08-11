import argparse
from pathlib import Path

from .adapter import generate

parser = argparse.ArgumentParser(description="Convert MatQnA into Harbor tasks")
parser.add_argument("--output-dir", type=Path, default=Path("datasets/matqna"))
parser.add_argument("--source", type=Path)
parser.add_argument("--task-ids", nargs="+")
parser.add_argument("--limit", type=int)
parser.add_argument("--all", action="store_true")
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()
generate(args.output_dir, args.source, args.task_ids, None if args.all else args.limit, args.overwrite)
