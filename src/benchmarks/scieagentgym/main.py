import argparse
from pathlib import Path

from .adapter import generate


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SciAgentGYM into Harbor tasks")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/scieagentgym"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--include-single", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    generate(
        args.output_dir,
        source=args.source,
        ids=args.task_ids,
        limit=None if args.all else args.limit,
        include_single=args.include_single,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
