from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset

if __package__:
    from .answer_schema import excluded_rows, parse_source_answer
else:
    from answer_schema import excluded_rows, parse_source_answer

DATASETS = {
    "coding": "wanglab/variant_effect_coding",
    "non-snv": "wanglab/variant_effect_non_snv",
}
REVISIONS = {
    "coding": "7684ab820d728fb5362328757079a1390a3f43bb",
    "non-snv": "aeba75d031b8cb507b3962ab486aa22700afa31c",
}
DEFAULT_MAX_ITERATIONS = 20
ROOT = Path(__file__).parent


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def generate(
    output: Path,
    setting: str,
    ids: list[int] | None,
    limit: int | None,
    overwrite: bool,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> int:
    if (
        output.exists()
        and any(output.glob(f"bioreason-vep-{setting}-*"))
        and not overwrite
    ):
        raise ValueError(
            "Use a fresh output directory or --overwrite to avoid mixing old and new task schemas"
        )
    dataset_name = DATASETS[setting]
    dataset = load_dataset(dataset_name, split="test", revision=REVISIONS[setting])
    variant_key = "variant_sequence" if setting == "coding" else "mutated_sequence"
    exclusions = excluded_rows(dataset, variant_key)
    selected = list(range(len(dataset))) if ids is None else ids
    if limit is not None:
        selected = selected[:limit]

    generated = 0
    excluded = []
    for index in selected:
        row = dataset[index]
        task = output / f"bioreason-vep-{setting}-{index:06d}"
        if index in exclusions:
            if task.exists() and overwrite:
                shutil.rmtree(task)
            excluded.append(
                {
                    "index": index,
                    "reason": exclusions[index],
                    "source_answer": row["answer"],
                }
            )
            continue
        answer = parse_source_answer(row["answer"])
        if task.exists():
            shutil.rmtree(task)
        for directory in ("environment/data", "solution", "tests/data"):
            (task / directory).mkdir(parents=True, exist_ok=True)

        case = {
            "setting": setting,
            "question": row["question"],
            "reference_sequence": row["reference_sequence"],
            "variant_sequence": row[variant_key],
        }
        (task / "environment/data/case.json").write_text(
            json.dumps(case), encoding="utf-8"
        )
        (task / "tests/data/info.json").write_text(
            json.dumps(
                {
                    "answer": answer,
                    "source_answer": row["answer"],
                    "source_question": row["question"],
                    "source_index": index,
                    "source_revision": REVISIONS[setting],
                }
            ),
            encoding="utf-8",
        )

        instruction = (ROOT / "task-template/instruction.md").read_text()
        instruction = instruction.replace("{{ setting }}", setting).replace(
            "{{ max_iterations }}", str(max_iterations)
        )
        (task / "instruction.md").write_text(instruction, encoding="utf-8")
        config = (ROOT / "task-template/task.toml").read_text()
        config = (
            config.replace("{{ setting }}", setting)
            .replace("{{ dataset_name }}", dataset_name)
            .replace("{{ task_id }}", f"{index:06d}")
            .replace("{{ source_revision }}", REVISIONS[setting])
        )
        (task / "task.toml").write_text(config, encoding="utf-8")
        solution = (ROOT / "task-template/solution/solve.sh").read_text()
        (task / "solution/solve.sh").write_text(
            solution.replace(
                "{{ answer_hex }}", json.dumps(answer).encode("utf-8").hex()
            ),
            encoding="utf-8",
        )
        (task / "solution/solve.sh").chmod(0o755)
        for name in ("test.sh", "verify.py"):
            shutil.copy2(ROOT / f"task-template/tests/{name}", task / f"tests/{name}")
        (task / "tests/test.sh").chmod(0o755)
        shutil.copy2(
            ROOT / "task-template/environment/Dockerfile",
            task / "environment/Dockerfile",
        )
        generated += 1
    output.mkdir(parents=True, exist_ok=True)
    (output / f"excluded-{setting}.json").write_text(json.dumps(excluded, indent=2))
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BioReason variant-effect benchmarks into Harbor tasks"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("datasets/bioreason-vep")
    )
    parser.add_argument("--setting", choices=[*DATASETS, "both"], default="both")
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--max-iterations",
        type=positive_int,
        default=DEFAULT_MAX_ITERATIONS,
        help="OpenHands SDK iteration budget stated in generated instructions",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    settings = DATASETS if args.setting == "both" else (args.setting,)
    for setting in settings:
        generate(
            args.output_dir,
            setting,
            args.task_ids,
            None if args.all else args.limit,
            args.overwrite,
            args.max_iterations,
        )


if __name__ == "__main__":
    main()
