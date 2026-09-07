from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import load_dataset

DATASETS = {
    "coding": "wanglab/variant_effect_coding",
    "non-snv": "wanglab/variant_effect_non_snv",
}
DEFAULT_MAX_ITERATIONS = 20
ROOT = Path(__file__).parent


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def label(answer: object) -> str:
    value = str(answer).strip().lower().replace("_", " ")
    if value.startswith("pathogenic"):
        return "pathogenic"
    if value.startswith("benign"):
        return "benign"
    raise ValueError(f"unsupported answer label: {answer!r}")


def generate(
    output: Path,
    setting: str,
    ids: list[int] | None,
    limit: int | None,
    overwrite: bool,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> int:
    dataset_name = DATASETS[setting]
    dataset = load_dataset(dataset_name, split="test")
    selected = list(range(len(dataset))) if ids is None else ids
    if limit is not None:
        selected = selected[:limit]

    generated = 0
    for index in selected:
        row = dataset[index]
        task = output / f"bioreason-vep-{setting}-{index:06d}"
        if task.exists() and not overwrite:
            continue
        if task.exists():
            shutil.rmtree(task)
        for directory in ("environment/data", "solution", "tests/data"):
            (task / directory).mkdir(parents=True, exist_ok=True)

        variant_key = "variant_sequence" if setting == "coding" else "mutated_sequence"
        task_label = label(row["answer"])
        case = {
            "setting": setting,
            "question": row["question"],
            "reference_sequence": row["reference_sequence"],
            "variant_sequence": row[variant_key],
        }
        (task / "environment/data/case.json").write_text(json.dumps(case), encoding="utf-8")
        (task / "tests/data/info.json").write_text(
            json.dumps({"answer": task_label}), encoding="utf-8"
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
        )
        (task / "task.toml").write_text(config, encoding="utf-8")
        solution = (ROOT / "task-template/solution/solve.sh").read_text()
        (task / "solution/solve.sh").write_text(
            solution.replace("{{ answer }}", task_label), encoding="utf-8"
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
