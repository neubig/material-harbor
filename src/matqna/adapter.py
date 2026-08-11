from __future__ import annotations

import ast
import base64
import json
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

DATA_URL = "https://huggingface.co/datasets/richardhzgg/matQnA/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
ROOT = Path(__file__).parent


def parse_value(value):
    if value is None:
        return None
    text = str(value).strip()
    if text in {"None", "nan", ""}:
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def choices_text(value: object) -> str:
    choices = parse_value(value)
    if not choices:
        return ""
    if isinstance(choices, (list, tuple)):
        return "Choices:\n" + "\n".join(map(str, choices))
    return f"Choices:\n{choices}"


def records(source: Path | None = None) -> pd.DataFrame:
    source = source or Path("/tmp/matQnA.parquet")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, source)
    return pd.read_parquet(source)


def generate(
    output: Path,
    source: Path | None = None,
    ids: list[str] | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> int:
    frame = records(source)
    selected = list(range(len(frame))) if ids is None else [int(value) for value in ids]
    if limit is not None:
        selected = selected[:limit]
    for index in selected:
        row = frame.iloc[index]
        task = output / f"matqna-{index:06d}"
        if task.exists() and not overwrite:
            continue
        if task.exists():
            shutil.rmtree(task)
        (task / "environment/data").mkdir(parents=True)
        (task / "solution").mkdir()
        (task / "tests/data").mkdir(parents=True)

        image = parse_value(row.images)
        image = image if isinstance(image, str) else str(image)
        (task / "environment/data/image.png").write_bytes(base64.b64decode(image))
        shutil.copy2(ROOT / "task-template/environment/Dockerfile", task / "environment/Dockerfile")

        answer = parse_value(row.scholar_reference_answer)
        answer = "" if answer is None else str(answer)
        encoded_answer = base64.b64encode(answer.encode("utf-8")).decode("ascii")
        instruction = (ROOT / "task-template/instruction.md").read_text()
        instruction = instruction.replace("{{ category }}", str(row.category))
        instruction = instruction.replace("{{ question }}", str(row.question))
        instruction = instruction.replace("{{ choices }}", choices_text(row.choices))
        (task / "instruction.md").write_text(instruction)

        config = (ROOT / "task-template/task.toml").read_text()
        (task / "task.toml").write_text(config.replace("{{ task_id }}", f"{index:06d}"))
        solution = (ROOT / "task-template/solution/solve.sh").read_text()
        (task / "solution/solve.sh").write_text(solution.replace("{{ answer }}", encoded_answer))
        (task / "solution/solve.sh").chmod(0o755)
        for name in ("test.sh", "verify.py"):
            shutil.copy2(ROOT / f"task-template/tests/{name}", task / f"tests/{name}")
        (task / "tests/test.sh").chmod(0o755)
        (task / "tests/data/info.json").write_text(
            json.dumps(
                {
                    "qa_type": str(row.qa_type),
                    "answer": answer,
                    "category": str(row.category),
                    "source_row": index,
                },
                ensure_ascii=False,
            )
        )
    return len(selected)
