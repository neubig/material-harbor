from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

DATA_URL = "https://github.com/CMarsRover/SciAgentGYM/archive/refs/heads/main.tar.gz"
CACHE = Path("/tmp/scieagentgym-source")
ROOT = Path(__file__).parent


def _safe_extract(bundle: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in bundle.getmembers():
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"archive member escapes destination: {member.name}")
    bundle.extractall(destination)


def source_root(source: Path | None = None) -> Path:
    if source is not None:
        return source
    if not (CACHE / "dataset").exists():
        archive = CACHE.with_suffix(".tar.gz")
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, archive)
        with tarfile.open(archive) as bundle:
            _safe_extract(bundle, CACHE.parent)
        extracted = next(CACHE.parent.glob("SciAgentGYM-*"))
        extracted.rename(CACHE)
        archive.unlink(missing_ok=True)
    return CACHE


def cases(source: Path, include_single: bool) -> list[dict]:
    paths = [source / "dataset" / "refine_merged_multi_questions.json"]
    if include_single:
        paths.append(source / "dataset" / "refine_merged_single_questions.json")
    result = []
    for path in paths:
        kind = "single" if "single" in path.name else "multi"
        for case in json.loads(path.read_text()):
            result.append({"kind": kind, "case": case})
    return result


def answer_for(case: dict) -> str:
    answer = case.get("answer")
    if answer is None:
        answer = case.get("metadata", {}).get("golden_answer", "")
    if isinstance(answer, (dict, list)):
        return json.dumps(answer, ensure_ascii=False, sort_keys=True)
    return str(answer)


def generate(
    output: Path,
    source: Path | None = None,
    ids: list[str] | None = None,
    limit: int | None = None,
    include_single: bool = False,
    overwrite: bool = False,
) -> int:
    all_cases = cases(source_root(source), include_single)
    selected = list(range(len(all_cases))) if ids is None else [int(value) for value in ids]
    if limit is not None:
        selected = selected[:limit]
    for index in selected:
        item = all_cases[index]
        case = item["case"]
        task_id = f"{item['kind']}-{index:04d}"
        task = output / f"scieagentgym-{task_id}"
        if task.exists() and not overwrite:
            continue
        if task.exists():
            shutil.rmtree(task)
        (task / "environment/data").mkdir(parents=True)
        (task / "solution").mkdir()
        (task / "tests/data").mkdir(parents=True)
        (task / "environment/Dockerfile").write_text(
            (ROOT / "task-template/environment/Dockerfile").read_text()
        )
        metadata = case.get("metadata", {})
        (task / "instruction.md").write_text(
            (ROOT / "task-template/instruction.md")
            .read_text()
            .replace("{{ subject }}", str(metadata.get("subject", "")))
            .replace("{{ topic }}", str(metadata.get("topic", "")))
            .replace("{{ question }}", str(case.get("question", "")))
            .replace("{{ tools }}", ", ".join(metadata.get("tool_expected", [])))
        )
        (task / "task.toml").write_text(
            (ROOT / "task-template/task.toml").read_text().replace("{{ task_id }}", task_id)
        )
        answer = answer_for(case)
        boxed_answer = f"###Answer###\n$\\boxed{{{answer}}}$"
        (task / "solution/solve.sh").write_text(
            (ROOT / "task-template/solution/solve.sh").read_text().replace(
                "{{ answer }}", boxed_answer.encode("utf-8").hex()
            )
        )
        (task / "solution/solve.sh").chmod(0o755)
        for name in ("test.sh", "verify.py"):
            shutil.copy2(ROOT / f"task-template/tests/{name}", task / f"tests/{name}")
        (task / "tests/test.sh").chmod(0o755)
        (task / "environment/data/case.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2)
        )
        (task / "tests/data/answer.txt").write_text(answer)
    return len(selected)
