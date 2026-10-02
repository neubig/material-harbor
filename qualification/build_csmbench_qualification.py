"""Freeze and build the CSMBench qualification100 task cohort."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from csmbench import main as source

ROOT = Path(__file__).resolve().parents[1]
SEED = "qualification-v1"


def freeze(path, count=100):
    rows = source.load_rows()
    selected = sorted(
        rows,
        key=lambda r: hashlib.sha256(f"{SEED}:{r['file_name']}".encode()).hexdigest(),
    )[:count]
    obj = {
        "version": 1,
        "source_revision": source.REVISION,
        "selection": f"lowest SHA256({SEED}:file_name), frozen before execution",
        "count": count,
        "indices": [r["index"] for r in selected],
        "rows": [
            {k: r[k] for k in ("index", "file_name", "paper_folder_name", "source", "scale")}
            for r in selected
        ],
    }
    path.write_text(json.dumps(obj, indent=2) + "\n")


def build(manifest, output):
    plan = json.loads(manifest.read_text())
    rows = {r["index"]: r for r in source.load_rows()}
    selected = [rows[i] for i in plan["indices"]]
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    audit = []
    for row in selected:
        image = ROOT / "csmbench/source/images" / row["file_name"]
        source.download(source.BASE + row["file_name"], image)
        info = source.validate_image(image)
        task = output / f"csmbench-mcqa-{row['index']:04d}"
        (task / "environment/data").mkdir(parents=True)
        (task / "tests").mkdir()
        (task / "solution").mkdir()
        shutil.copyfile(image, task / "environment/data/image.jpg")
        prompt = source.build_prompt(json.loads(row["options"]))
        (task / "environment/data/question.json").write_text(
            json.dumps({"prompt": prompt, "options": row["options"], "image": "/app/data/image.jpg"}, indent=2) + "\n"
        )
        (task / "instruction.md").write_text(
            prompt + "\n\nThe image is /app/data/image.jpg. Inspect the actual image; text alone is not the complete task. "
            "Write your final answer to /app/answer.txt as exactly one uppercase ASCII letter (A, B, C, or D), optionally followed by one LF newline. No explanation.\n"
        )
        (task / "environment/Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\nCOPY data /app/data\n")
        (task / "task.toml").write_text(
            f'version = "1.0"\n[metadata]\ncategory = "materials-science"\ndataset = "CSMBench MCQA"\nsource_revision = "{source.REVISION}"\n'
            "[verifier]\ntimeout_sec = 60.0\n[agent]\ntimeout_sec = 900.0\n[environment]\nbuild_timeout_sec = 600.0\ncpus = 1\nmemory_mb = 2048\nstorage_mb = 4096\n"
        )
        shutil.copyfile(ROOT / "csmbench/verify.py", task / "tests/verify.py")
        (task / "tests/label.json").write_text(json.dumps({"correct_answer": row["correct_answer"]}) + "\n")
        (task / "tests/test.sh").write_text("#!/bin/sh\nset -eu\npython /tests/verify.py\n")
        (task / "solution/solve.sh").write_text(
            f"#!/bin/sh\nset -eu\nprintf '{row['correct_answer']}\\n' > /app/answer.txt\n"
        )
        for executable in (task / "tests/test.sh", task / "solution/solve.sh"):
            executable.chmod(0o755)
        audit.append({"task_id": task.name, "index": row["index"], "paper_folder_name": row["paper_folder_name"], **info})
    (output.parent / "qualification100-audit.json").write_text(json.dumps({"tasks": audit}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    freeze(args.manifest, args.count) if args.freeze else build(args.manifest, args.output)


if __name__ == "__main__":
    main()
