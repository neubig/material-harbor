import argparse
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

REVISION = "f8cc3bdcc6357c88b8c3648306522b9c422dc95a"


def generate(source, capsule, output, ids, expected_sha256):
    if hashlib.sha256(capsule.read_bytes()).hexdigest() != expected_sha256:
        raise ValueError("Capsule SHA256 mismatch")
    records = {
        r["question_id"]: r for r in map(json.loads, source.read_text().splitlines())
    }
    for qid in ids:
        rec = records[qid]
        if rec["eval_mode"] not in ("range_verifier", "str_verifier"):
            raise ValueError("No validated key-free semantics for LLM mode")
        task = output / qid
        task.mkdir(parents=True, exist_ok=False)
        for name in ("environment/data", "tests", "solution"):
            (task / name).mkdir(parents=True)
        with zipfile.ZipFile(capsule) as archive:
            prefix = f"CapsuleData-{rec['capsule_uuid']}/"
            selected = [
                i
                for i in archive.infolist()
                if i.filename.startswith(prefix) and not i.is_dir()
            ]
            if not selected:
                raise ValueError("No capsule data found")
            for item in selected:
                relative = Path(item.filename.removeprefix(prefix))
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or relative.suffix.lower() == ".ipynb"
                ):
                    raise ValueError("Unsafe or notebook data entry")
                dest = task / "environment/data" / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(item))
        (task / "instruction.md").write_text(
            rec["question"]
            + "\nUse data in /workspace. Write only <answer>VALUE</answer> to /workspace/answer.txt.\n"
        )
        (task / "task.toml").write_text("""version = "1.0"
[metadata]
category = "biology-diagnostic-only"
[agent]
timeout_sec = 240
[verifier]
timeout_sec = 30
[environment]
build_timeout_sec = 600
cpus = 2
memory_mb = 4096
storage_mb = 10240
""")
        (task / "environment/Dockerfile").write_text(
            "FROM python:3.12-slim\nRUN pip install --no-cache-dir pandas==2.2.3 numpy==2.2.6 scipy==1.15.3\nWORKDIR /workspace\nCOPY data/ /workspace/\n"
        )
        shutil.copyfile(Path(__file__).with_name("verify.py"), task / "tests/verify.py")
        (task / "tests/gold.json").write_text(json.dumps(rec))
        (task / "tests/test.sh").write_text("#!/bin/sh\npython /tests/verify.py\n")
        value = (
            str(ast.literal_eval(rec["ideal"])[0])
            if rec["eval_mode"] == "range_verifier"
            else rec["ideal"]
        )
        (task / "solution/solve.sh").write_text(
            "#!/bin/sh\ncat > /workspace/answer.txt <<'BIXBENCH_GOLD'\n<answer>"
            + value
            + "</answer>\nBIXBENCH_GOLD\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()
    generate(args.source, args.capsule, args.output, args.ids, args.sha256)
