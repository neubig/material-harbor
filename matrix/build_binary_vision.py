"""Build a prospective image-bearing MATRIX full-credit binary cohort."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from build_vision import IMAGE, IMAGES_DIR, INSTRUCTION_SUFFIX, REVISION

ROOT = Path(__file__).resolve().parent
SEED = "matrix-full-credit-binary-qualification-v1"


def build(output, manifest_path, count=100):
    rows = [json.loads(line) for line in (ROOT / "source/test.jsonl").read_text().splitlines() if line.strip()]
    vision = [row for row in rows if row["type"] == "vision"]
    pilot = json.loads((ROOT / "manifest-vision.json").read_text())
    excluded = {row["qid"] for row in pilot["tasks"]}
    selected = sorted((row for row in vision if row["qid"] not in excluded),
                      key=lambda row: hashlib.sha256(f"{SEED}:{row['qid']}".encode()).hexdigest())[:count]
    if len(selected) != count or len({row["qid"] for row in selected}) != count:
        raise ValueError("Invalid frozen selection")
    manifest = {
        "version": 1,
        "revision": REVISION,
        "seed": SEED,
        "population": 250,
        "excluded_prior_pilot_qids": sorted(excluded),
        "train_validation_exact_prompt_overlap": 0,
        "selection": f"lowest SHA256({SEED}:qid) among 250 vision rows excluding prior pilot10; frozen before outcomes",
        "metric": "Derived binary full-credit endpoint: judge score exactly 1 succeeds, 0/0.25/0.5/0.75 fail; paired five-level reward retained",
        "tasks": [{"task_id": "matrix-binary-vision-" + row["qid"], "qid": row["qid"],
                   "kind": row["kind"], "question_sha256": hashlib.sha256(row["question"].encode()).hexdigest(),
                   "image_sha256": hashlib.sha256((IMAGES_DIR / Path(row["image_path"]).name).read_bytes()).hexdigest()}
                  for row in selected],
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to alter frozen manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if output.exists():
        existing = {path.name for path in output.iterdir() if (path / "task.toml").exists()}
        if existing == {row["task_id"] for row in manifest["tasks"]}:
            return manifest
        raise ValueError("Existing task cohort differs from manifest")
    output.mkdir(parents=True)
    by_id = {row["qid"]: row for row in selected}
    for item in manifest["tasks"]:
        row = by_id[item["qid"]]
        task = output / item["task_id"]
        (task / "environment/data").mkdir(parents=True)
        (task / "tests").mkdir()
        source_image = IMAGES_DIR / Path(row["image_path"]).name
        shutil.copyfile(source_image, task / "environment/data/image.png")
        (task / "environment/Dockerfile").write_text(f"FROM {IMAGE}\nWORKDIR /app\nCOPY data /app/data\n")
        question = row["question"].replace("{image}", "").strip()
        (task / "instruction.md").write_text(
            "The image for this question is /app/data/image.png.\n\n" + question + INSTRUCTION_SUFFIX)
        (task / "tests/gold.json").write_text(json.dumps(row, indent=2))
        shutil.copyfile(ROOT / "verify.py", task / "tests/verify.py")
        shutil.copyfile(ROOT / "verify_binary.py", task / "tests/verify_binary.py")
        (task / "tests/test.sh").write_text("#!/bin/sh\nset -eu\npython -I /tests/verify_binary.py\n")
        (task / "tests/test.sh").chmod(0o755)
        (task / "task.toml").write_text(f'''schema_version = "1.0"
[metadata]
dataset = "radical-ai/MATRIX"
source_revision = "{REVISION}"
protocol = "matrix-full-credit-binary-v1"
modality = "vision"
[agent]
timeout_sec = 900.0
[verifier]
timeout_sec = 300.0
[environment]
workdir = "/tmp"
build_timeout_sec = 600.0
cpus = 2
memory_mb = 4096
storage_mb = 2048
network_mode = "public"
''')
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "tasks-binary-vision100")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "qualification/matrix-binary-vision-manifest.json")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.manifest, args.count), indent=2)[:2000])
