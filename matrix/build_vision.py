"""Build image-bearing MATRIX tasks from the 250-row vision split.

The text-only builder selects from the 220 ``type == "text"`` rows. This one
selects from the 250 ``type == "vision"`` rows and ships each row's figure into
the agent container, so the task cannot be answered from the prompt alone.

Selection is deterministic on a published seed and made before any outcome is
observed. Grading reuses the frozen MATRIX judge (see ``verify.py``): the
five-level GPT-5.1 rubric, reconstructed official-style rather than
official-exact, because no runnable official grader is published.

Usage:
    python matrix/build_vision.py --count 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REVISION = "80b39472f8a22c4e47ec40b6a9b78c7077af01eb"
SEED = "matrix-vision-pilot-20260814-v1"
IMAGE = "material-harbor/openhands-sdk-vision:dad4aa5"
VERIFIER_IMAGE = (
    "python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
)
IMAGES_DIR = ROOT / "source" / "test" / "images"

INSTRUCTION_SUFFIX = (
    "\n\nInspect the image at the path above; the text alone is not the complete task."
    "\n\nWrite your complete explanatory answer as plain UTF-8 text to "
    "/logs/artifacts/answer.txt (at most 65536 bytes).\n"
)


def select(rows, count: int) -> list[dict]:
    vision = [r for r in rows if r["type"] == "vision"]
    return sorted(
        vision,
        key=lambda r: (hashlib.sha256((SEED + ":" + r["qid"]).encode()).hexdigest(), r["qid"]),
    )[:count]


def build(output: Path, count: int, source: Path) -> dict:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if sum(r["type"] == "vision" for r in rows) != 250:
        raise ValueError("Unexpected MATRIX vision row count")

    selected = select(rows, count)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    built = []
    for row in selected:
        source_image = IMAGES_DIR / Path(row["image_path"]).name
        if not source_image.exists():
            raise FileNotFoundError(source_image)

        task = output / ("matrix-vision-" + row["qid"])
        environment = task / "environment"
        tests = task / "tests"
        (environment / "data").mkdir(parents=True)
        tests.mkdir(parents=True)

        shutil.copyfile(source_image, environment / "data" / "image.png")
        question = row["question"].replace("{image}", "").strip()
        (task / "instruction.md").write_text(
            f"The image for this question is /app/data/image.png.\n\n"
            f"{question}{INSTRUCTION_SUFFIX}"
        )
        (tests / "gold.json").write_text(json.dumps(row, indent=2))
        shutil.copyfile(ROOT / "verify.py", tests / "verify.py")
        (tests / "test.sh").write_text("#!/bin/sh\nset -eu\npython -I /tests/verify.py\n")
        (tests / "test.sh").chmod(0o755)
        (task / "task.toml").write_text(f'''schema_version = "1.0"
[metadata]
dataset = "radical-ai/MATRIX"
source_revision = "{REVISION}"
protocol = "matrix-five-level-official-style-v2"
modality = "vision"
[agent]
timeout_sec = 900.0
[verifier]
timeout_sec = 300.0
environment_mode = "separate"
[verifier.environment]
docker_image = "{VERIFIER_IMAGE}"
network_mode = "public"
cpus = 1
memory_mb = 512
[environment]
docker_image = "{IMAGE}"
workdir = "/tmp"
build_timeout_sec = 600.0
cpus = 2
memory_mb = 4096
storage_mb = 2048
network_mode = "public"
''')
        built.append({"task_id": task.name, "qid": row["qid"], "kind": row["kind"],
                      "image": "environment/data/image.png"})

    manifest = {
        "revision": REVISION,
        "seed": SEED,
        "modality": "vision",
        "selection": f"lowest sha256(seed + colon + qid) among all 250 vision rows; count {count}",
        "tasks": built,
    }
    (output.parent / "manifest-vision.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--source", type=Path, default=ROOT / "source" / "test.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "tasks-vision")
    args = parser.parse_args()
    result = build(args.output, args.count, args.source)
    print(json.dumps(result, indent=2)[:1500])
