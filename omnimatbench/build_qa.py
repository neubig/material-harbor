"""Build image-bearing OmniMatBench QA tasks from the public QA rubric files.

The CAL path in ``main.py`` deliberately selects rows with no ``image_url``.
This builder does the opposite: it selects QA items whose ``image_url`` is set,
fetches the corresponding figure, and bakes it into the agent image, so the
answer cannot be produced from the prompt text alone.

Grading reuses the published ``key_points`` and ``scoring_weights`` as a
GPT-5.1 judge rubric. The release publishes the rubric data and a CAL scorer
but no runnable QA scorer here, so the judge is a reconstructed adapter and is
recorded as a proxy metric rather than official accuracy.

Usage:
    python -m omnimatbench.build_qa --count 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVISION = "f933f03c733bb378ce5dd85b96452d9d5683c17e"
SEED = "omnimatbench-qa-vision-20260814-v1"
IMAGE = "material-harbor/openhands-sdk-vision:dad4aa5"
VERIFIER_IMAGE = (
    "python:3.12-slim@sha256:78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
)
RAW = f"https://raw.githubusercontent.com/wanhaoliu/OmniMatBench/{REVISION}/"
CACHE = ROOT / "source" / "qa"

CATEGORIES = [f"{i:02d}" for i in range(1, 20)]

INSTRUCTION_SUFFIX = (
    "\n\nInspect the image at the path above; the text alone is not the complete task."
    "\n\nWrite your complete explanatory answer as plain UTF-8 text to "
    "/logs/artifacts/answer.txt (at most 65536 bytes).\n"
)


def fetch(url: str, target: Path) -> bytes:
    if target.is_file():
        return target.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        content = response.read()
    target.write_bytes(content)
    return content


def load_items() -> list[dict]:
    items = []
    for category in CATEGORIES:
        listing = json.loads(
            fetch(
                f"https://api.github.com/repos/wanhaoliu/OmniMatBench/contents/qa/{category}?ref={REVISION}",
                CACHE / category / "_listing.json",
            )
        )
        rubric = next(
            (entry for entry in listing if entry["name"].endswith("_rubric.json")), None
        )
        if rubric is None:
            continue
        rows = json.loads(fetch(rubric["download_url"], CACHE / category / "rubric.json"))
        for row in rows:
            if row.get("image_url"):
                items.append({"category": category, **row})
    return items


def select(items: list[dict], count: int) -> list[dict]:
    return sorted(
        items,
        key=lambda r: (
            hashlib.sha256(f"{SEED}:{r['category']}/{r['id']}".encode()).hexdigest(),
            r["category"],
            r["id"],
        ),
    )[:count]


def build(output: Path, count: int) -> dict:
    items = load_items()
    selected = select(items, count)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    built = []
    for row in selected:
        name = f"{row['category']}_QA_{row['id']}"
        image_name = f"{name}_img.png"
        url = f"{RAW}qa/{row['category']}/images/{image_name}"
        raw = fetch(url, CACHE / row["category"] / "images" / image_name)

        task = output / f"omnimatbench-qa-{row['category']}-{row['id']}"
        environment = task / "environment"
        tests = task / "tests"
        (environment / "data").mkdir(parents=True)
        tests.mkdir(parents=True)

        (environment / "data" / "image.png").write_bytes(raw)
        (environment / "Dockerfile").write_text(
            f"FROM {IMAGE}\nWORKDIR /app\nCOPY data /app/data\n"
        )
        (task / "instruction.md").write_text(
            f"The image for this question is /app/data/image.png.\n\n"
            f"{row['question'].strip()}{INSTRUCTION_SUFFIX}"
        )
        (tests / "gold.json").write_text(
            json.dumps(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "key_points": row.get("key_points"),
                    "scoring_weights": row.get("scoring_weights"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        shutil.copyfile(ROOT / "verify_qa.py", tests / "verify.py")
        (tests / "test.sh").write_text("#!/bin/sh\nset -eu\npython -I /tests/verify.py\n")
        (tests / "test.sh").chmod(0o755)
        (task / "task.toml").write_text(f'''schema_version = "1.0"
[metadata]
dataset = "wanhaoliu/OmniMatBench"
source_revision = "{REVISION}"
protocol = "omnimatbench-qa-keypoints-v1"
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
        built.append(
            {
                "task_id": task.name,
                "category": row["category"],
                "id": row["id"],
                "image_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )

    manifest = {
        "revision": REVISION,
        "seed": SEED,
        "modality": "vision",
        "subset": "QA items with nonempty image_url",
        "eligible": len(items),
        "selection": f"lowest sha256(seed + colon + category/id) among {len(items)} eligible QA items; count {count}",
        "tasks": built,
    }
    (ROOT / "manifest-qa-vision.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "tasks-qa-vision")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.count), indent=2)[:2000])
