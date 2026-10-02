"""Build Harbor tasks whose answers are only obtainable by reading the pixels.

Why synthetic tasks instead of a published benchmark
---------------------------------------------------
A benchmark score tells you whether an agent solved tasks, not whether images
survived serialization. A text-only model can score above chance on multiple
choice by guessing, and a task whose answer is also inferable from the prompt
cannot distinguish "read the figure" from "ignored the figure".

These tasks are deliberately unguessable:

* the six-digit code is drawn from a random per-run seed and appears nowhere in
  the instruction text;
* the shape census (count of each of three colours) is likewise random.

A model that never receives the pixels cannot answer either question, so a
correct answer is direct evidence that the image arrived and was understood.
The text-only control is the same tasks run with image transport disabled, which
must collapse to (near) zero.

Images are rendered on the host so the task image needs no Pillow.

Usage:
    python harbor_vision/vision_probe/build.py --output <dir> --count 12 --seed <int>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# A generous canvas keeps glyphs large enough that any working vision pipeline
# reads them reliably, so a failure points at transport rather than resolution.
WIDTH, HEIGHT = 640, 320
COLOURS = {"red": (220, 40, 40), "green": (40, 170, 70), "blue": (40, 90, 220)}

INSTRUCTION = """Read the figure at /app/data/figure.png and answer two questions about it.

1. A six-digit code is printed in large black digits near the top.
2. Below it are several coloured shapes.

Write exactly two lines to /app/answer.txt, in this order, with no other text:

code=<the six digits, no spaces>
red=<count of red shapes>,green=<count of green shapes>,blue=<count of blue shapes>

Example of the required format (values shown are not the answers):

code=000000
red=0,green=0,blue=0

You must inspect the image; the values are not stated anywhere in this text.
"""


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Prefer a scalable default; fall back to the bitmap default."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1 has no size argument
        return ImageFont.load_default()


def render(code: str, counts: dict[str, int], seed: int) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    draw.text((40, 30), code, fill=(0, 0, 0), font=_font(96))

    # Shapes sit on a jittered grid so positions vary per task while staying
    # non-overlapping, which keeps the counts unambiguous for the model.
    rng = random.Random(seed ^ 0x5EED)
    slots = [(x, y) for y in range(180, 300, 60) for x in range(60, 600, 60)]
    rng.shuffle(slots)

    shapes = [name for name, n in counts.items() for _ in range(n)]
    rng.shuffle(shapes)

    for name, (x, y) in zip(shapes, slots):
        colour = COLOURS[name]
        radius = 18
        if rng.random() < 0.5:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)
        else:
            draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=colour)

    return image


def make_task(
    root: Path, index: int, seed: int, base_image: str, proxy_host: str | None = None
) -> dict:
    rng = random.Random(seed * 1000 + index)
    code = "".join(str(rng.randrange(10)) for _ in range(6))
    # The 20 grid slots bound how many shapes can be drawn without overlap.
    counts = {name: rng.randrange(0, 5) for name in COLOURS}
    if sum(counts.values()) == 0:
        counts["red"] = 1

    task_id = f"vision-probe-{index:02d}"
    task = root / task_id
    (task / "environment" / "data").mkdir(parents=True, exist_ok=True)
    (task / "tests").mkdir(parents=True, exist_ok=True)

    figure = task / "environment" / "data" / "figure.png"
    render(code, counts, seed * 1000 + index).save(figure, format="PNG")

    (task / "instruction.md").write_text(INSTRUCTION)
    # Inheriting the prebaked image keeps Harbor's "SDK already installed" check
    # true, so its pip step is skipped and the pinned PR SDK is what runs.
    (task / "environment" / "Dockerfile").write_text(
        f"FROM {base_image}\nWORKDIR /app\nCOPY data /app/data\n"
    )
    if proxy_host:
        # Task-authored compose networking is respected by Harbor. host-gateway
        # resolves to the host as seen from the container, which is where the
        # recording proxy listens; no fixed bridge subnet has to be assumed.
        (task / "environment" / "docker-compose.yaml").write_text(
            "services:\n"
            "  main:\n"
            "    extra_hosts:\n"
            f'      - "{proxy_host}:host-gateway"\n'
        )
    (task / "tests" / "gold.json").write_text(
        json.dumps({"code": code, "counts": counts}, indent=2) + "\n"
    )
    (task / "tests" / "verify.py").write_text(VERIFY)
    # Shared verifier mode (Harbor 0.22 default): the tests directory is mounted at
    # /tests inside the agent's own container, so no separate verifier image is
    # needed and the reward lands in /logs/verifier/reward.txt.
    (task / "tests" / "test.sh").write_text(
        "#!/bin/sh\nset -eu\npython /tests/verify.py\n"
    )
    (task / "task.toml").write_text(
        f"""version = "1.0"

[metadata]
category = "vision-transport-probe"
generator_seed = "{seed}"
figure_sha256 = "{hashlib.sha256(figure.read_bytes()).hexdigest()}"

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 900.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
"""
    )
    return {"task_id": task_id, "code": code, "counts": counts}


# Strict parser: the reward is 1 only when both the code and the full census match,
# so a partially-guessed answer cannot inflate the score.
VERIFY = '''"""Verifier for the vision transport probe; labels are staged only by Harbor."""
import argparse
import json
import re
from pathlib import Path

LINE = re.compile(r"^code=(\\d{6})$")
CENSUS = re.compile(r"^red=(\\d+),green=(\\d+),blue=(\\d+)$")


def parse(text):
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if len(lines) != 2:
        return None
    code_match = LINE.match(lines[0])
    census_match = CENSUS.match(lines[1])
    if not code_match or not census_match:
        return None
    return {
        "code": code_match.group(1),
        "counts": {
            "red": int(census_match.group(1)),
            "green": int(census_match.group(2)),
            "blue": int(census_match.group(3)),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--answer", type=Path, default=Path("/app/answer.txt"))
    p.add_argument("--gold", type=Path, default=Path("/tests/gold.json"))
    p.add_argument("--reward", type=Path, default=Path("/logs/verifier/reward.txt"))
    args = p.parse_args()

    gold = json.loads(args.gold.read_text())
    reward = 0.0
    try:
        if args.answer.is_file() and not args.answer.is_symlink():
            parsed = parse(args.answer.read_text())
            if parsed is not None:
                reward = float(
                    parsed["code"] == gold["code"]
                    and parsed["counts"] == gold["counts"]
                )
    except (OSError, UnicodeError, ValueError):
        reward = 0.0

    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(str(reward) + "\\n")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-image", default="python:3.12-slim")
    parser.add_argument(
        "--proxy-host",
        default=None,
        help="Hostname mapped to host-gateway in task compose (enables the "
        "recording proxy to be reached from inside the container).",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    tasks = [
        make_task(args.output, i, args.seed, args.base_image, args.proxy_host)
        for i in range(args.count)
    ]
    (args.output / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "count": args.count, "tasks": tasks}, indent=2)
        + "\n"
    )
    print(f"built {len(tasks)} tasks in {args.output} (seed {args.seed})")


if __name__ == "__main__":
    main()
