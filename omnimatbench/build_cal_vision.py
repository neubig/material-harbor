"""Build a frozen image-bearing OmniMatBench CAL cohort."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from omnimatbench.main import REVISION, ROOT, SOURCE, validate_sources, write_json

SEED = "omnimatbench-cal-vision-qualification-v1"


def load_rows():
    validate_sources()
    image_hashes = json.loads((ROOT / "cal-image-sha256.json").read_text())
    if image_hashes["revision"] != REVISION or image_hashes["count"] != 142:
        raise ValueError("Image manifest identity mismatch")
    rows = []
    for source in sorted((SOURCE / "cal").rglob("*.jsonl")):
        category = source.relative_to(SOURCE).parts[1]
        for line_no, line in enumerate(source.read_text().splitlines(), 1):
            row = json.loads(line)
            if not row.get("image_url"):
                continue
            candidates = [SOURCE / name for name in image_hashes["sha256"]
                          if (SOURCE / name).parent == source.parent / "images"
                          and (SOURCE / name).stem == row["image_url"]]
            if len(candidates) != 1:
                raise ValueError(f"Image resolution failed: {row['image_url']}")
            image = candidates[0]
            rel_image = str(image.relative_to(SOURCE))
            if hashlib.sha256(image.read_bytes()).hexdigest() != image_hashes["sha256"][rel_image]:
                raise ValueError(f"Image checksum mismatch: {rel_image}")
            key = f"cal/{category}/{row['id']}"
            rows.append({"key": key, "source": source, "line": line_no, "image": image, "row": row})
    if len(rows) != 142 or len({r["key"] for r in rows}) != 142 or len({r["row"]["question"] for r in rows}) != 142:
        raise ValueError("Unexpected image CAL population")
    return rows


def freeze(rows, manifest, count=100):
    selected = sorted(rows, key=lambda r: hashlib.sha256(f"{SEED}:{r['key']}".encode()).hexdigest())[:count]
    value = {
        "version": 1,
        "revision": REVISION,
        "population": 142,
        "count": count,
        "selection": f"lowest SHA256({SEED}:namespaced_id), frozen before outcomes",
        "metric": "Pinned upstream score_exact, binary conjunctive exact comparator; missing/invalid answers are zero",
        "image_policy": "Every task includes and must expose its released source image; no no-image controls",
        "selected": [
            {
                "key": item["key"],
                "source": str(item["source"].relative_to(SOURCE)),
                "line": item["line"],
                "question_sha256": hashlib.sha256(item["row"]["question"].encode()).hexdigest(),
                "image": str(item["image"].relative_to(SOURCE)),
                "image_sha256": hashlib.sha256(item["image"].read_bytes()).hexdigest(),
            }
            for item in selected
        ],
    }
    write_json(manifest, value)


def build(rows, manifest, output):
    frozen = json.loads(manifest.read_text())
    lookup = {r["key"]: r for r in rows}
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    for item in frozen["selected"]:
        record = lookup[item["key"]]
        row = record["row"]
        task = output / ("omnimatbench-" + item["key"].replace("/", "-"))
        for sub in ("environment/data", "tests/native", "solution"):
            (task / sub).mkdir(parents=True)
        write_json(task / "environment/data/case.json", {k: row[k] for k in ("question", "final_answer_format")})
        figure_name = "figure" + record["image"].suffix.lower()
        shutil.copy2(record["image"], task / "environment/data" / figure_name)
        (task / "environment/Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\nCOPY data /app/data\n")
        (task / "instruction.md").write_text(
            "# OmniMatBench CAL — image-bearing subset\n\n"
            f"Solve the official question in `/app/data/case.json` using `/app/data/{figure_name}`; you must inspect the image. "
            "Follow the requested units and rounding. Write `/app/answer.json` as a valid JSON list following "
            "`final_answer_format` grouping and slot order. Fill each blank with a string or finite number. Do not put "
            "reasoning, Markdown fences, wrappers, or tags in that file. The verifier uses the pinned upstream binary "
            "`score_exact` comparator; all answer slots must match. Missing, malformed, or incomplete answers receive zero.\n"
        )
        (task / "task.toml").write_text(f'''schema_version = "1.0"
[task]
name = "omnimatbench/{item['key'].replace('/', '-')}"
[metadata]
dataset = "OmniMatBench CAL image-bearing subset"
source_revision = "{REVISION}"
source_id = "{item['key']}"
source = "https://github.com/wanhaoliu/OmniMatBench"
grading = "native-score_exact-conjunctive; strict JSON-list transport"
[agent]
timeout_sec = 1800.0
[verifier]
timeout_sec = 60.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 2048
storage_mb = 4096
''')
        write_json(task / "tests/gold.json", row["final_answer_list"])
        shutil.copy2(ROOT / "verify.py", task / "tests/verify.py")
        shutil.copy2(SOURCE / "LICENSE", task / "tests/native/LICENSE")
        for name in ("eval_cal_results.py", "omnimat_paths.py"):
            shutil.copy2(SOURCE / "scripts/cal" / name, task / "tests/native" / name)
        (task / "tests/test.sh").write_text("#!/bin/sh\nset -eu\npython /tests/verify.py\n")
        (task / "tests/test.sh").chmod(0o755)
        (task / "solution/solve.sh").write_text(
            "#!/bin/sh\nset -eu\ncat > /app/answer.json <<'OMNIMAT_GOLD'\n"
            + json.dumps(row["final_answer_list"], ensure_ascii=False) + "\nOMNIMAT_GOLD\n"
        )
        (task / "solution/solve.sh").chmod(0o755)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "qualification/omnimat-cal-vision-manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "tasks/cal-vision-qualification100")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    rows = load_rows()
    if args.freeze:
        freeze(rows, args.manifest)
    else:
        build(rows, args.manifest, args.output)


if __name__ == "__main__":
    main()
