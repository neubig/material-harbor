from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import random
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE_MANIFEST = ROOT / "source-manifest.json"
SAMPLE_MANIFEST = ROOT / "random100-manifest.json"
PINNED_SOURCE_MANIFEST_SHA256 = "46b5aec09b028e214707b39eb145b9fa17b695ffde8fe8ef887c255489fd31dd"
PINNED_SAMPLE_MANIFEST_SHA256 = "e121123b316dfcac4bf85bea0f2bc66c8afc0afadf2b8cc73350c263dd1a9d4d"
SAMPLE_SEED = 2026100901
TASK_ID_RE = re.compile(r"^(?:test|validation)-[0-9]{6}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_ASSET_BYTES = 8 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
MAX_IMAGE_PIXELS = 16 * 1024 * 1024


def _read_regular(path: Path, max_bytes: int) -> bytes:
    if path.is_symlink():
        raise ValueError(f"refusing symlink: {path}")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ValueError(f"cannot safely open {path}: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"not a regular file: {path}")
        if metadata.st_size > max_bytes:
            raise ValueError(f"file exceeds {max_bytes} bytes: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(max_bytes + 1)
    finally:
        os.close(descriptor)
    if len(data) > max_bytes:
        raise ValueError(f"file exceeds {max_bytes} bytes: {path}")
    return data


def _json_object(path: Path, max_bytes: int) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular(path, max_bytes).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _validate_asset(image: dict[str, Any], assets_dir: Path | None = None) -> bytes:
    expected_keys = {
        "asset",
        "sha256",
        "pixel_sha256",
        "size",
        "width",
        "height",
        "format",
        "historical_source_path",
    }
    if set(image) != expected_keys:
        raise ValueError("image metadata has unexpected fields")
    digest = image["sha256"]
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError("invalid image SHA-256")
    expected_asset = f"assets/images/{digest}.png"
    if image["asset"] != expected_asset:
        raise ValueError("image asset path is not content-addressed")
    assets_root = (assets_dir or (ROOT / "assets/images")).resolve(strict=True)
    path = assets_root / f"{digest}.png"
    if path.parent.resolve(strict=True) != assets_root:
        raise ValueError("image asset escapes the asset directory")
    size = image["size"]
    if type(size) is not int or not 1 <= size <= MAX_ASSET_BYTES:
        raise ValueError("invalid image size")
    data = _read_regular(path, MAX_ASSET_BYTES)
    if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"image bytes do not match manifest: {path.name}")
    width, height = image["width"], image["height"]
    if (
        type(width) is not int
        or type(height) is not int
        or not 1 <= width <= MAX_IMAGE_DIMENSION
        or not 1 <= height <= MAX_IMAGE_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise ValueError("invalid image dimensions")
    if image["format"] != "PNG":
        raise ValueError("unsupported image format")
    pixel_digest = image["pixel_sha256"]
    if not isinstance(pixel_digest, str) or not SHA256_RE.fullmatch(pixel_digest):
        raise ValueError("invalid decoded-pixel SHA-256")
    try:
        with Image.open(io.BytesIO(data)) as decoded:
            decoded.load()
            if decoded.format != "PNG" or decoded.size != (width, height):
                raise ValueError("decoded image metadata does not match manifest")
            actual_pixel_digest = hashlib.sha256(decoded.convert("RGB").tobytes()).hexdigest()
            if actual_pixel_digest != pixel_digest:
                raise ValueError("decoded pixels do not match manifest")
    except OSError as exc:
        raise ValueError(f"invalid PNG asset: {path.name}") from exc
    return data


def load_source_manifest(assets_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source_bytes = _read_regular(SOURCE_MANIFEST, 2 * 1024 * 1024)
    if hashlib.sha256(source_bytes).hexdigest() != PINNED_SOURCE_MANIFEST_SHA256:
        raise ValueError("source manifest does not match the pinned recovered universe")
    manifest = _json_object(SOURCE_MANIFEST, 2 * 1024 * 1024)
    if manifest.get("schema_version") != 1 or manifest.get("population_size") != 250:
        raise ValueError("unexpected source manifest version or population")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 250:
        raise ValueError("source manifest must contain exactly 250 records")
    by_id: dict[str, dict[str, Any]] = {}
    candidate_ids: set[str] = set()
    unique_assets: set[str] = set()
    expected_record_keys = {
        "task_id",
        "original_split",
        "source_index",
        "candidate_id",
        "paper_id",
        "difficulty",
        "question",
        "answer",
        "image",
        "prior_candidate_cohort",
    }
    for record in records:
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise ValueError("source record does not match the frozen schema")
        task_id = record.get("task_id")
        if not isinstance(task_id, str) or not TASK_ID_RE.fullmatch(task_id):
            raise ValueError("invalid source task ID")
        if task_id in by_id:
            raise ValueError(f"duplicate source task ID: {task_id}")
        split, source_index_text = task_id.rsplit("-", 1)
        if record["original_split"] != split or record["source_index"] != int(source_index_text):
            raise ValueError(f"source location does not match task ID: {task_id}")
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
            raise ValueError("missing or duplicate candidate ID")
        if not isinstance(record["paper_id"], str) or not record["paper_id"]:
            raise ValueError(f"invalid paper ID for {task_id}")
        if record["difficulty"] not in {"hard", "very_hard"}:
            raise ValueError(f"invalid difficulty for {task_id}")
        if record["prior_candidate_cohort"] not in {None, "evaluation", "reserve"}:
            raise ValueError(f"invalid prior candidate cohort for {task_id}")
        question, answer = record.get("question"), record.get("answer")
        if not isinstance(question, str) or not question.strip() or len(question.encode()) > 4096:
            raise ValueError(f"invalid question for {task_id}")
        if not isinstance(answer, str) or not answer.strip() or len(answer.encode()) > 8192:
            raise ValueError(f"invalid answer for {task_id}")
        image = record.get("image")
        if not isinstance(image, dict):
            raise ValueError(f"missing image metadata for {task_id}")
        _validate_asset(image, assets_dir)
        unique_assets.add(image["sha256"])
        candidate_ids.add(candidate_id)
        by_id[task_id] = record
    ordered_ids = sorted(by_id)
    ordered_hash = hashlib.sha256(("\n".join(ordered_ids) + "\n").encode()).hexdigest()
    if manifest.get("ordered_task_ids_sha256") != ordered_hash:
        raise ValueError("source universe ID hash mismatch")
    if manifest.get("unique_encoded_images") != len(unique_assets):
        raise ValueError("source manifest unique-image count mismatch")
    return manifest, by_id


def load_random100(eligible_ids: list[str]) -> tuple[dict[str, Any], list[str]]:
    sample_bytes = _read_regular(SAMPLE_MANIFEST, 64 * 1024)
    if hashlib.sha256(sample_bytes).hexdigest() != PINNED_SAMPLE_MANIFEST_SHA256:
        raise ValueError("random100 manifest does not match the frozen sample")
    sample = _json_object(SAMPLE_MANIFEST, 64 * 1024)
    source_digest = hashlib.sha256(_read_regular(SOURCE_MANIFEST, 2 * 1024 * 1024)).hexdigest()
    if sample.get("source_manifest_sha256") != source_digest:
        raise ValueError("random100 manifest is not pinned to this source manifest")
    if sample.get("population_size") != len(eligible_ids) or sample.get("sample_size") != 100:
        raise ValueError("random100 population or sample size mismatch")
    seed = sample.get("seed")
    selected = sample.get("sample")
    if seed != SAMPLE_SEED or not isinstance(selected, list):
        raise ValueError("invalid random100 seed or sample")
    expected = random.Random(seed).sample(sorted(eligible_ids), 100)
    if selected != expected or len(set(selected)) != 100:
        raise ValueError("random100 does not reproduce without replacement")
    return sample, selected


def _render(template: str, replacements: dict[str, str]) -> str:
    for marker, value in replacements.items():
        if template.count(marker) != 1:
            raise ValueError(f"template marker must occur exactly once: {marker}")
        template = template.replace(marker, value)
    return template


def _write_task(output: Path, record: dict[str, Any], image: bytes, overwrite: bool) -> bool:
    task_id = record["task_id"]
    task = output / f"materials-figure-qa-{task_id}"
    if task.exists() or task.is_symlink():
        if not overwrite:
            return False
        if task.is_symlink() or not task.is_dir():
            raise ValueError(f"refusing to overwrite non-directory task path: {task}")
        shutil.rmtree(task)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{task.name}-", dir=output) as temporary:
        staging = Path(temporary)
        (staging / "environment/data").mkdir(parents=True)
        (staging / "solution").mkdir()
        (staging / "tests/data").mkdir(parents=True)
        shutil.copy2(ROOT / "task-template/environment/Dockerfile", staging / "environment/Dockerfile")
        (staging / "environment/data/image.png").write_bytes(image)
        instruction = _render(
            (ROOT / "task-template/instruction.md").read_text(),
            {"{{ question }}": record["question"]},
        )
        (staging / "instruction.md").write_text(instruction)
        task_toml = _render(
            (ROOT / "task-template/task.toml").read_text(),
            {"{{ task_id }}": task_id, "{{ difficulty }}": record["difficulty"]},
        )
        (staging / "task.toml").write_text(task_toml)
        encoded_answer = base64.b64encode(record["answer"].encode()).decode()
        solve = _render(
            (ROOT / "task-template/solution/solve.sh").read_text(),
            {"{{ answer }}": encoded_answer},
        )
        (staging / "solution/solve.sh").write_text(solve)
        for name in ("test.sh", "verify.py"):
            shutil.copy2(ROOT / "task-template/tests" / name, staging / "tests" / name)
        (staging / "tests/data/image.png").write_bytes(image)
        private = {
            "schema_version": 1,
            "task_id": task_id,
            "candidate_id": record["candidate_id"],
            "question": record["question"],
            "reference_answer": record["answer"],
            "image_sha256": record["image"]["sha256"],
            "image_size": record["image"]["size"],
        }
        (staging / "tests/data/info.json").write_text(
            json.dumps(private, indent=2, ensure_ascii=False) + "\n"
        )
        (staging / "solution/solve.sh").chmod(0o755)
        (staging / "tests/test.sh").chmod(0o755)
        (staging / "environment/data/image.png").chmod(0o444)
        (staging / "tests/data/image.png").chmod(0o444)
        os.replace(staging, task)
    return True


def generate(
    output: Path,
    task_ids: list[str] | None = None,
    *,
    assets_dir: Path | None = None,
    all_tasks: bool = False,
    overwrite: bool = False,
) -> int:
    source, by_id = load_source_manifest(assets_dir)
    sample, random100 = load_random100(list(by_id))
    if output.is_symlink() or (output.exists() and not output.is_dir()):
        raise ValueError(f"output must be a real directory: {output}")
    if task_ids is not None and all_tasks:
        raise ValueError("task_ids and all_tasks are mutually exclusive")
    selected = sorted(by_id) if all_tasks else random100 if task_ids is None else task_ids
    if len(selected) != len(set(selected)):
        raise ValueError("duplicate requested task IDs")
    unknown = [task_id for task_id in selected if task_id not in by_id]
    if unknown:
        raise ValueError(f"unknown task IDs: {unknown}")
    written = 0
    for task_id in selected:
        record = by_id[task_id]
        image = _validate_asset(record["image"], assets_dir)
        written += int(_write_task(output, record, image, overwrite))
    run_manifest = {
        "source_manifest_sha256": hashlib.sha256(SOURCE_MANIFEST.read_bytes()).hexdigest(),
        "random100_manifest_sha256": hashlib.sha256(SAMPLE_MANIFEST.read_bytes()).hexdigest(),
        "eligible_population_size": source["population_size"],
        "selection": "all" if all_tasks else "explicit" if task_ids is not None else sample["name"],
        "task_ids": selected,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build recovered-source Materials Figure QA Harbor tasks")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/materials-figure-qa-random100"))
    parser.add_argument("--assets-dir", type=Path, help="directory containing recovered content-addressed PNG files")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="generate the full frozen 250-task universe")
    selection.add_argument("--task-ids", nargs="+", help="generate explicit frozen-universe task IDs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(generate(args.output_dir, args.task_ids, assets_dir=args.assets_dir, all_tasks=args.all, overwrite=args.overwrite))


if __name__ == "__main__":
    main()
