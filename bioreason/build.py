"""Build a frozen CPU Harbor cohort for BioReason-Pro GO annotation recovery."""
import argparse
import ast
import hashlib
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from verify import load_ontology, normalize_terms

ROOT = Path(__file__).resolve().parent
SEED = "bioreason-go-complete-qualification-v1"
DATASET_REVISION = "a18315a77d27dd4fe5a28b9b757695cff4ba4e3f"
SOURCE_REVISION = "a93ac2a96387103c100c4d6a6674d6fe1dfaf9e8"
DATASET_SHA256 = "f8acd81f3b08400783a34f4189428e4f41fa0c648bc8e4fa58b10ee58bb66828"
ONTOLOGY_SHA256 = "0c0062efbb1b199a6a54e62c0c48ff0038d39173aecc7edfab66a20090d5ec2d"
EXCLUSIONS_SHA256 = "17971869a3a292f35c41e6058f1556e65980e5cc481dfe8d4241e5f47874b056"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_go(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    parsed = ast.literal_eval(value) if isinstance(value, str) else list(value)
    return sorted(set(parsed))


def build(dataset, ontology, exclusions, output, manifest_path, count=100):
    expected = [(dataset, DATASET_SHA256), (ontology, ONTOLOGY_SHA256), (exclusions, EXCLUSIONS_SHA256)]
    for path, digest in expected:
        if sha256(path) != digest:
            raise ValueError(f"Pinned source hash mismatch: {path}")
    parents, aliases, obsolete = load_ontology(ontology)
    excluded_ids = {line.strip() for line in Path(exclusions).read_text().splitlines() if line.strip()}
    frame = pd.read_parquet(dataset)
    records = []
    for _, row in frame.iterrows():
        if row.protein_id in excluded_ids:
            continue
        aspects = {aspect: sorted(normalize_terms(parse_go(row[column]), parents, aliases, obsolete))
                   for aspect, column in (("bp", "go_bp"), ("mf", "go_mf"), ("cc", "go_cc"))}
        signature = json.dumps(aspects, sort_keys=True)
        records.append({"protein_id": row.protein_id, "sequence": row.sequence,
                        "organism": row.organism, "interpro_formatted": row.interpro_formatted,
                        "go_pred": row.go_pred, "aspects": aspects, "signature": signature})
    by_sequence = {}
    for row in records:
        by_sequence.setdefault(row["sequence"], []).append(row)
    inconsistent = {sequence for sequence, group in by_sequence.items()
                    if len({row["signature"] for row in group}) > 1}
    deduplicated = [sorted(group, key=lambda row: row["protein_id"])[0]
                    for sequence, group in by_sequence.items() if sequence not in inconsistent]
    selected = sorted(deduplicated,
                      key=lambda row: hashlib.sha256(f"{SEED}:{row['sequence']}".encode()).hexdigest())[:count]
    tasks = []
    for index, row in enumerate(selected):
        opaque = hashlib.sha256(f"{SEED}:{row['protein_id']}".encode()).hexdigest()[:16]
        tasks.append({"task_id": f"bioreason-go-{index:03d}-{opaque}", "opaque_id": opaque,
                      "sequence_sha256": hashlib.sha256(row["sequence"].encode()).hexdigest(),
                      "source_protein_id_sha256": hashlib.sha256(row["protein_id"].encode()).hexdigest()})
    manifest = {
        "version": 1, "dataset_revision": DATASET_REVISION, "source_revision": SOURCE_REVISION,
        "seed": SEED, "track": "official-context-assisted-annotation-recovery",
        "input_allowlist": ["sequence", "organism", "interpro_formatted", "go_pred"],
        "verifier_only": ["protein_id", "go_ids", "go_bp", "go_mf", "go_cc", "protein_names", "protein_function", "subcellular_location"],
        "rows": len(frame), "official_exclusion_ids": len(excluded_ids),
        "rows_after_official_exclusion": len(records), "unique_sequences_after_official_exclusion": len(by_sequence),
        "quarantined_inconsistent_sequence_groups": len(inconsistent),
        "deduplicated_clean_population": len(deduplicated), "count": count,
        "selection": f"lowest SHA256({SEED}:sequence) after official ID exclusion, inconsistent-target sequence quarantine, and exact sequence deduplication; frozen before outcomes",
        "success": "One final GO-ID list; alias-resolved is_a/part_of closure must equal released closure separately for every nonempty labeled aspect. Missing/extra terms fail; absent-label aspects ignored.",
        "tasks": tasks,
    }
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to alter frozen manifest")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if output.exists():
        existing = {path.name for path in output.iterdir() if (path / "task.toml").exists()}
        if existing == {row["task_id"] for row in tasks}:
            return manifest
        raise ValueError("Existing cohort differs from manifest")
    output.mkdir(parents=True)
    for item, row in zip(tasks, selected):
        task = output / item["task_id"]
        (task / "environment/data").mkdir(parents=True)
        (task / "tests").mkdir()
        case = {key: row[key] for key in manifest["input_allowlist"]}
        (task / "environment/data/case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2))
        (task / "environment/Dockerfile").write_text("FROM python:3.12-slim\nWORKDIR /app\nCOPY data /app/data\n")
        (task / "instruction.md").write_text(
            "Recover the frozen released Gene Ontology annotations for the anonymous protein in `/app/data/case.json`. "
            "This declared assisted track includes pinned GO-GPT predictions as context; inspect the sequence, organism, and InterPro context too. "
            "You may use `/tests/go-basic.obo` for ontology reasoning. Write exactly `/logs/artifacts/answer.json` as JSON "
            "`{\"go_ids\": [\"GO:0000000\", ...]}`. Include the terms you support; the verifier canonicalizes aliases and "
            "propagates `is_a`/`part_of`. Unknown/obsolete IDs, malformed output, missing or extra annotations in a labeled aspect fail. "
            "Predictions in aspects with no released label are ignored. No partial credit is awarded.\n")
        (task / "tests/gold.json").write_text(json.dumps({"aspects": row["aspects"]}, indent=2))
        shutil.copyfile(ontology, task / "tests/go-basic.obo")
        shutil.copyfile(ROOT / "verify.py", task / "tests/verify.py")
        (task / "tests/test.sh").write_text("#!/bin/sh\nset -eu\npython -I /tests/verify.py\n")
        (task / "tests/test.sh").chmod(0o755)
        (task / "task.toml").write_text(f'''schema_version = "1.0"
[metadata]
dataset = "wanglab/bioreason-pro-test-data"
dataset_revision = "{DATASET_REVISION}"
source_revision = "{SOURCE_REVISION}"
protocol = "bioreason-go-complete-v1"
track = "official-context-assisted-annotation-recovery"
[agent]
timeout_sec = 900.0
[verifier]
timeout_sec = 60.0
[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 1024
storage_mb = 512
''')
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--ontology", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "tasks-go100")
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "qualification/bioreason-go-manifest.json")
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.ontology, args.exclusions, args.output, args.manifest, args.count), indent=2)[:2000])
