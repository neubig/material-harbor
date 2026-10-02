"""Third workbook pass: fix stale cells flagged in review, and reconcile sheets.

Corrections applied here, each backed by a primary artifact or API check:

* D19  MatSciFig access is authorized now; 391,285 corpus rows verified. Those
       are corpus records, not released benchmark tasks, so the cell says so
       instead of repeating the obsolete gated blocker.
* E19  MatSciFig's own evaluation code uses MatSciBERT for text retrieval
       (m3rg-iitd/matscibert) with a CLIP image encoder, so that is the named
       domain model, not a general VLM.
* G19/J19 drop the gated-access blocker.
* E20-E23 replace the blanket "Intern-S1; task fit unvalidated" green with the
       evidence that actually exists: MATRIX-PT for MATRIX, and unknown (yellow)
       where no domain model is verified for that benchmark's own tasks.
* E24  names the exact paper model and checkpoint caveat.
* D22  MATCHA counts are now verified against the parsed files, and the stale
       "image ZIP unaudited" phrasing is replaced with what is and is not known.
* J21  MATRIX is Pending (yellow): no qualification gate is resolved because the
       measured number is a graded proxy rather than binary accuracy, so no red
       failure is asserted.

Usage:
    python research/update_workbook_vision3.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import openpyxl
from openpyxl.styles import PatternFill

ROOT = Path(__file__).resolve().parent.parent
WORKBOOK = ROOT / "GENESIS Benchmark Brainstorming.xlsx"
BACKUP = ROOT / "research" / "workbook-backup-pre-vision3.xlsx"

GREEN = PatternFill("solid", fgColor="FFC6EFCE")
YELLOW = PatternFill("solid", fgColor="FFFFEB9C")
RED = PatternFill("solid", fgColor="FFFFC7CE")

E19 = (
    "Yes — MatSciBERT retrieval. The official MatSciFig retrieval code pins "
    "text_encoder = m3rg-iitd/matscibert with image_encoder = "
    "openai/clip-vit-base-patch32 (CMEG-IITR/cmpfig retrieval/config.py). "
    "MATRIX-PT and general VLMs are not the benchmark's own model."
)
E20 = (
    "Unknown — no domain foundation model is verified against MatMech's tasks. "
    "The corpus is an extraction/analysis dataset rather than a benchmark with a "
    "released answer contract, so no model is demonstrated relevant here."
)
E21 = (
    "Yes — MATRIX-PT (radical-ai/MATRIX-PT), the benchmark's own post-training "
    "release: a LoRA adapter on Qwen/Qwen2-VL-7B, image-text-to-text."
)
E22 = (
    "Unknown — no domain foundation model is verified against MATCHA's questions. "
    "MatCha (FreedomIntelligence/MatCha) is the dataset itself, not a released "
    "scoring model, so task fit is unestablished."
)
E23 = (
    "Unknown — no domain foundation model is verified against CSMBench's tasks. "
    "The release ships MCQA data and a scoring script, not a domain model."
)
E24 = (
    "Caveat — the paper reports Intern-S1-Pro, which is NOT the public "
    "internlm/Intern-S1 checkpoint; the two must not be conflated. No model is "
    "verified against this subset's own tasks here, so fit stays unknown."
)

D22 = (
    "1,261 records / 1,500 nested VQA questions parsed, 1,500 distinct "
    "image-geometry tuples, 899 images in the released ZIP. Question count is "
    "verified; the full asset audit is incomplete (the earlier TIFF audit failed), "
    "though 10 image-bearing Harbor tasks now prove their figures ship correctly."
)

J21 = (
    "Pending (yellow, not a proven failure): the only MATRIX number is a "
    "judge-graded five-level proxy on an adapter-reconstructed rubric, not binary "
    "accuracy, so the 10-70% accuracy band has not been applied and no gate is "
    "resolved either way."
)

D19 = (
    "391,285 corpus rows verified under authorized access (92 parquet shards, "
    "48.7 GB). These are corpus records, NOT released benchmark tasks with an "
    "answer contract, so they do not count toward the 50-task bar."
)
G19 = "Unmeasured (corpus only; no released task set to run)"
J19 = "No (no released task/answer contract; corpus rows are not benchmark tasks)"

EDITS = [
    ("Sheet1", "D19", D19, YELLOW),
    ("Sheet1", "E19", E19, YELLOW),
    ("Sheet1", "G19", G19, YELLOW),
    ("Sheet1", "J19", J19, RED),
    ("Sheet1", "E20", E20, YELLOW),
    ("Sheet1", "E21", E21, GREEN),
    ("Sheet1", "E22", E22, YELLOW),
    ("Sheet1", "E23", E23, YELLOW),
    ("Sheet1", "E24", E24, YELLOW),
    ("Sheet1", "D22", D22, GREEN),
    ("Sheet1", "J21", J21, YELLOW),
    # Mirror the same corrections into the detailed sheet.
    ("Detailed assessments", "D19", D19, YELLOW),
    ("Detailed assessments", "E19", E19, YELLOW),
    ("Detailed assessments", "G19", G19, YELLOW),
    ("Detailed assessments", "I19", J19, RED),
    ("Detailed assessments", "E20", E20, YELLOW),
    ("Detailed assessments", "E21", E21, GREEN),
    ("Detailed assessments", "E22", E22, YELLOW),
    ("Detailed assessments", "E23", E23, YELLOW),
    ("Detailed assessments", "E24", E24, YELLOW),
    ("Detailed assessments", "D22", D22, GREEN),
]

EVIDENCE_ROWS = [
    (
        "MatSciFig access and evaluation code",
        "CMEG-IITR/cmpfig retrieval/config.py",
        "Authorized access verified: 391,285 corpus rows across 92 parquet shards "
        "(48.7 GB), which are corpus records rather than released benchmark tasks. "
        "The official retrieval code pins text_encoder m3rg-iitd/matscibert with "
        "image_encoder openai/clip-vit-base-patch32.",
    ),
    (
        "MATRIX-PT foundation model",
        "https://huggingface.co/radical-ai/MATRIX-PT",
        "Benchmark's own post-training release: LoRA adapter on Qwen/Qwen2-VL-7B, "
        "pipeline image-text-to-text.",
    ),
    (
        "MATCHA verified counts",
        "material-harbor/matcha/source/matcha_vqa_inputs.jsonl",
        "Parsed directly: 1,261 records, 1,500 nested VQA questions, 1,500 distinct "
        "image-geometry tuples, 899 images in the ZIP. Question count verified; full "
        "asset audit incomplete.",
    ),
]


def apply(dry_run: bool) -> dict:
    wb = openpyxl.load_workbook(WORKBOOK)
    changed = []
    for sheet, coord, value, fill in EDITS:
        ws = wb[sheet]
        changed.append(
            {"sheet": sheet, "cell": coord, "before": ws[coord].value, "after": value}
        )
        if not dry_run:
            ws[coord] = value
            ws[coord].fill = fill

    if not dry_run:
        ws = wb["Evidence"]
        start = 1
        for row in range(1, ws.max_row + 1):
            if any(ws.cell(row, col).value is not None for col in range(1, 4)):
                start = row + 1
        for source, url, note in EVIDENCE_ROWS:
            ws.cell(start, 1, source)
            ws.cell(start, 2, url)
            ws.cell(start, 3, note)
            start += 1
        wb.save(WORKBOOK)
    return {"changed": changed, "dry_run": dry_run}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not BACKUP.exists():
        shutil.copyfile(WORKBOOK, BACKUP)
    print(json.dumps(apply(args.dry_run), indent=2)[:6000])
