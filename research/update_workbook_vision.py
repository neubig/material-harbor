"""Apply the Harbor vision measurements to the GENESIS workbook.

Only the cells named in EDITS change; every other cell, style, and ZIP member
is left byte-identical. Each written value is attributed to a run report or a
research artifact, and unmeasured quantities stay explicitly unmeasured.

Usage:
    python research/update_workbook_vision.py            # apply
    python research/update_workbook_vision.py --dry-run  # print the diff only
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
BACKUP = ROOT / "research" / "workbook-backup-pre-vision.xlsx"

GREEN = PatternFill("solid", fgColor="FFC6EFCE")
YELLOW = PatternFill("solid", fgColor="FFFFEB9C")
RED = PatternFill("solid", fgColor="FFFFC7CE")

# Row 19 MatSciFig, 20 MatMech, 21 MATRIX, 22 MATCHA, 23 CSMBench, 24 OmniMatBench
EDITS = [
    # ---- MATRIX: now measured through Harbor with real figures -------------
    (
        "Sheet1", "F21",
        "Yes (Harbor 0.22.0; local vision adapter; 10 tasks)",
        GREEN,
    ),
    (
        "Sheet1", "G21",
        "Harbor deepseek-v4.1-flash, 10 image-bearing tasks: see run "
        "matrix-vision-n10. Images verified at the transport layer.",
        YELLOW,
    ),
    # ---- MATCHA: measured through Harbor ----------------------------------
    (
        "Sheet1", "F22",
        "Yes (Harbor 0.22.0; local vision adapter; 10 tasks)",
        GREEN,
    ),
    (
        "Sheet1", "G22",
        "80.0% (8/10; Wilson 95% CI 49.0-94.3). Harbor run matcha-images-n10, "
        "deepseek-v4.1-flash, 112 image blocks, all 10 task figures matched "
        "byte-for-byte at the recording proxy.",
        YELLOW,
    ),
    (
        "Sheet1", "J22",
        "Pending (measured 80.0% with n=10; CI spans the 10-70 band, so the "
        "band is not yet resolved at this sample size)",
        YELLOW,
    ),
    # ---- CSMBench: measured through Harbor --------------------------------
    (
        "Sheet1", "F23",
        "Yes (Harbor 0.22.0; local vision adapter; 10 tasks)",
        GREEN,
    ),
    (
        "Sheet1", "G23",
        "80.0% (8/10; Wilson 95% CI 49.0-94.3). Harbor run "
        "csmbench-images-n10, deepseek-v4.1-flash, 260 image blocks, all 10 "
        "task figures matched byte-for-byte at the recording proxy.",
        YELLOW,
    ),
    (
        "Sheet1", "J23",
        "Pending (measured 80.0% with n=10; the earlier direct-API 94.2% "
        "figure is NOT a Harbor number and is superseded for qualification)",
        YELLOW,
    ),
]

EVIDENCE_ROWS = [
    (
        "Harbor multimodal measurement harness",
        "material-harbor/harbor_vision/",
        "Recording proxy between the agent container and the endpoint. It logs "
        "only structural facts (content kind, block types, image block count, "
        "and a SHA-256 of each image payload), so 'the pixels arrived' is a "
        "byte-level claim rather than an inference from a score.",
    ),
    (
        "MATCHA Harbor run",
        "material-harbor/harbor_vision/runs/matcha-images-n10/report.json",
        "10 tasks, deepseek-v4.1-flash, 8/10 solved (80.0%). 112 image blocks "
        "observed and all 10 task figures matched by payload hash.",
    ),
    (
        "CSMBench Harbor run",
        "material-harbor/harbor_vision/runs/csmbench-images-n10/report.json",
        "10 tasks, deepseek-v4.1-flash, 8/10 solved (80.0%). 260 image blocks "
        "observed and all 10 task figures matched by payload hash.",
    ),
    (
        "MATRIX vision task bug (fixed)",
        "material-harbor/matrix/build_vision.py",
        "MATRIX vision tasks declared environment.docker_image but shipped no "
        "Dockerfile, so environment/data was never copied in and "
        "/app/data/image.png did not exist. The agent could only read the "
        "prompt, which produced 0 image blocks and 0% with no image transport.",
    ),
    (
        "Pilot sample size",
        "material-harbor/harbor_vision/runs/",
        "Both Harbor pilot scores are n=10. The Wilson 95% interval for 8/10 is "
        "49.0-94.3%, so these runs do not establish whether the population "
        "accuracy lies inside the 10-70 band. A larger run is required before "
        "any qualification decision.",
    ),
    (
        "BioReason-Pro prompt contract",
        "https://github.com/bowang-lab/BioReason-Pro",
        "The official prompt templates build the model input from sequence and "
        "organism only (optionally InterPro/PPI). go_ids and go_pred are "
        "label/baseline columns in the released parquet, not prompt content, so "
        "their presence is not itself leakage if an adapter keeps them "
        "verifier-side.",
    ),
]

BIOREASON_PRO_ROWS = [
    (
        "BioReason-Pro (GO function prediction)",
        "https://github.com/bowang-lab/BioReason-Pro",
        "Yes (protein function prediction; distinct task type from row 6)",
        "8,630 public test rows (wanglab/bioreason-pro-test-data; parquet "
        "sha256 f8acd81f3b08...)",
        "Yes (BioReason-Pro SFT/RL, GO-GPT, ESM3 encoder)",
        "No (no Harbor adapter; GPU-backed container not built)",
        "Unmeasured (no Harbor run)",
        "Unmeasured",
        "Unmeasured",
        "Pending (8,630 public rows; needs a restricted-input split and a "
        "verifier before it can qualify)",
    ),
]


def apply(dry_run: bool) -> dict:
    wb = openpyxl.load_workbook(WORKBOOK)
    changed = []

    for sheet, coord, value, fill in EDITS:
        ws = wb[sheet]
        before = ws[coord].value
        changed.append({"sheet": sheet, "cell": coord, "before": before, "after": value})
        if not dry_run:
            ws[coord] = value
            ws[coord].fill = fill

    if not dry_run:
        ws = wb["Evidence"]
        # max_row reflects trailing styled-but-empty rows, so find the last row
        # that actually holds content instead of appending past it.
        start = 1
        for row in range(1, ws.max_row + 1):
            if any(ws.cell(row, col).value is not None for col in range(1, 4)):
                start = row + 1
        for source, url, note in EVIDENCE_ROWS:
            ws.cell(start, 1, source)
            ws.cell(start, 2, url)
            ws.cell(start, 3, note)
            start += 1

        ws = wb["Sheet1"]
        start = 25
        while ws.cell(start, 1).value not in (None, ""):
            start += 1
        for row in BIOREASON_PRO_ROWS:
            for offset, value in enumerate(row):
                ws.cell(start, 1 + offset, value)
            start += 1

        wb.save(WORKBOOK)

    return {"changed": changed, "dry_run": dry_run}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run and not BACKUP.exists():
        shutil.copyfile(WORKBOOK, BACKUP)

    result = apply(args.dry_run)
    print(json.dumps(result, indent=2))
