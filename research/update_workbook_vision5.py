"""Fifth workbook pass: reconcile the Detailed assessments sheet with Sheet1.

The detailed sheet still carried pre-Harbor text for MATRIX, MATCHA and CSMBench
("No Harbor run", "not ported locally") and two overall-fit cells that leaned on
no-image control runs, which the user overrode. This pass makes those rows state
what was actually measured and where the uncertainty sits.

Usage:
    python research/update_workbook_vision5.py
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
BACKUP = ROOT / "research" / "workbook-backup-pre-vision5.xlsx"

GREEN = PatternFill("solid", fgColor="FFC6EFCE")
YELLOW = PatternFill("solid", fgColor="FFFFEB9C")

F21 = "Yes (Harbor 0.22.0; local vision adapter; 10 of 250 vision rows)"
G21 = (
    "PROXY METRIC, not binary accuracy: mean five-level graded reward 0.800 over "
    "10/10 trials (Harbor run matrix-vision-n10, deepseek-v4.1-flash, real "
    "figures; levels 0, 0.5, 0.75, 0.75, 1, 1, 1, 1, 1, 1). Judge gpt-5.1; the "
    "release publishes no vision rubrics, so the rubric is adapter-reconstructed "
    "and the number is not official accuracy."
)
I21 = (
    "Pending (yellow, not a proven failure). Adapter and run both work; what is "
    "missing is a binary-accuracy gate, because the only number is a graded "
    "proxy on a reconstructed rubric. Three text kinds were also reachable; the "
    "measured scope was the 10 vision tasks."
)

F22 = "Yes (Harbor 0.22.0; local vision adapter; 10 of 1,500 questions)"
G22 = (
    "80.0% (8/10; Wilson 95% CI 49.0-94.3) in Harbor run matcha-images-n10, "
    "deepseek-v4.1-flash. All 10 task figures matched byte-for-byte across 112 "
    "recorded image blocks, so the images genuinely reached the model. The "
    "earlier direct-API 54.2% (65/120) figure is NOT a Harbor number."
)
I22 = (
    "Pending. Measured 80.0% at n=10; the CI spans the 10-70% band, so the band "
    "is not resolved at this sample size. Image delivery is proven; what is "
    "missing is a larger sample and an independent verifier FP/FN audit."
)

F23 = "Yes (Harbor 0.22.0; local vision adapter; 10 of 1,041 figures)"
G23 = (
    "80.0% (8/10; Wilson 95% CI 49.0-94.3) in Harbor run csmbench-images-n10, "
    "deepseek-v4.1-flash. All 10 task figures matched byte-for-byte across 260 "
    "recorded image blocks. The earlier direct-API 94.2% (113/120) figure is NOT "
    "a Harbor number and is superseded for qualification."
)
I23 = (
    "Pending. Measured 80.0% at n=10, which is inside the band, but the CI is "
    "wide and the verifier FP/FN audit is absent, so no gate is resolved. The "
    "two dataset configurations reuse the same figures and must not be summed."
)

I24 = (
    "Pending. Image-bearing QA ran in Harbor (see the main sheet); the text-only "
    "CAL cohort is a separate path and is not evidence about image understanding."
)

EDITS = [
    ("Detailed assessments", "F21", F21, GREEN),
    ("Detailed assessments", "G21", G21, YELLOW),
    ("Detailed assessments", "I21", I21, YELLOW),
    ("Detailed assessments", "F22", F22, GREEN),
    ("Detailed assessments", "G22", G22, YELLOW),
    ("Detailed assessments", "I22", I22, YELLOW),
    ("Detailed assessments", "F23", F23, GREEN),
    ("Detailed assessments", "G23", G23, YELLOW),
    ("Detailed assessments", "I23", I23, YELLOW),
    ("Detailed assessments", "I24", I24, YELLOW),
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
        wb.save(WORKBOOK)
    return {"changed": changed, "dry_run": dry_run}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not BACKUP.exists():
        shutil.copyfile(WORKBOOK, BACKUP)
    print(json.dumps(apply(args.dry_run), indent=2)[:4000])
