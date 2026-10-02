"""Sixth workbook pass: correct the OmniMatBench headline metric.

The previous pass presented a conditional mean (0.957 over the 7 trials that
produced an answer) as the row's number and described the other three trials as
"infrastructure losses". Both were wrong for a fixed-budget benchmark.

Running out of the agent's own iteration budget is an end-to-end task failure,
not an infrastructure fault: those trials get zero, exactly like a wrong answer.
So the primary number is the all-attempted mean 0.670 (6.7/10), and the
conditional mean is reported as a clearly secondary, selection-conditioned view.

One trial did hit a genuine harness exception (AgentTimeoutError, 1800 s, with
null agent metrics). That missingness is recorded separately rather than being
silently pooled or used to excuse the other two.

Usage:
    python research/update_workbook_vision6.py
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
BACKUP = ROOT / "research" / "workbook-backup-pre-vision6.xlsx"

YELLOW = PatternFill("solid", fgColor="FFFFEB9C")

G24 = (
    "PROXY METRIC, not binary accuracy: 0.670 mean over ALL 10 attempted trials "
    "(6.7/10; fixed-budget protocol, the 3 trials that produced no answer count "
    "as 0) in Harbor run omnimatbench-qa-vision-n10, deepseek-v4.1-flash, real "
    "figures. Secondary, selection-conditioned view: 0.957 over only the 7/10 "
    "trials that answered (worst 0.8) -- do not read this as the benchmark "
    "result. Of the 3 unanswered: 2 exhausted the agent iteration budget (a task "
    "failure, not infrastructure) and 1 hit a harness AgentTimeoutError. Judge "
    "gpt-5.1; the release publishes key points but no runnable QA scorer, so the "
    "coverage judgement is this adapter's."
)

J24 = (
    "Pending (yellow): the primary measured number is a judge-graded coverage "
    "proxy, not binary accuracy, so no accuracy-band gate is resolved; the "
    "all-attempted mean 0.670 sits inside the band but the proxy is not "
    "comparable to it, and the conditional 0.957 is selection-conditioned. Image "
    "transport itself is proven: all 10 task figures were byte-for-byte verified "
    "in the recorded requests."
)

D24 = (
    "1,000 problems (498 QA; 502 CAL). 140 QA items are image-bearing and were "
    "the measured scope."
)

EVIDENCE = [
    (
        "OmniMatBench all-attempted vs conditional mean",
        "harbor_vision/runs/omnimatbench-qa-vision-n10",
        "Fixed-budget protocol: all 10 attempted trials, unanswered counted as 0, "
        "gives mean 0.670 (6.7/10). Conditional on answering, the 7 judged trials "
        "give 0.957 (6.7/7). Of the 3 unanswered trials, 2 (06-009, 13-006) "
        "raised no exception and simply exhausted the 40-iteration agent budget "
        "(31,660 and 48,756 output tokens) -- an end-to-end task failure, not an "
        "infrastructure fault; 1 (16-011) raised AgentTimeoutError after 1800 s "
        "with null agent metrics and is the only genuine harness exception.",
    ),
    (
        "OmniMatBench image transport",
        "harbor_vision/runs/omnimatbench-qa-vision-n10/report.json",
        "169 recorded requests; all 10 staged task figures matched byte-for-byte "
        "in the transported image blocks, so the figures genuinely reached the "
        "model.",
    ),
]

EDITS = [
    ("Sheet1", "D24", D24, None),
    ("Sheet1", "G24", G24, YELLOW),
    ("Sheet1", "J24", J24, YELLOW),
    ("Detailed assessments", "D24", D24, None),
    ("Detailed assessments", "G24", G24, YELLOW),
    ("Detailed assessments", "I24", J24, YELLOW),
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
            if fill is not None:
                ws[coord].fill = fill
    if not dry_run:
        ws = wb["Evidence"]
        start = 1
        for row in range(1, ws.max_row + 1):
            if any(ws.cell(row, col).value is not None for col in range(1, 4)):
                start = row + 1
        for source, url, note in EVIDENCE:
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
