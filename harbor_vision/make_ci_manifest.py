"""Build CI manifests for the finished Harbor vision runs and summarise them.

The runs are small (n=10) fixed pilot sets, so the point estimate is weak and
the interval is the honest headline. Reuses the project's frozen
scripts/summarize_accuracy.py rather than reimplementing the bootstrap.

Usage:
    python harbor_vision/make_ci_manifest.py matcha-images-n10 csmbench-images-n10
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = HERE / "runs"
SUMMARY = ROOT / "scripts" / "summarize_accuracy.py"


def build(run_name: str) -> Path:
    run_dir = RUNS / run_name
    jobs = sorted(p for p in (run_dir / "jobs").iterdir() if p.is_dir())
    if len(jobs) != 1:
        raise SystemExit(f"{run_name}: expected one job dir, found {jobs}")

    trials = []
    expected_config = None
    expected_agent_info = None
    for result_path in sorted(jobs[0].glob("*/result.json")):
        data = json.loads(result_path.read_text())
        if not data.get("trial_name"):
            continue
        if expected_config is None:
            expected_config = data["config"]
            expected_agent_info = data["agent_info"]
        trials.append(
            {
                "task_id": data["task_name"],
                "result_path": str(result_path.relative_to(run_dir)),
                "paper_id": data["task_name"],
                "validation_issues": [],
            }
        )

    manifest = {
        "sampling_plan": (
            f"Fixed {len(trials)} tasks from the deterministic published selection "
            f"for {run_name}; selection frozen before outcomes were observed."
        ),
        "expected_config": expected_config,
        "expected_agent_info": expected_agent_info,
        "tasks": trials,
    }
    out = run_dir / "ci-manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


if __name__ == "__main__":
    for name in sys.argv[1:]:
        manifest = build(name)
        report = manifest.parent / "ci-report.json"
        proc = subprocess.run(
            [sys.executable, str(SUMMARY), str(manifest), "--scope", "iid",
             "--output", str(report)],
            capture_output=True, text=True,
        )
        print(f"--- {name} (exit {proc.returncode}) ---")
        print(proc.stdout.strip() or proc.stderr.strip())
