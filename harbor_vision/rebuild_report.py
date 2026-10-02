"""Rewrite a finished run's report.json from its on-disk evidence.

Needed because collect_rewards previously counted the job-level aggregate
result.json as a trial, so every pre-fix report carries one extra null-reward
row. Recomputing from the job directory reproduces the report without
re-running Harbor.

Usage:
    python harbor_vision/rebuild_report.py matcha-images-n10
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from harbor_vision.run_benchmark import (  # noqa: E402
    BASE_IMAGE,
    AGENT_IMPORT,
    RUNS,
    analyze_transport,
    collect_rewards,
)


def rebuild(name: str) -> dict:
    run_dir = RUNS / name
    report = json.loads((run_dir / "report.json").read_text())
    staged = report["staged_manifest"]

    job_dirs = sorted(p.name for p in (run_dir / "jobs").iterdir() if p.is_dir())
    if len(job_dirs) != 1:
        raise SystemExit(f"expected exactly one job dir, found {job_dirs}")

    rewards = collect_rewards(run_dir / "jobs", job_dirs[0])
    scored = [
        r["reward"] for r in rewards.values() if isinstance(r.get("reward"), (int, float))
    ]
    report["rewards"] = rewards
    report["scored_trials"] = len(scored)
    report["solved"] = sum(1 for r in scored if r and r > 0)
    report["mean_reward"] = (sum(scored) / len(scored)) if scored else None
    report["accuracy_percent"] = (100.0 * sum(scored) / len(scored)) if scored else None
    report["transport"] = analyze_transport(run_dir / "proxy", staged)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    for run_name in sys.argv[1:]:
        out = rebuild(run_name)
        print(
            json.dumps(
                {k: out[k] for k in ("run", "model", "scored_trials", "solved",
                                     "mean_reward", "accuracy_percent")},
                indent=2,
            )
        )
