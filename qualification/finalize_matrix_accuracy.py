#!/usr/bin/env python3
"""Replay the frozen MATRIX verifier over preserved agent answers only."""
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from matrix.verify import ENDPOINT, MODEL, judge

RUN = ROOT / "harbor_vision/runs/matrix-binary-vision-qualification-n100-network-retry"
CONFIDENCE = 1 - 0.05 / 3


def wilson(k, n):
    z = NormalDist().inv_cdf(1 - (1 - CONFIDENCE) / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [c - r, c + r]


def grade(item):
    task_id, answer_path = item
    gold = json.loads((ROOT / "matrix/tasks-binary-vision100" / task_id / "tests/gold.json").read_text())
    if answer_path is None:
        return {"task_id": task_id, "kind": gold["kind"], "status": "missing_answer", "graded_reward": None, "binary_reward": 0}
    answer = answer_path.read_text(errors="strict").strip()
    if not answer:
        return {"task_id": task_id, "kind": gold["kind"], "status": "missing_answer", "graded_reward": None, "binary_reward": 0}
    for attempt in range(3):
        try:
            result = judge(gold, answer)
            return {"task_id": task_id, "kind": gold["kind"], "status": "judged",
                    "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                    "graded_reward": result["score"], "binary_reward": int(result["score"] == 1),
                    "judgment": result}
        except Exception as exc:
            error = type(exc).__name__
            time.sleep(2 ** attempt)
    return {"task_id": task_id, "kind": gold["kind"], "status": "judge_infrastructure_error",
            "error_type": error, "graded_reward": None, "binary_reward": 0}


def main():
    os.environ["MATRIX_JUDGE_URL"] = ENDPOINT
    os.environ["MATRIX_JUDGE_MODEL"] = MODEL
    if "MATRIX_JUDGE_API_KEY" not in os.environ:
        raise SystemExit("MATRIX_JUDGE_API_KEY is required")
    expected = [row["task_id"] for row in json.loads((ROOT / "qualification/matrix-binary-vision-manifest.json").read_text())["tasks"]]
    found = {}
    for result_path in RUN.rglob("result.json"):
        value = json.loads(result_path.read_text())
        task_id = value.get("task_name")
        if not task_id:
            continue
        trial = result_path.parent
        candidates = list((trial / "artifacts/logs/artifacts").glob("answer.txt"))
        found[task_id] = candidates[0] if candidates else None
    if set(found) != set(expected):
        raise SystemExit(f"run incomplete: found {len(found)} of {len(expected)} scheduled task results")
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(grade, (task_id, found[task_id])) for task_id in expected]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["task_id"])
    successes = sum(row["binary_reward"] for row in rows)
    report = {"version": 1, "protocol": "matrix-full-credit-binary-v1", "model": "openai/deepseek-v4.1-flash",
              "verifier_model": MODEL, "scheduled": 100, "attempted": 100, "successes": successes,
              "failures_including_missing": 100 - successes,
              "missing_or_infrastructure_counted_zero": sum(row["status"] != "judged" for row in rows),
              "accuracy": successes / 100, "confidence": CONFIDENCE, "wilson": wilson(successes, 100),
              "cluster_bootstrap": None,
              "cluster_caveat": "The release provides no paper/source grouping for MATRIX vision rows. A population-independent cluster interval is invalid; if the n=100 Wilson look is otherwise conclusive, the precommitted full 250-image population must still be run for a definitive finite-population decision.",
              "verifier_replay_reason": "The Harbor agent attempts are preserved unchanged. Their in-container verifier lacked injected configuration and produced no reward; only the same frozen verifier was replayed over the preserved answer bytes.",
              "fidelity_caveat": "MATRIX publishes vision kinds and a loader but no vision rubrics. This is a reconstructed official-style full-credit binary endpoint, not an official native vision verifier.",
              "rows": rows}
    (ROOT / "qualification/matrix-accuracy-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
