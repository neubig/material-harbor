#!/usr/bin/env python3
"""Replay MATRIX verification in task containers over preserved Harbor artifacts."""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "harbor_vision/runs/matrix-binary-vision-qualification-n100-network-retry"
RUNS = [RUN]
TASKS = ROOT / "matrix/tasks-binary-vision100"
ENDPOINT = "https://llm-proxy.app.all-hands.dev/v1/chat/completions"
MODEL = "gpt-5.1"
CONFIDENCE = 1 - 0.05 / 3


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k, n):
    z = NormalDist().inv_cdf(1 - (1 - CONFIDENCE) / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - r), min(1, c + r)]


def find_trials():
    found = {}
    for run in RUNS:
        for path in run.rglob("result.json"):
            value = json.loads(path.read_text())
            task = value.get("task_name")
            if task:
                if task in found:
                    raise ValueError(f"Duplicate completed task across runs: {task}")
                found[task] = path.parent
    return found


def replay(item):
    task_id, trial = item
    task = TASKS / task_id
    answers = list((trial / "artifacts/logs/artifacts").glob("answer.txt"))
    if not answers or not answers[0].read_bytes().strip():
        return {"task_id": task_id, "status": "missing_answer", "binary_reward": 0}
    answer = answers[0]
    tag = "matrix-replay-" + task_id.rsplit("-", 1)[-1][:16]
    out = Path(tempfile.mkdtemp(prefix=tag + "-", dir="/tmp"))
    env = os.environ.copy()
    env["MATRIX_JUDGE_API_KEY"] = env["LLM_API_KEY"]
    build = subprocess.run(["docker", "build", "-q", "-t", tag, str(task / "environment")],
                           text=True, capture_output=True, timeout=600)
    if build.returncode:
        shutil.rmtree(out, ignore_errors=True)
        return {"task_id": task_id, "status": "container_build_error", "binary_reward": 0,
                "diagnostic": build.stderr[-500:]}
    image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}", tag], text=True).strip()
    command = ["docker", "run", "--rm", "--network", "host",
               "-e", "MATRIX_JUDGE_API_KEY", "-e", f"MATRIX_JUDGE_URL={ENDPOINT}",
               "-e", f"MATRIX_JUDGE_MODEL={MODEL}",
               "-v", f"{answer.parent.resolve()}:/logs/artifacts:ro",
               "-v", f"{(task / 'tests').resolve()}:/tests:ro",
               "-v", f"{out.resolve()}:/logs/verifier:rw", tag, "/bin/sh", "/tests/test.sh"]
    run = subprocess.run(command, env=env, text=True, capture_output=True, timeout=360)
    result_path = out / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        reward = result.get("reward")
        row = {"task_id": task_id, "status": result.get("status"),
               "graded_reward": reward if type(reward) in (int, float) else None,
               "binary_reward": int(reward == 1), "answer_sha256": sha(answer),
               "container_image_id": image_id, "verifier_result_sha256": sha(result_path)}
        if result.get("error_type"):
            row["error_type"] = result["error_type"]
    else:
        row = {"task_id": task_id, "status": "container_verifier_error", "binary_reward": 0,
               "answer_sha256": sha(answer), "container_image_id": image_id,
               "diagnostic": (run.stderr or run.stdout)[-500:]}
    subprocess.run(["docker", "image", "rm", tag], text=True, capture_output=True)
    shutil.rmtree(out, ignore_errors=True)
    return row


def main():
    global RUN, RUNS, TASKS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=str(RUN.relative_to(ROOT)),
                        help="Comma-separated disjoint run directories")
    parser.add_argument("--tasks", default=str(TASKS.relative_to(ROOT)))
    parser.add_argument("--output", default="qualification/matrix-accuracy-report.json")
    args = parser.parse_args()
    RUNS = [ROOT / value for value in args.run.split(",")]
    RUN = RUNS[0]
    TASKS = ROOT / args.tasks
    if "LLM_API_KEY" not in os.environ:
        raise SystemExit("LLM_API_KEY is required")
    expected = sorted(path.name for path in TASKS.iterdir() if path.is_dir())
    total = len(expected)
    trials = find_trials()
    if set(trials) != set(expected):
        raise SystemExit(f"run incomplete: {len(trials)} of {total} scheduled results")
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(replay, [(task_id, trials[task_id]) for task_id in expected]))
    rows.sort(key=lambda row: row["task_id"])
    successes = sum(row["binary_reward"] for row in rows)
    fractional = {}
    for row in rows:
        key = "missing_or_error" if row.get("graded_reward") is None else str(row["graded_reward"])
        fractional[key] = fractional.get(key, 0) + 1
    report = {"version": 2, "protocol": "matrix-full-credit-binary-v1",
              "execution": "Each retained answer was verified by /tests/test.sh inside its task Dockerfile image. The agent was not rerun. Missing artifacts and verifier infrastructure failures count as zero.",
              "source_runs": [str(run.relative_to(ROOT)) for run in RUNS], "solver_model": "openai/deepseek-v4.1-flash",
              "verifier_model": MODEL, "scheduled": total, "attempted": total,
              "retained_answer_artifacts": sum(row["status"] != "missing_answer" for row in rows),
              "successes": successes, "failures_including_missing": total - successes,
              "binary_accuracy": successes / total, "binary_mapping": "Only exact rubric score 1 succeeds.",
              "fractional_score_distribution": fractional,
              "missing_or_infrastructure_counted_zero": sum(row["status"] != "judged" for row in rows),
              "confidence": CONFIDENCE, "wilson": wilson(successes, total),
              "container_provenance": {"dockerfile_sha256": sha(next(TASKS.glob("*/environment/Dockerfile"))),
                                       "test_sh_sha256": sha(next(TASKS.glob("*/tests/test.sh"))),
                                       "verify_binary_sha256": sha(next(TASKS.glob("*/tests/verify_binary.py"))),
                                       "network": "host access enabled only for authorized GPT-5.1 judge endpoint"},
              "inference_caveat": "MATRIX provides no released paper/source grouping. The primary100 and frozen additional140 together form the complete predeclared bounded 240-image population.",
              "fidelity_caveat": "MATRIX publishes five vision kinds and a loader, but no vision evaluation rubric. This is a derived adapter rubric, not an official or reconstructed-official rubric.",
              "rows": rows}
    (ROOT / args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
