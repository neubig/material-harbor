"""Run the vision transport probe through real Harbor and record transport evidence.

What a run establishes
----------------------
Every task is image-bearing: the answer is a random six-digit code and shape
census that appears nowhere in the prompt. A score above zero therefore requires
the pixels to have arrived and been read, and a score of zero points at
transport. The adapter enables the vision capability
(``capability_overrides={'supports_vision': True}``) so the PR5460 fix keeps
image blocks alive instead of letting the string serializer drop them.

The recording proxy adds byte-level evidence: for each API request it logs
whether the message content was a list or a string, how many image blocks it
carried, and the SHA-256 of the decoded image bytes. That hash is compared
against the figure actually shipped in the task, so "the pixels arrived" is
verifiable rather than inferred.

Usage:
    python harbor_vision/vision_probe/run.py --count 8 --seed 20260814
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

BASE_IMAGE = "material-harbor/openhands-sdk-vision:dad4aa5"
PROXY_HOST = "llm-proxy-host"
PROXY_PORT = 18110
# The agent's custom import path; Harbor accepts `module.path:ClassName`.
AGENT_IMPORT = "harbor_vision.vision_openhands_sdk:VisionOpenHandsSDK"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_tasks(destination: Path, count: int, seed: int) -> dict:
    if destination.exists():
        shutil.rmtree(destination)
    subprocess.run(
        [
            sys.executable,
            str(HERE / "build.py"),
            "--output",
            str(destination),
            "--count",
            str(count),
            "--seed",
            str(seed),
            "--base-image",
            BASE_IMAGE,
            "--proxy-host",
            PROXY_HOST,
        ],
        check=True,
    )
    return json.loads((destination / "manifest.json").read_text())


def start_proxy(log_dir: Path, seconds: float) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    upstream = os.environ["LLM_BASE_URL"]
    # The agent points at http://<proxy>/v1 and LiteLLM appends /chat/completions,
    # so the proxied path already carries the version segment. Upstream must be the
    # bare origin or the version would be duplicated and the API would 404.
    upstream = upstream.rstrip("/")
    if upstream.endswith("/v1"):
        upstream = upstream[: -len("/v1")]
    process = subprocess.Popen(
        [
            sys.executable,
            str(HERE.parent / "recording_proxy.py"),
            "--port",
            str(PROXY_PORT),
            "--upstream",
            upstream,
            "--log-dir",
            str(log_dir),
            "--record-requests",
            "--seconds",
            str(seconds),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(2)
    return process


def analyze_transport(log_dir: Path, manifest: dict) -> dict:
    """Summarize recorded requests and check image hashes against task figures."""
    figure_hashes = {}
    for task in manifest["tasks"]:
        figure = (
            RUNS / manifest["run"] / "tasks" / task["task_id"]
            / "environment" / "data" / "figure.png"
        )
        if figure.exists():
            figure_hashes[sha256(figure)] = task["task_id"]

    requests = []
    for path in sorted(log_dir.glob("request-*.json")):
        requests.append(json.loads(path.read_text()))

    list_requests = 0
    string_requests = 0
    image_requests = 0
    matched_hashes = set()
    for record in requests:
        for message in record.get("messages", []):
            content = message.get("content", {})
            if content.get("kind") == "list":
                list_requests += 1
            elif content.get("kind") == "string":
                string_requests += 1
            for image in content.get("images", []):
                image_requests += 1
                if image.get("sha256") in figure_hashes:
                    matched_hashes.add(image["sha256"])

    return {
        "recorded_requests": len(requests),
        "list_content_messages": list_requests,
        "string_content_messages": string_requests,
        "image_blocks_seen": image_requests,
        "image_blocks_matching_task_figures": len(matched_hashes),
        "task_figures_available": len(figure_hashes),
        "matched_task_ids": sorted(figure_hashes[h] for h in matched_hashes),
    }


def run_trial(task_dir: Path, output: Path, model: str, version: str | None,
              timeout: float) -> dict:
    job_name = f"vp-images-{int(time.time())}"
    command = [
        str(ROOT / ".venv" / "bin" / "harbor"),
        "run",
        "-a",
        AGENT_IMPORT,
        "-m",
        model,
        "-p",
        str(task_dir),
        "-o",
        str(output),
        "--job-name",
        job_name,
        "-y",
        "-q",
        "-n",
        "2",
        "--agent-timeout-multiplier",
        "2",
        "--ak",
        "vision_supports_vision=true",
        "--ak",
        "load_skills=false",
        "--ak",
        "max_iterations=20",
        "--ae",
        f"LLM_BASE_URL=http://{PROXY_HOST}:{PROXY_PORT}/v1",
    ]
    if version:
        command += ["--ak", f"version={version}"]
    env = dict(os.environ)
    # Harbor imports the custom agent class in its own process; the repo root must
    # be importable there, which a subprocess does not inherit from cwd alone.
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing}" if existing else str(ROOT)
    result = subprocess.run(
        command, capture_output=True, text=True, env=env, timeout=timeout
    )
    return {
        "job_name": job_name,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def collect_rewards(output: Path, job_name: str) -> dict:
    """Read rewards from Harbor's per-trial result.json files."""
    rewards = {}
    for path in (output / job_name).rglob("result.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rewards_for_trial = (data.get("verifier_result") or {}).get("rewards") or {}
        # Prefer the trial directory name: task_name is generic ("task") for
        # locally built task dirs, while the trial name carries the task id.
        trial_name = data.get("trial_name") or path.parent.name
        rewards[trial_name] = {
            "reward": rewards_for_trial.get("reward"),
            "exception_type": (data.get("exception_info") or {}).get("exception_type"),
            "task_checksum": data.get("task_checksum"),
        }
    return rewards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--model", default="openai/deepseek-v4.1-flash")
    parser.add_argument("--version", default=None)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    run_name = f"images-seed{args.seed}-n{args.count}"
    run_dir = RUNS / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_tasks(run_dir / "tasks", args.count, args.seed)
    manifest["run"] = run_name
    manifest["model"] = args.model
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    proxy_logs = run_dir / "proxy"
    proxy = start_proxy(proxy_logs, args.timeout + 300)
    try:
        outcome = run_trial(
            run_dir / "tasks", run_dir / "jobs", args.model, args.version, args.timeout
        )
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proxy.kill()

    transport = analyze_transport(proxy_logs, manifest)
    rewards = collect_rewards(run_dir / "jobs", outcome["job_name"])
    scored = [
        r["reward"]
        for r in rewards.values()
        if isinstance(r.get("reward"), (int, float))
    ]

    report = {
        "run": run_name,
        "model": args.model,
        "base_image": BASE_IMAGE,
        "agent_import_path": AGENT_IMPORT,
        "sdk_commit": "dad4aa50208c27bf60cea4040b1db4dee2fd71c3",
        "task_count": args.count,
        "seed": args.seed,
        "transport": transport,
        "rewards": rewards,
        "scored_trials": len(scored),
        "mean_reward": (sum(scored) / len(scored)) if scored else None,
        "solved": sum(1 for r in scored if r and r > 0),
        "harbor": outcome,
        "task_manifest": manifest["tasks"],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: report[k] for k in (
        "run", "model", "scored_trials", "mean_reward", "solved", "transport"
    )}, indent=2))


if __name__ == "__main__":
    main()
