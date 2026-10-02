"""Run any image-bearing Harbor task set with vision transport enabled and recorded.

Why a staging copy is made
--------------------------
Two edits are needed per task and neither belongs in the checked-in task
definitions:

1. The environment must inherit the prebaked image that already carries the
   PR5460 SDK. Harbor's installer detects the existing venv and skips its pip
   step, so the pinned SDK is what runs. Without this, Harbor installs released
   1.50.1, which drops images even with the vision override.
2. ``extra_hosts`` maps the recording proxy to host-gateway so the agent's
   ``LLM_BASE_URL`` can point at the proxy rather than the endpoint directly.

The proxy is what makes the run self-evidencing: it records, per request,
whether content was a list or a string and the SHA-256 of every image payload,
so the pixels can be matched against the task's own figure bytes.

Usage:
    python harbor_vision/run_benchmark.py \
        --source csmbench/tasks/pilot10 --name csmbench-images --count 10
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

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"

BASE_IMAGE = "material-harbor/openhands-sdk-vision:dad4aa5"
PROXY_HOST = "llm-proxy-host"
PROXY_PORT = 18110
AGENT_IMPORT = "harbor_vision.vision_openhands_sdk:VisionOpenHandsSDK"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(source: Path, destination: Path, count: int | None) -> list[dict]:
    """Copy tasks, rebasing each environment onto the prebaked vision image."""
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    staged = []
    for task_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        if not (task_dir / "task.toml").exists():
            continue
        if count is not None and len(staged) >= count:
            break

        target = destination / task_dir.name
        shutil.copytree(task_dir, target)

        dockerfile = target / "environment" / "Dockerfile"
        if dockerfile.exists():
            # Rebase the build onto the prebaked vision image so the pinned
            # PR5460 SDK is what runs instead of Harbor's installed release.
            _, _, rest = dockerfile.read_text().partition("\n")
            dockerfile.write_text(f"FROM {BASE_IMAGE}\n{rest}")
        elif BASE_IMAGE not in (target / "task.toml").read_text():
            continue

        (target / "environment" / "docker-compose.yaml").write_text(
            "services:\n"
            "  main:\n"
            "    extra_hosts:\n"
            f'      - "{PROXY_HOST}:host-gateway"\n'
        )

        images = []
        for path in (target / "environment").rglob("*"):
            if path.is_file() and path.suffix.lower() in (
                ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif", ".webp"
            ):
                images.append({"path": str(path.relative_to(target)), "sha256": sha256(path)})

        staged.append({"task_id": task_dir.name, "images": images})

    return staged


def start_proxy(log_dir: Path, seconds: float) -> subprocess.Popen:
    log_dir.mkdir(parents=True, exist_ok=True)
    upstream = os.environ["LLM_BASE_URL"].rstrip("/")
    if upstream.endswith("/v1"):
        upstream = upstream[: -len("/v1")]
    process = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "recording_proxy.py"),
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
    if process.poll() is not None:
        # Without this the port never binds, every agent gets a connection
        # error, and the run reports a plausible-looking 0% instead of failing.
        raise SystemExit(
            "recording proxy exited during startup: "
            + (process.stdout.read() if process.stdout else "")
        )
    return process


def analyze_transport(log_dir: Path, staged: list[dict]) -> dict:
    known = {}
    for task in staged:
        for image in task["images"]:
            known[image["sha256"]] = task["task_id"]

    requests = [json.loads(p.read_text()) for p in sorted(log_dir.glob("request-*.json"))]

    list_messages = string_messages = image_blocks = 0
    matched: dict[str, str] = {}
    for record in requests:
        for message in record.get("messages", []):
            content = message.get("content", {})
            kind = content.get("kind")
            if kind == "list":
                list_messages += 1
            elif kind == "string":
                string_messages += 1
            for image in content.get("images", []):
                image_blocks += 1
                if image.get("sha256") in known:
                    matched[image["sha256"]] = known[image["sha256"]]

    return {
        "recorded_requests": len(requests),
        "list_content_messages": list_messages,
        "string_content_messages": string_messages,
        "image_blocks_seen": image_blocks,
        "image_blocks_matching_task_images": len(matched),
        "task_images_available": len(known),
        "matched_task_ids": sorted(set(matched.values())),
    }


def collect_rewards(output: Path, job_name: str) -> dict:
    rewards = {}
    for path in (output / job_name).rglob("result.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # The job directory also holds an aggregate result.json (job id, stats)
        # alongside one per-trial file. Only trials carry a trial_name; counting
        # the aggregate would add a spurious null reward row.
        if not data.get("trial_name"):
            continue
        trial_name = data["trial_name"]
        rewards[trial_name] = {
            "reward": ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            "exception_type": (data.get("exception_info") or {}).get("exception_type"),
            "task_id": (data.get("task_id") or {}).get("path") if isinstance(
                data.get("task_id"), dict
            ) else None,
        }
    return rewards


def summarize_rewards(rewards: dict, expected: int) -> dict:
    scored = [r['reward'] for r in rewards.values()
              if isinstance(r.get('reward'), (int, float))]
    attempted = len(rewards)
    complete = attempted == expected and attempted > 0
    binary = all(value in (0, 1) for value in scored)
    return {
        'attempted_trials': attempted,
        'unobserved_tasks': max(0, expected - attempted),
        'scored_trials': len(scored),
        'missing_rewards': attempted - len(scored),
        'solved': sum(value == 1 for value in scored),
        'mean_reward': sum(scored) / attempted if attempted else None,
        'accuracy_percent': 100 * sum(scored) / attempted if complete and binary else None,
        'metric': 'binary' if binary else 'graded_proxy_not_accuracy',
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--model", default="openai/deepseek-v4.1-flash")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--timeout", type=float, default=5400)
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--verifier-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable for the verifier container; repeatable.",
    )
    args = parser.parse_args()

    run_dir = RUNS / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    staged = stage(ROOT / args.source, run_dir / "tasks", args.count)
    if not staged:
        raise SystemExit(f"no tasks staged from {args.source}")

    proxy = start_proxy(run_dir / "proxy", args.timeout + 300)
    job_name = f"{args.name}-{int(time.time())}"
    try:
        command = [
            str(ROOT / ".venv" / "bin" / "harbor"), "run",
            "-a", AGENT_IMPORT,
            "-m", args.model,
            "-p", str(run_dir / "tasks"),
            "-o", str(run_dir / "jobs"),
            "--job-name", job_name,
            "-y", "-q",
            "-n", str(args.concurrency),
            "--agent-timeout-multiplier", "2",
            "--ak", "vision_supports_vision=true",
            "--ak", "load_skills=false",
            "--ak", f"max_iterations={args.max_iterations}",
            "--ae", f"LLM_BASE_URL=http://{PROXY_HOST}:{PROXY_PORT}/v1",
        ]
        if args.version:
            command += ["--ak", f"version={args.version}"]
        for entry in args.verifier_env:
            command += ["--ve", entry]
        env = dict(os.environ)
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{existing}" if existing else str(ROOT)
        result = subprocess.run(
            command, capture_output=True, text=True, env=env, timeout=args.timeout
        )
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proxy.kill()

    transport = analyze_transport(run_dir / "proxy", staged)
    rewards = collect_rewards(run_dir / "jobs", job_name)
    report = {
        "run": args.name,
        "source": str(args.source),
        "model": args.model,
        "base_image": BASE_IMAGE,
        "agent_import_path": AGENT_IMPORT,
        "sdk_commit": "dad4aa50208c27bf60cea4040b1db4dee2fd71c3",
        "vision_supports_vision": True,
        "max_iterations": args.max_iterations,
        "staged_tasks": len(staged),
        "task_count_with_images": sum(1 for t in staged if t["images"]),
        "transport": transport,
        "rewards": rewards,
        **summarize_rewards(rewards, len(staged)),
        "harbor_returncode": result.returncode,
        "harbor_stdout_tail": result.stdout[-3000:],
        "harbor_stderr_tail": result.stderr[-3000:],
        "staged_manifest": staged,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps({k: report[k] for k in (
        "run", "model", "staged_tasks", "scored_trials", "solved",
        "mean_reward", "accuracy_percent", "transport",
    )}, indent=2))


if __name__ == "__main__":
    main()
