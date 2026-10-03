#!/usr/bin/env python3
"""Audit MATRIX image transport and classify every missing answer.

The audit is retrospective and never reruns an agent. It requires an exact source
image byte hash in a recorded outbound model request during the trial window.
Source staging, an image tool call, or an unrelated concurrent image does not
count as transport proof.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ROOT / "harbor_vision/runs/matrix-binary-vision-qualification-n100-network-retry",
    ROOT / "harbor_vision/runs/matrix-binary-vision-qualification-additional140",
    ROOT / "harbor_vision/runs/matrix-binary-vision-qualification-additional140-resume109",
    ROOT / "harbor_vision/runs/matrix-binary-vision-qualification-additional140-restart2-remaining14",
]
DETAIL_SOURCES = [
    RUNS[0] / "container-replay-detail.json",
    RUNS[-1] / "container-replay-additional140-detail.json",
]
DETAIL_OUTPUT = ROOT / "harbor_vision/runs/matrix-transport-failure-audit-detail.json"
REPORT_OUTPUT = ROOT / "qualification/matrix-transport-failure-audit-report.json"


def iso_timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def wilson(successes, denominator, confidence=1 - 0.05 / 3):
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    rate = successes / denominator
    scale = 1 + z * z / denominator
    center = (rate + z * z / (2 * denominator)) / scale
    radius = z * (rate * (1 - rate) / denominator + z * z / (4 * denominator**2)) ** 0.5 / scale
    return [center - radius, center + radius]


def proxy_images(run):
    records = []
    malformed = []
    for path in sorted((run / "proxy").glob("request-*.json")):
        try:
            request = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            malformed.append({"file": path.name, "bytes": path.stat().st_size, "error": type(exc).__name__})
            continue
        for message in request.get("messages", []):
            content = message.get("content")
            for image in content.get("images", []) if isinstance(content, dict) else []:
                records.append({
                    "sha256": image["sha256"],
                    "bytes": image["bytes"],
                    "sequence": request["sequence"],
                    "timestamp": request["timestamp"],
                    "model": request.get("model"),
                })
    return records, malformed


def replay_rows():
    rows = {}
    for path in DETAIL_SOURCES:
        for row in json.loads(path.read_text())["rows"]:
            task_id = row["task_id"]
            if task_id in rows:
                raise RuntimeError(f"duplicate replay row: {task_id}")
            rows[task_id] = row
    return rows


def trial_rows():
    trials = {}
    proxy_diagnostics = {}
    for run in RUNS:
        requests, malformed = proxy_images(run)
        proxy_diagnostics[run.name] = {
            "valid_image_records": len(requests),
            "malformed_request_files": malformed,
        }
        for result_path in run.glob("jobs/*/*/result.json"):
            result = json.loads(result_path.read_text())
            task_id = result.get("task_name")
            if not task_id:
                continue
            if task_id in trials:
                raise RuntimeError(f"duplicate completed trial: {task_id}")
            trial = result_path.parent
            source = run / "tasks" / task_id / "environment/data/image.png"
            if not source.exists():
                raise RuntimeError(f"missing staged image: {source}")
            start = iso_timestamp(result["started_at"])
            end = iso_timestamp(result["finished_at"])
            window = [row for row in requests if start - 1 <= row["timestamp"] <= end + 1]
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            exact = [
                row for row in window
                if row["sha256"] == source_sha and row["model"] in {
                    "openai/deepseek-v4.1-flash", "deepseek-v4.1-flash"
                }
            ]
            answer_paths = list((trial / "artifacts/logs/artifacts").glob("answer.txt"))
            log_path = trial / "agent/openhands_sdk.txt"
            log = log_path.read_text(errors="replace") if log_path.exists() else ""
            trials[task_id] = {
                "task_id": task_id,
                "run": run.name,
                "trial": trial.name,
                "source_sha256": source_sha,
                "source_bytes": source.stat().st_size,
                "proxy_images_in_trial_window": len(window),
                "exact_original_request_sequences": sorted({row["sequence"] for row in exact}),
                "exact_original_transport_proven": bool(exact),
                "has_answer": bool(answer_paths and answer_paths[0].read_text(errors="replace").strip()),
                "result_exception_type": (result.get("exception_info") or {}).get("exception_type"),
                "agent_execution_recorded": bool(result.get("agent_execution")),
                "agent_result_recorded": bool(result.get("agent_result")),
                "trajectory_present": (trial / "agent/trajectory.json").exists(),
                "agent_log_present": log_path.exists(),
                "agent_log": log,
            }
    return trials, proxy_diagnostics


def missing_cause(row):
    log = row.pop("agent_log")
    exception = (row["result_exception_type"] or "").lower()
    lowered = log.lower()
    if not row["agent_execution_recorded"]:
        return "pre_agent_infrastructure_failure"
    if any(marker in lowered for marker in (
        "insufficient_quota", "budget exceeded", "exceeded your budget",
        "insufficient credits", "credit balance is too low"
    )):
        return "model_budget_error"
    if "rate limit" in lowered or "ratelimit" in lowered or "status code: 429" in lowered:
        return "model_rate_limit_error"
    if "timed out" in lowered or "timeout" in exception:
        return "agent_timeout"
    if "maximum" in lowered and "iterations" in lowered:
        return "max_iterations_without_answer"
    if "conversationerrorevent" in lowered or "llm" in lowered and "error" in lowered:
        return "agent_or_model_error_without_answer"
    if not row["trajectory_present"] or not row["agent_log_present"]:
        return "incomplete_agent_logs"
    return "completed_agent_without_answer"


def main():
    replays = replay_rows()
    trials, proxy_diagnostics = trial_rows()
    if len(replays) != 240 or set(replays) != set(trials):
        raise RuntimeError(f"population mismatch: replay={len(replays)}, trials={len(trials)}")
    source_counts = Counter(row["source_sha256"] for row in trials.values())
    rows = []
    for task_id in sorted(trials):
        row = trials[task_id]
        replay = replays[task_id]
        row["source_digest_unique_in_population"] = source_counts[row["source_sha256"]] == 1
        row["binary_reward"] = replay["binary_reward"]
        row["replay_status"] = replay["status"]
        row["outcome_class"] = (
            "full_credit" if replay["binary_reward"] == 1 else
            "retained_answer_not_full_credit" if row["has_answer"] else
            "missing_answer"
        )
        row["missing_answer_cause"] = missing_cause(row) if not row["has_answer"] else None
        rows.append(row)

    for row in rows:
        row["transport_class"] = (
            "exact_original_bytes_in_model_request" if row["exact_original_transport_proven"] else
            "proxy_image_present_but_source_transport_unproven" if row["proxy_images_in_trial_window"] else
            "no_proxy_image_in_trial_window"
        )
    transport = Counter(row["transport_class"] for row in rows)
    missing = Counter(row["missing_answer_cause"] for row in rows if row["missing_answer_cause"])
    outcome = Counter(row["outcome_class"] for row in rows)
    transport_by_outcome = {
        name: dict(sorted(Counter(row["transport_class"] for row in rows if row["outcome_class"] == name).items()))
        for name in sorted(outcome)
    }
    failed_transport_by_missing_cause = dict(sorted(Counter(
        row["missing_answer_cause"] for row in rows if not row["exact_original_transport_proven"]
    ).items()))
    successes = outcome["full_credit"]
    retained = successes + outcome["retained_answer_not_full_credit"]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    DETAIL_OUTPUT.write_text(json.dumps({"rows": rows}, indent=2) + "\n")
    report = {
        "version": 1,
        "protocol": "matrix-exact-outbound-image-transport-and-missing-outcome-audit-v1",
        "scheduled": len(rows),
        "transport_requirement": "Exact staged source image bytes must appear in a recorded outbound model request during that task trial. Staging and image tool invocation alone do not pass.",
        "solver_model": "openai/deepseek-v4.1-flash",
        "proxy_diagnostics": proxy_diagnostics,
        "transport_counts": dict(sorted(transport.items())),
        "transport_failures": len(rows) - transport["exact_original_bytes_in_model_request"],
        "transport_success_rate": transport["exact_original_bytes_in_model_request"] / len(rows),
        "transport_gate": "pass" if transport["exact_original_bytes_in_model_request"] == len(rows) and all(row["source_digest_unique_in_population"] for row in rows) else "fail",
        "transport_gate_rule": "All 240 scheduled image-bearing attempts require independently proven source-image transport; absent, malformed, unmatched, and unprovable cases fail conservatively.",
        "source_digest_unique_tasks": sum(row["source_digest_unique_in_population"] for row in rows),
        "transport_by_outcome": transport_by_outcome,
        "failed_transport_by_missing_cause": failed_transport_by_missing_cause,
        "outcome_counts": dict(sorted(outcome.items())),
        "missing_answer_cause_counts": dict(sorted(missing.items())),
        "primary_all_scheduled": {
            "successes": successes,
            "denominator": len(rows),
            "binary_accuracy": successes / len(rows),
            "simultaneous_wilson98_333": wilson(successes, len(rows)),
            "rule": "All missing answers and infrastructure outcomes remain zero."
        },
        "retained_answer_sensitivity": {
            "successes": successes,
            "denominator": retained,
            "binary_accuracy": successes / retained,
            "simultaneous_wilson98_333": wilson(successes, retained),
            "strict_band_supported": wilson(successes, retained)[0] > 0.10 and wilson(successes, retained)[1] < 0.70,
            "interpretation": "Sensitivity diagnostic only; never replaces the all-scheduled primary result. Its simultaneous interval checks whether the primary in-band conclusion depends on missing outcomes."
        },
        "qualification_interpretation": "The strict 10-70 accuracy gate is supportable only if exact image transport is proven across all scheduled attempts and missing causes do not make the retained-answer sensitivity exceed the 70% ceiling.",
        "detail_rows_count": len(rows),
        "detail_rows_sha256": hashlib.sha256(payload).hexdigest(),
        "detail_rows_policy": "Per-task transport sequences, hashes, trial names, and failure details remain untracked under harbor_vision/runs.",
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
