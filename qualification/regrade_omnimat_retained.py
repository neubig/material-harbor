#!/usr/bin/env python3
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omnimatbench import verify

RUN = ROOT / "harbor_vision/runs/omnimat-cal-vision-qualification-n100"


def arrays(text):
    found = []
    decoder = json.JSONDecoder()
    offset = 0
    while match := re.search(r"\[", text[offset:]):
        start = offset + match.start()
        try:
            value, length = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            offset = start + 1
            continue
        if isinstance(value, list):
            found.append(value)
        offset = start + length
    return found


def extract_answer(trajectory):
    candidates = []
    for step in trajectory.get("steps", []):
        for call in step.get("tool_calls") or []:
            arguments = call.get("arguments") or {}
            if str(arguments.get("path", "")).endswith("answer.json"):
                written = arrays(str(arguments.get("file_text", "")))
                if written:
                    candidates.append(written[0])
                continue
            command = str(arguments.get("command", ""))
            writes_answer = "answer.json" in command and (
                re.search(r">+\s*(?:/app/)?answer\.json", command)
                or re.search(r"open\([^\n]*answer\.json[^\n]*[,'\"]w", command)
                or re.search(r"json\.dump\([^\n]*answer", command)
            )
            if not writes_answer:
                continue
            observed = []
            for result in (step.get("observation") or {}).get("results", []):
                content = str(result.get("content", ""))
                actual_output = content.rsplit("\x1b[?2004l", 1)[-1]
                observed.extend(arrays(actual_output))
            written = observed or arrays(command)
            if written:
                candidates.append(written[0])
    return candidates[-1] if candidates else None


def native_grade(raw, gold):
    try:
        prediction = json.loads(raw)
        if not isinstance(prediction, list) or not verify.valid(prediction):
            return 0
        scored = verify.native.score_item(
            {"id": "case", "llm_answer": prediction},
            {"case": verify.native.flatten_answer(gold)},
            rel_tol=0.1,
            zero_tol=1e-12,
        )
        return scored["score_exact"]
    except Exception:
        return 0


def main():
    run_report = json.loads((RUN / "report.json").read_text())
    job = next(path for path in (RUN / "jobs").iterdir() if path.is_dir())
    rows = []
    for trial in sorted(path for path in job.iterdir() if path.is_dir()):
        task_id = trial.name.split("__", 1)[0]
        trajectory_path = trial / "agent/trajectory.json"
        answer = extract_answer(json.loads(trajectory_path.read_text())) if trajectory_path.exists() else None
        gold = json.loads((RUN / "tasks" / task_id / "tests/gold.json").read_text())
        raw = json.dumps(answer) if answer is not None else ""
        native_reward = native_grade(raw, gold)
        adapter_v2_reward = verify.grade_adapter_v2(raw, gold)["reward"] if answer is not None else 0
        harbor_row = next(value for value in run_report["rewards"].values()
                          if value["task_id"].endswith("/" + task_id))
        rows.append({
            "task_id": task_id,
            "answer_artifact_retained": answer is not None,
            "native_v1": native_reward,
            "adapter_v2": adapter_v2_reward,
            "harbor_native_v1": harbor_row["reward"],
        })
    if len(rows) != 100:
        raise RuntimeError(f"expected 100 retained attempts, found {len(rows)}")
    mismatches = [row["task_id"] for row in rows if row["native_v1"] != row["harbor_native_v1"]]
    transitions = [row["task_id"] for row in rows if row["native_v1"] != row["adapter_v2"]]
    report = {
        "version": 2,
        "cohort": "omnimat-cal-vision-qualification-n100",
        "attempts": len(rows),
        "answer_artifacts_retained": sum(row["answer_artifact_retained"] for row in rows),
        "missing_answer_artifacts_counted_zero": sum(not row["answer_artifact_retained"] for row in rows),
        "native_v1": {
            "correct": sum(row["native_v1"] == 1 for row in rows),
            "incorrect": sum(row["native_v1"] != 1 for row in rows),
            "matches_original_harbor_rewards": not mismatches,
            "mismatched_task_ids": mismatches,
        },
        "adapter_v2": {
            "correct": sum(row["adapter_v2"] == 1 for row in rows),
            "incorrect": sum(row["adapter_v2"] != 1 for row in rows),
            "changed_task_ids": transitions,
        },
        "scope": "All retained attempts were regraded. No task was rerun and no instruction changed. Missing answer artifacts remain zero under both versions.",
        "headline": "Native-v1 remains the precommitted accuracy. Adapter-v2 is a post-cohort secondary analysis and is not official-exact.",
    }
    output = ROOT / "qualification/omnimat-retained-dual-regrade-report.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
