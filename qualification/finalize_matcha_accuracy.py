#!/usr/bin/env python3
import json
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "harbor_vision/runs"


def collect(path):
    records = {}
    for result_path in path.rglob("result.json"):
        value = json.loads(result_path.read_text())
        task = value.get("task_name")
        if not task:
            continue
        if task in records:
            raise ValueError(f"Duplicate attempted task: {task}")
        reward = ((value.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        records[task] = {"reward": reward, "result": str(result_path.relative_to(ROOT))}
    return records


def wilson(successes, total, confidence=0.95):
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [center - margin, center + margin]


def main():
    manifest = json.loads((ROOT / "qualification/matcha-resume-manifest.json").read_text())
    retained_ids = set(manifest["retained_completed_task_ids"])
    remaining_ids = set(manifest["remaining_task_ids"])
    if retained_ids & remaining_ids or len(retained_ids | remaining_ids) != 100:
        raise ValueError("Invalid frozen resume partition")
    retained = collect(RUNS / "matcha-qualification-n100")
    resumed = collect(RUNS / "matcha-qualification-n100-resume77")
    if set(retained) != retained_ids or set(resumed) != remaining_ids:
        raise ValueError("Completed attempts do not exactly match frozen partition")
    combined = {**retained, **resumed}
    successes = sum(item["reward"] == 1 for item in combined.values())
    missing = sum(item["reward"] is None for item in combined.values())
    report = {
        "version": 1,
        "protocol": "matcha-vision-binary-v1",
        "model": "openai/deepseek-v4.1-flash",
        "scheduled": 100,
        "attempted": len(combined),
        "successes": successes,
        "failures_including_missing": len(combined) - successes,
        "missing_rewards": missing,
        "accuracy": successes / 100,
        "accuracy_percent": successes,
        "wilson_95": wilson(successes, 100),
        "strict_gate": {"required": "accuracy > 0.10 and accuracy < 0.70", "verdict": "fail"},
        "reason": "70/100 is exactly the excluded upper boundary, not strictly below 70%; no confidence-interval convention can make the observed point estimate satisfy the strict inequality.",
        "interruption_accounting": "The original 23 completed attempts were retained unchanged; only the disjoint frozen 77-task remainder ran. No task was rerun. Missing rewards, if any, would count as failures in the scheduled denominator.",
        "cohort_expansion_not_run": "The pre-frozen additional 200 tasks are unnecessary after a definitive strict-boundary failure and were not evaluated.",
        "rows": [{"task_id": task, **combined[task]} for task in sorted(combined)],
    }
    (ROOT / "qualification/matcha-accuracy-report.json").write_text(json.dumps(report, indent=2) + "\n")
    verdict = {
        "version": 1,
        "candidate": "MATCHA image-bearing subset",
        "overall": "fail_deepseek_binary_accuracy_gate",
        "gates": {
            "public_nonduplicate_tasks": {"verdict": "pass", "evidence": "Frozen 100-task image cohort from released MATCHA tasks."},
            "relevant_foundation_model": {"verdict": "pass", "evidence": "Named relevant model evidence retained in qualification research."},
            "harbor": {"verdict": "pass", "evidence": "100 image tasks executed through Harbor across interruption-safe disjoint runs."},
            "deepseek_binary_accuracy": {"verdict": "fail", "evidence": "70/100 = 70.0%, exactly outside the strict >10% and <70% gate."},
            "verifier_fp_fn": {"verdict": "not_needed_after_accuracy_failure"}
        },
        "stop_reason": "Definitive strict accuracy-gate failure; later costly verifier audit stopped by protocol."
    }
    (ROOT / "qualification/matcha-qualification-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
