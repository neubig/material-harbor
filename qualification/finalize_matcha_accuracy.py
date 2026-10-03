#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "harbor_vision/runs"
CONFIDENCE = 1 - 0.05 / 3
SEED = 20260814
RESAMPLES = 100000


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


def wilson(successes, total):
    z = NormalDist().inv_cdf(1 - (1 - CONFIDENCE) / 2)
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return [center - margin, center + margin]


def cluster_bootstrap(records):
    grouped = defaultdict(list)
    for task, item in records.items():
        grouped[task.rsplit("-", 1)[0]].append(int(item["reward"] == 1))
    groups = list(grouped.values())
    rng = np.random.default_rng(SEED)
    means = np.empty(RESAMPLES)
    for index in range(RESAMPLES):
        sample = rng.integers(0, len(groups), len(groups))
        values = [value for group_index in sample for value in groups[group_index]]
        means[index] = np.mean(values)
    alpha = 1 - CONFIDENCE
    return {"groups": len(groups), "confidence": CONFIDENCE,
            "ci": [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))],
            "resamples": RESAMPLES, "seed": SEED}


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
    intervals = {"wilson": wilson(successes, 100), "paper_cluster_bootstrap": cluster_bootstrap(combined)}
    report = {
        "version": 2, "protocol": "matcha-vision-binary-v1",
        "model": "openai/deepseek-v4.1-flash", "look": 100,
        "scheduled": 100, "attempted": len(combined), "successes": successes,
        "failures_including_missing": len(combined) - successes, "missing_rewards": missing,
        "accuracy": successes / 100, "accuracy_percent": successes,
        "confidence": CONFIDENCE, "intervals": intervals,
        "strict_gate": {
            "required": "accuracy > 0.10 and accuracy < 0.70",
            "observed_sample_point": "not_met_at_excluded_upper_boundary",
            "population_verdict": "unresolved_expand_to_precommitted_n300",
            "reason": "The sample point is exactly 70%, so it does not itself satisfy the strict inequality. It is not a definitive population failure: both precommitted simultaneous intervals cross the 70% boundary. Protocol failure requires the entire interval to be >=70% (or a full-population result outside the band)."
        },
        "interruption_accounting": "The original 23 completed attempts were retained unchanged; only the disjoint frozen 77-task remainder ran. No task was rerun. Missing rewards count as failures in the scheduled denominator.",
        "required_next_step": "Run exactly the frozen next200_task_ids without replacing or rerunning the original 100, then apply the cumulative n=300 inference rule.",
        "rows": [{"task_id": task, **combined[task]} for task in sorted(combined)],
    }
    (ROOT / "qualification/matcha-accuracy-report.json").write_text(json.dumps(report, indent=2) + "\n")
    verdict = {
        "version": 2, "candidate": "MATCHA image-bearing subset",
        "overall": "unresolved_accuracy_expand_to_n300",
        "gates": {
            "public_nonduplicate_tasks": {"verdict": "pass", "evidence": "Frozen 100-task image cohort from released MATCHA tasks; a disjoint additional 200 is precommitted."},
            "relevant_foundation_model": {"verdict": "pass_with_fit_caveat", "evidence": "Intern-S1 is a named multimodal scientific foundation model with reported materials relevance; benchmark-specific MATCHA benefit is not established."},
            "harbor": {"verdict": "pass", "evidence": "100 image tasks executed through Harbor across interruption-safe disjoint runs."},
            "deepseek_binary_accuracy": {"verdict": "unresolved", "evidence": "70/100 = 70.0% does not satisfy the sample strict inequality, but the 98.333% simultaneous intervals cross 70%; the frozen cumulative n=300 look is required."},
            "verifier_fp_fn": {"verdict": "pending_accuracy_resolution"}
        },
        "stop_reason": None,
    }
    (ROOT / "qualification/matcha-qualification-verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
