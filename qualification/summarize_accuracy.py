"""Summarize precommitted qualification looks without dropping missing attempts."""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIDENCE = 1 - 0.05 / 3
SEED = 20260814
RESAMPLES = 100000


def wilson(k, n):
    z = NormalDist().inv_cdf(1 - (1 - CONFIDENCE) / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - r), min(1, c + r)]


def clusters(candidate, task_ids):
    if candidate == "csmbench":
        manifest = json.loads((ROOT / "csmbench/manifest.json").read_text())
        groups = {f"csmbench-mcqa-{row['index']:04d}": row["paper_folder_name"] for row in manifest["rows"]}
        return [groups[task_id] for task_id in task_ids]
    if candidate == "omnimat":
        lookup = {}
        for source in (ROOT / "omnimatbench/source/cal").rglob("*.jsonl"):
            category = source.relative_to(ROOT / "omnimatbench/source").parts[1]
            for line in source.read_text().splitlines():
                row = json.loads(line)
                if row.get("image_url"):
                    lookup[f"omnimatbench-cal-{category}-{row['id']}"] = row["source_name"]
        return [lookup[task_id] for task_id in task_ids]
    raise ValueError(candidate)


def cluster_bootstrap(values, group_ids):
    grouped = defaultdict(list)
    for value, group in zip(values, group_ids):
        grouped[group].append(value)
    groups = list(grouped.values())
    rng = np.random.default_rng(SEED)
    means = np.empty(RESAMPLES)
    for i in range(RESAMPLES):
        sampled = rng.integers(0, len(groups), len(groups))
        drawn = [value for index in sampled for value in groups[index]]
        means[i] = np.mean(drawn)
    alpha = 1 - CONFIDENCE
    return {
        "groups": len(groups),
        "ci": [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))],
        "resamples": RESAMPLES,
        "seed": SEED,
    }


def summarize(candidate, report_path):
    report = json.loads(report_path.read_text())
    expected = report["staged_tasks"]
    rows = []
    for value in report["rewards"].values():
        task_id = Path(value["task_id"]).name
        reward = value.get("reward")
        rows.append((task_id, reward if reward in (0, 1) else 0))
    task_ids = [task_id for task_id, _ in rows]
    values = [value for _, value in rows]
    complete = len(rows) == expected
    output = {
        "version": 1,
        "candidate": candidate,
        "model": report["model"],
        "metric": "binary_accuracy_all_attempted",
        "confidence": CONFIDENCE,
        "expected": expected,
        "attempted": len(rows),
        "unattempted": expected - len(rows),
        "missing_rewards_counted_zero": sum(value.get("reward") not in (0, 1) for value in report["rewards"].values()),
        "successes": sum(values),
        "point_estimate": sum(values) / expected if complete else None,
        "wilson": wilson(sum(values), expected) if complete else None,
        "cluster_bootstrap": cluster_bootstrap(values, clusters(candidate, task_ids)) if complete else None,
        "transport": report["transport"],
        "complete": complete,
    }
    if not complete:
        output["classification"] = "incomplete"
    else:
        intervals = [output["wilson"], output["cluster_bootstrap"]["ci"]]
        if all(low > 0.10 and high < 0.70 for low, high in intervals):
            output["classification"] = "pass_strict_band"
        elif all(high <= 0.10 for _, high in intervals) or all(low >= 0.70 for low, _ in intervals):
            output["classification"] = "fail_outside_band"
        else:
            output["classification"] = "inconclusive_expand_if_population_allows"
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("csmbench", "omnimat"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.candidate, args.report)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
