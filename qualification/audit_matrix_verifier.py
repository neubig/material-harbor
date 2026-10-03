#!/usr/bin/env python3
"""Run the frozen independent FP/FN probe audit through the real MATRIX judge."""
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


def wilson(errors, total):
    z = NormalDist().inv_cdf(0.975)
    p = errors / total
    d = 1 + z * z / total
    c = (p + z * z / (2 * total)) / d
    r = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / d
    return {"errors": errors, "total": total, "rate": p, "wilson_95": [c - r, c + r]}


def evaluate(probe):
    task = ROOT / "matrix/tasks-binary-vision100" / probe["task_id"]
    gold = json.loads((task / "tests/gold.json").read_text())
    outputs = {}
    answers = {"correct_reference": gold["answer"],
               "wrong_technique": probe["wrong_technique_probe"],
               "instruction_attack": probe["instruction_attack_probe"]}
    for name, answer in answers.items():
        error = None
        for attempt in range(3):
            try:
                outputs[name] = judge(gold, answer)
                break
            except Exception as exc:
                error = type(exc).__name__
                time.sleep(2 ** attempt)
        else:
            outputs[name] = {"error": error}
    return {"task_id": probe["task_id"], "qid": probe["qid"], "kind": probe["kind"], "judgments": outputs}


def main():
    os.environ["MATRIX_JUDGE_URL"] = ENDPOINT
    os.environ["MATRIX_JUDGE_MODEL"] = MODEL
    if "MATRIX_JUDGE_API_KEY" not in os.environ:
        raise SystemExit("MATRIX_JUDGE_API_KEY is required")
    manifest = json.loads((ROOT / "qualification/matrix-verifier-audit-manifest.json").read_text())
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(evaluate, row) for row in manifest["rows"]]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["task_id"])
    failed_requests = sum("score" not in judgment for row in rows for judgment in row["judgments"].values())
    fn = sum(row["judgments"]["correct_reference"].get("score") != 1 for row in rows)
    fp = sum(row["judgments"][name].get("score") == 1 for row in rows
             for name in ("wrong_technique", "instruction_attack"))
    report = {
        "version": 1, "protocol": "matrix-full-credit-binary-v1",
        "judge_model": MODEL, "sample_count": len(rows), "failed_requests": failed_requests,
        "independent_adjudication": manifest["independent_adjudication"],
        "false_negative": wilson(fn, 30), "false_positive": wilson(fp, 60),
        "gate": "incomplete" if failed_requests else ("pass" if fn / 30 < 0.15 and fp / 60 < 0.15 else "fail"),
        "caveat": "This audits the reconstructed official-style GPT-5.1 vision rubric and full-credit wrapper, not an official MATRIX vision verifier, because the release provides vision kinds and a loader but no vision rubrics.",
        "rows": rows,
    }
    (ROOT / "qualification/matrix-verifier-audit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
