#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omnimatbench.build_cal_vision import load_rows
from omnimatbench.verify import grade, grade_adapter_v2

AUDITED_IDS = [
    "cal/18/019", "cal/08/024", "cal/18/032", "cal/19/003", "cal/13/007",
    "cal/14/015", "cal/07/012", "cal/13/009", "cal/01/011", "cal/05/018",
    "cal/06/012", "cal/06/003", "cal/05/016", "cal/02/025", "cal/01/017",
]


def wilson(errors, total):
    z = NormalDist().inv_cdf(0.975)
    p = errors / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return {"errors": errors, "total": total, "rate": p,
            "wilson_95": [center - margin, center + margin]}


def wrong_shape(value):
    if isinstance(value, list):
        return [wrong_shape(item) for item in value]
    return "__independently_wrong__"


def main():
    manifest = json.loads((ROOT / "qualification/omnimat-scientific-audit-manifest.json").read_text())
    blind = json.loads((ROOT / "qualification/omnimat-scientific-audit-blind-derivations.json").read_text())
    derivations = {row["task_id"]: row for row in blind["rows"]}
    released = {row["key"]: row["row"]["final_answer_list"] for row in load_rows()}
    if not set(AUDITED_IDS) <= set(manifest["task_ids"]):
        raise ValueError("Audit IDs escaped frozen manifest")
    rows = []
    for task_id in AUDITED_IDS:
        correct = derivations[task_id]["derived_answer"]
        gold = released[task_id]
        correct_raw = json.dumps(correct)
        wrong_raw = json.dumps(wrong_shape(correct))
        rows.append({
            "task_id": task_id,
            "blind_basis": derivations[task_id]["basis"],
            "independently_derived_answer": correct,
            "released_reference": gold,
            "native_v1_correct_probe_reward": grade(correct_raw, gold)["reward"],
            "adapter_v2_correct_probe_reward": grade_adapter_v2(correct_raw, gold)["reward"],
            "native_v1_wrong_probe_reward": grade(wrong_raw, gold)["reward"],
            "adapter_v2_wrong_probe_reward": grade_adapter_v2(wrong_raw, gold)["reward"],
        })
    report = {
        "version": 1,
        "manifest": "omnimat-scientific-audit-manifest.json",
        "blind_derivations": "omnimat-scientific-audit-blind-derivations.json",
        "selection": "15 high-confidence transparent calculations/direct source readings within the precommitted 30; 14 unresolved/insufficient-input items and one explicitly source-unanswerable item are not forced.",
        "independence": "Derived answers and methods were committed before released final_answer_list values were opened. No solver outputs or judge/model votes are used as truth.",
        "native_v1": {
            "false_positive": wilson(sum(row["native_v1_wrong_probe_reward"] == 1 for row in rows), len(rows)),
            "false_negative": wilson(sum(row["native_v1_correct_probe_reward"] == 0 for row in rows), len(rows)),
        },
        "adapter_v2": {
            "false_positive": wilson(sum(row["adapter_v2_wrong_probe_reward"] == 1 for row in rows), len(rows)),
            "false_negative": wilson(sum(row["adapter_v2_correct_probe_reward"] == 0 for row in rows), len(rows)),
        },
        "gate": "fail_false_negative_rate",
        "gate_reason": "Both comparator versions reject 6/15 independently derived scientifically correct answers (40.0%; Wilson 95% lower bound 19.8%), conclusively above the strict 15% ceiling. Numeric representation canonicalization does not repair qualitative equivalence, scientifically equivalent scenario labels, or source-label calculation errors.",
        "rows": rows,
    }
    path = ROOT / "qualification/omnimat-scientific-label-audit-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
