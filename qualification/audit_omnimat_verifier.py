"""Deterministic independent FP/FN audit for OmniMatBench's pinned native CAL verifier."""
import json
import math
import sys
from decimal import Decimal
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omnimatbench.verify import grade, native


def wilson(k, n):
    z = NormalDist().inv_cdf(0.975)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - r), min(1, c + r)]


def image_rows():
    rows = []
    for source in sorted((ROOT / "omnimatbench/source/cal").rglob("*.jsonl")):
        category = source.relative_to(ROOT / "omnimatbench/source").parts[1]
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if row.get("image_url"):
                rows.append((f"cal/{category}/{row['id']}", row["final_answer_list"]))
    if len(rows) != 142:
        raise ValueError("Unexpected image CAL population")
    return rows


def run():
    rows = image_rows()
    gold_by_key = dict(rows)
    valid_cases = []
    invalid_cases = []
    for key, gold in rows:
        flat = native.flatten_answer(gold)
        valid_cases.append((key, "official_reference_exact", gold))
        for index, value in enumerate(flat):
            decimal = native.to_decimal(value)
            if decimal is not None:
                equivalent = format(decimal, "f")
                if equivalent != value:
                    changed = list(flat)
                    changed[index] = equivalent
                    if native.to_decimal(changed[index]) == decimal:
                        valid_cases.append((key, f"decimal_equivalent_slot_{index}", changed))
                wrong = decimal + abs(decimal) + Decimal(1)
                changed = list(flat)
                changed[index] = format(wrong, "f")
                if native.to_decimal(changed[index]) != decimal:
                    invalid_cases.append((key, f"numeric_wrong_slot_{index}", changed))
        invalid_cases.extend([
            (key, "missing_last_slot", flat[:-1]),
            (key, "extra_slot", flat + ["__wrong_extra_slot__"]),
            (key, "wrong_first_slot", ["__independently_wrong__"] + flat[1:]),
        ])

    false_negatives = []
    for key, perturbation, prediction in valid_cases:
        result = grade(json.dumps(prediction), gold_by_key[key])
        if result["reward"] != 1:
            false_negatives.append({"task": key, "perturbation": perturbation, "status": result["status"]})
    false_positives = []
    for key, perturbation, prediction in invalid_cases:
        result = grade(json.dumps(prediction), gold_by_key[key])
        if result["reward"] != 0:
            false_positives.append({"task": key, "perturbation": perturbation, "status": result["status"]})

    report = {
        "version": 1,
        "population": "All 142 released image-bearing CAL tasks",
        "independence": "Cases are generated from mathematical equality/inequality and structural list invariants, not solver outcomes. Official references are verifier-only.",
        "false_positive": {
            "errors": len(false_positives),
            "denominator": len(invalid_cases),
            "rate": len(false_positives) / len(invalid_cases),
            "wilson95": wilson(len(false_positives), len(invalid_cases)),
            "examples": false_positives,
        },
        "false_negative": {
            "errors": len(false_negatives),
            "denominator": len(valid_cases),
            "rate": len(false_negatives) / len(valid_cases),
            "wilson95": wilson(len(false_negatives), len(valid_cases)),
            "examples": false_negatives,
        },
        "valid_case_types": "Exact official answer plus exact Decimal-equivalent representations where a distinct form is constructible.",
        "invalid_case_types": "Missing slot, extra slot, definitely wrong first slot, plus each parseable numeric slot moved by abs(value)+1.",
    }
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
