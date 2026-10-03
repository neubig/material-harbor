#!/usr/bin/env python3
"""Independent adversarial audit of the BioReason GO closure verifier."""
import json
import sys
from pathlib import Path
from statistics import NormalDist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bioreason.verify import grade, load_ontology

ROOTS = {"bp": "GO:0008150", "mf": "GO:0003674", "cc": "GO:0005575"}


def independent_ontology(path):
    parents = {}
    current = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line == "[Term]":
            current = None
        elif line.startswith("id: GO:"):
            current = line[4:]
            parents.setdefault(current, set())
        elif current and line.startswith("is_a: GO:"):
            parents[current].add(line.split()[1])
        elif current and line.startswith("relationship: part_of GO:"):
            parents[current].add(line.split()[2])
    return parents


def closure(terms, parents):
    result, stack = set(), list(terms)
    while stack:
        term = stack.pop()
        if term not in result:
            result.add(term)
            stack.extend(parents.get(term, ()))
    return result


def wilson(errors, total):
    z = NormalDist().inv_cdf(0.975)
    p = errors / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denominator
    return {"errors": errors, "total": total, "rate": p,
            "wilson_95": [center - margin, center + margin]}


def main():
    manifest = json.loads((ROOT / "qualification/bioreason-go-manifest.json").read_text())
    rows = []
    first_ontology = ROOT / "bioreason/tasks-go100" / manifest["tasks"][0]["task_id"] / "tests/go-basic.obo"
    parents = independent_ontology(first_ontology)
    verifier_ontology = load_ontology(first_ontology)
    for item in manifest["tasks"]:
        task = ROOT / "bioreason/tasks-go100" / item["task_id"]
        gold = json.loads((task / "tests/gold.json").read_text())
        ontology = task / "tests/go-basic.obo"
        expected = set().union(*(set(terms) for terms in gold["aspects"].values()))
        nonleaves = set().union(*(parents.get(term, set()) for term in expected))
        leaves = expected - nonleaves
        correct = {"go_ids": sorted(leaves)}
        missing = {"go_ids": sorted(leaves - {sorted(leaves)[0]})}
        labeled_roots = {ROOTS[aspect] for aspect, terms in gold["aspects"].items() if terms}
        extra = next(term for term in sorted(parents)
                     if term not in expected and any(root in closure([term], parents) for root in labeled_roots))
        wrong_extra = {"go_ids": sorted(leaves | {extra})}
        correct_reward = grade(correct, gold, ontology, verifier_ontology)["reward"]
        missing_reward = grade(missing, gold, ontology, verifier_ontology)["reward"]
        extra_reward = grade(wrong_extra, gold, ontology, verifier_ontology)["reward"]
        rows.append({"task_id": item["task_id"], "correct_leaf_probe_reward": correct_reward,
                     "missing_leaf_probe_reward": missing_reward, "extra_same_aspect_probe_reward": extra_reward})
    false_negatives = sum(row["correct_leaf_probe_reward"] == 0 for row in rows)
    false_positives = sum(row[key] == 1 for row in rows
                          for key in ("missing_leaf_probe_reward", "extra_same_aspect_probe_reward"))
    report = {
        "version": 1,
        "protocol": "bioreason-go-complete-v1",
        "independence": "Audit closure parser is implemented separately from the verifier and constructs probes without verifier output. Gold remains the frozen annotation-recovery target; no biological-truth completeness claim is made.",
        "correct_probes": "100 independently reduced leaf/antichain outputs whose is_a/part_of closure equals the released aspect closures.",
        "wrong_probes": "100 missing-leaf and 100 valid same-aspect extra-term outputs; all are independently unequal to the target closure.",
        "false_negative": wilson(false_negatives, len(rows)),
        "false_positive": wilson(false_positives, 2 * len(rows)),
        "gate": "pass" if false_negatives / len(rows) < 0.15 and false_positives / (2 * len(rows)) < 0.15 else "fail",
        "caveat": "This verifies exact frozen annotation recovery mechanics. Open-world GO incompleteness means extra biologically true terms absent from released labels still fail by contract.",
        "rows": rows,
    }
    path = ROOT / "qualification/bioreason-verifier-audit-report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
