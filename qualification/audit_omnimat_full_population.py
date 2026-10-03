#!/usr/bin/env python3
"""Full-population task-level native OmniMat verifier audit."""
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omnimatbench.build_cal_vision import load_rows
from omnimatbench.verify import grade, native

ENDPOINT = "https://llm-proxy.app.all-hands.dev/v1/chat/completions"
MODELS = ("openai/gpt-5.1", "claude-sonnet-4-6")


def ask(model, candidate, gold):
    prompt = f"""Independently adjudicate a narrow algebraic-equivalence question. Do not judge style or exact text. Determine whether the alternate expression is mathematically identical to the released reference for all values where the reference is defined. The alternate must preserve the requested equation/value and may differ only by algebraic rearrangement or notation.

Task: {candidate['task_id']}
Reference: {gold}
Alternate: {candidate['alternate']}
Claimed identity: {candidate['identity']}

Return only JSON: {{"equivalent": true|false, "confidence": "high"|"medium"|"low", "reason": "concise proof or counterexample"}}"""
    response = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {os.environ['LLM_API_KEY']}", "Content-Type": "application/json"},
                             json={"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": 5000}, timeout=300)
    response.raise_for_status()
    text = response.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON from {model}: {text[:200]}")
    value = json.loads(match.group())
    return {"model": model, "equivalent": value.get("equivalent") is True,
            "confidence": value.get("confidence"), "reason": value.get("reason")}


def main():
    protocol_path = ROOT / "qualification/omnimat-full-population-fn-audit-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    released = {row["key"]: row["row"]["final_answer_list"] for row in load_rows()}
    candidates = protocol["symbolic_candidates"]
    adjudications = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {}
        for candidate in candidates:
            flat = native.flatten_answer(released[candidate["task_id"]])
            for model in MODELS:
                futures[pool.submit(ask, model, candidate, flat[candidate["slot"]])] = (candidate["task_id"], model)
        for future in as_completed(futures):
            task_id, model = futures[future]
            try:
                value = future.result()
            except Exception as exc:
                value = {"model": model, "equivalent": False, "confidence": "error", "reason": type(exc).__name__}
            adjudications.setdefault(task_id, []).append(value)

    symbolic = {}
    for candidate in candidates:
        task_id = candidate["task_id"]
        judges = sorted(adjudications[task_id], key=lambda row: row["model"])
        valid = len(judges) == 2 and all(row["equivalent"] and row["confidence"] == "high" for row in judges)
        flat = native.flatten_answer(released[task_id])
        reference = flat[candidate["slot"]]
        alternate = list(flat)
        alternate[candidate["slot"]] = candidate["alternate"]
        reward = grade(json.dumps(alternate), released[task_id])["reward"] if valid else None
        symbolic[task_id] = {"slot": candidate["slot"], "reference": reference, "alternate": candidate["alternate"],
                             "identity": candidate["identity"], "judges": judges, "consensus_high_equivalent": valid,
                             "native_reward": reward}

    task_rows = []
    for task_id, gold in sorted(released.items()):
        flat = native.flatten_answer(gold)
        numeric_probes = []
        for index, value in enumerate(flat):
            decimal = native.to_decimal(value)
            if decimal is None:
                continue
            equivalent = format(decimal, "f")
            if equivalent != value and native.to_decimal(equivalent) == decimal:
                mutated = list(flat)
                mutated[index] = equivalent
                reward = grade(json.dumps(mutated), gold)["reward"]
                numeric_probes.append({"slot": index, "reference": value, "alternate": equivalent,
                                       "exact_decimal": str(decimal), "decimal_equality_asserted": True,
                                       "native_reward": reward})
        valid_rejected = [probe for probe in numeric_probes if probe["native_reward"] == 0]
        symbolic_probe = symbolic.get(task_id)
        symbolic_rejected = bool(symbolic_probe and symbolic_probe["consensus_high_equivalent"] and symbolic_probe["native_reward"] == 0)
        wrong = list(flat)
        first_decimal = native.to_decimal(wrong[0]) if wrong else None
        if first_decimal is not None:
            wrong[0] = format(first_decimal + abs(first_decimal) + Decimal(1), "f")
            wrong_proof = "first numeric value differs by abs(reference)+1"
        else:
            wrong[0] = "__independently_wrong__"
            wrong_proof = "first source answer replaced by an unrelated sentinel"
        fp_reward = grade(json.dumps(wrong), gold)["reward"]
        task_rows.append({"task_id": task_id, "numeric_equivalent_probes": numeric_probes,
                          "symbolic_probe": symbolic_probe, "false_negative_task": bool(valid_rejected or symbolic_rejected),
                          "fn_basis": "exact_decimal_equivalence" if valid_rejected else ("cross_provider_high_confidence_algebraic_identity" if symbolic_rejected else None),
                          "invalid_probe_proof": wrong_proof, "invalid_probe_native_reward": fp_reward,
                          "false_positive_task": fp_reward == 1})

    fn_tasks = [row["task_id"] for row in task_rows if row["false_negative_task"]]
    fp_tasks = [row["task_id"] for row in task_rows if row["false_positive_task"]]
    report = {"version": 1, "verifier": "native-v1 frozen headline", "population": 142,
              "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
              "unit": "distinct released task; at most one FN and one FP per task",
              "false_negative": {"errors": len(fn_tasks), "denominator": 142, "rate": len(fn_tasks) / 142,
                                 "finite_population_gate": "fail" if len(fn_tasks) >= 22 else "unresolved_or_pass",
                                 "threshold_proof": "22/142 > 15%; all 142 tasks audited, no sample extrapolation", "task_ids": fn_tasks},
              "false_positive": {"errors": len(fp_tasks), "denominator": 142, "rate": len(fp_tasks) / 142,
                                 "audit_result": "pass_observed_adversarial_population" if len(fp_tasks) / 142 < 0.15 else "fail",
                                 "scope": "one mechanically definitely wrong arity-preserving adversarial response per released image task", "task_ids": fp_tasks},
              "preliminary_selected_scientific_audit": "Retained separately but not used in these rates.",
              "source_unanswerable_accounting": "No task is excluded. A source-unanswerable task contributes to the 142 denominator and is not labeled FN absent a proven equivalent response.",
              "rows": task_rows}
    (ROOT / "qualification/omnimat-full-population-fn-audit-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
