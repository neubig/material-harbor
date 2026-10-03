from __future__ import annotations

import argparse
from decimal import DecimalException
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / 'native' if (ROOT / 'native').exists() else ROOT / 'source/scripts/cal'
sys.path.insert(0, str(NATIVE))
SPEC = importlib.util.spec_from_file_location('omnimat_native', NATIVE / 'eval_cal_results.py')
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


def valid(value):
    if isinstance(value, list):
        return bool(value) and all(valid(x) for x in value)
    if type(value) in (int, float):
        return math.isfinite(value)
    return isinstance(value, str) and bool(value.strip())


def canonicalize_exact_numeric_equivalents(prediction, gold):
    predicted = native.flatten_answer(prediction)
    expected = native.flatten_answer(gold)
    if len(predicted) != len(expected):
        return predicted
    for index, (value, reference) in enumerate(zip(predicted, expected)):
        value_decimal = native.to_decimal(value)
        reference_decimal = native.to_decimal(reference)
        if value_decimal is not None and reference_decimal is not None and value_decimal == reference_decimal:
            predicted[index] = reference
    return predicted


def _grade(raw, gold, canonicalize=False):
    try:
        prediction = json.loads(raw)
        if not isinstance(prediction, list) or not valid(prediction):
            return {'reward': 0, 'status': 'malformed'}
        if canonicalize:
            prediction = canonicalize_exact_numeric_equivalents(prediction, gold)
        scored = native.score_item({'id': 'case', 'llm_answer': prediction},
                                   {'case': native.flatten_answer(gold)},
                                   rel_tol=0.1, zero_tol=1e-12)
        return {'reward': scored['score_exact'], 'status': 'scored'}
    except (ValueError, TypeError, RecursionError, OverflowError):
        return {'reward': 0, 'status': 'malformed'}
    except DecimalException as exc:
        return {'reward': 0, 'status': 'native_exception', 'exception': type(exc).__name__}


def grade(raw, gold):
    return _grade(raw, gold)


def grade_adapter_v2(raw, gold):
    return _grade(raw, gold, canonicalize=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer', type=Path, default=Path('/app/answer.json'))
    parser.add_argument('--gold', type=Path, default=ROOT / 'gold.json')
    parser.add_argument('--reward', type=Path, default=Path('/logs/verifier/reward.txt'))
    args = parser.parse_args()
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text('0\n')
    gold = json.loads(args.gold.read_text())
    try:
        result = grade(args.answer.read_text(encoding='utf-8'), gold)
    except (OSError, UnicodeError):
        result = {'reward': 0, 'status': 'missing_or_unreadable'}
    args.reward.write_text(str(result['reward']) + '\n')
    (args.reward.parent / 'omnimatbench-status.json').write_text(json.dumps(result) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
