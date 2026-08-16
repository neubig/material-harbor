import json
import math
import os
import re
import urllib.request
from pathlib import Path


def extract_json(text):
    candidates = []
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.I):
        candidates.append(match.group(1))
    for start, char in enumerate(text):
        if char != "{":
            continue
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : end + 1])
                    break
    parsed = []
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        score = len(candidate) + (5000 if isinstance(value, dict) else 0)
        parsed.append((score, value))
    return max(parsed, default=(0, None))[1]


def number(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    return value


def count_leaves(value):
    if isinstance(value, dict):
        return sum(count_leaves(item) for item in value.values())
    if isinstance(value, list):
        return sum(count_leaves(item) for item in value)
    return 1


def compare(actual, expected, tolerance=0.05):
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return 0, count_leaves(expected)
        matched = 0
        total = count_leaves(expected)
        for key, value in expected.items():
            if key in actual:
                child_matched, _ = compare(actual[key], value, tolerance)
                matched += child_matched
        return matched, total
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return 0, count_leaves(expected)
        matched = total = 0
        for left, right in zip(actual, expected):
            child_matched, child_total = compare(left, right, tolerance)
            matched += child_matched
            total += child_total
        return matched, total
    actual_num, expected_num = number(actual), number(expected)
    if isinstance(actual_num, (int, float)) and isinstance(expected_num, (int, float)):
        return (1 if math.isclose(float(actual_num), float(expected_num), rel_tol=tolerance, abs_tol=tolerance) else 0), 1
    if isinstance(actual_num, str) and isinstance(expected_num, str):
        return (1 if actual_num.strip().lower() == expected_num.strip().lower() else 0), 1
    return (1 if actual_num == expected_num else 0), 1


case = json.loads(Path('/app/data/case.json').read_text(encoding='utf-8'))['case']
metadata = case.get('metadata', {})
answer_path = Path('/app/answer.txt')
actual_text = answer_path.read_text(encoding='utf-8').strip() if answer_path.exists() else ''
actual = extract_json(actual_text)

# SciAgentGYM's normal evaluator judges the top-level answer semantically.
expected_text = str(case.get('answer', '')).strip()
if actual is None:
    judge_reward = None
    api_key = os.environ.get('LLM_API_KEY')
    base_url = os.environ.get('LLM_BASE_URL', '').rstrip('/')
    judge_model = os.environ.get('LLM_JUDGE_MODEL', 'openai/deepseek-v4-flash')
    if api_key and base_url:
        prompt = f'''Judge whether the AI answer is correct.
Ignore formatting differences, irrelevant explanation, and prefixes. Accept logically,
computationally, or numerically equivalent answers. Return only CORRECT or INCORRECT.

Question:\n{case.get('question', '')}

Standard answer:\n{expected_text}

AI answer:\n{actual_text}'''
        payload = json.dumps({'model': judge_model, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 5}).encode()
        request = urllib.request.Request(
            f'{base_url}/v1/chat/completions',
            data=payload,
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)['choices'][0]['message']['content'].strip().upper()
            judge_reward = float(result == 'CORRECT')
        except Exception:
            judge_reward = None
    if judge_reward is not None:
        reward = judge_reward
    else:
        actual_text_lower = actual_text.lower()
        expected_text_lower = expected_text.lower()
        normalized_actual = re.sub(r"\\boxed\\s*\\{|[}${]", "", actual_text_lower)
        normalized_expected = re.sub(r"\\boxed\\s*\\{|[}${]", "", expected_text_lower)
        reward = float(bool(normalized_expected) and normalized_expected in normalized_actual)
else:
    # Refined-style responses are scored recursively against golden_answer.
    golden = metadata.get('golden_answer', expected_text)
    if isinstance(golden, list):
        golden = golden[0] if golden else None
    if isinstance(golden, dict) and 'final_answer' in golden:
        golden = golden['final_answer']
    if not isinstance(golden, dict):
        golden = {'answer': golden}
    matched, total = compare(actual, golden)
    reward = matched / total if total else 0.0

Path('/logs/verifier/reward.txt').write_text(str(reward))
if reward < 0.8:
    raise SystemExit(1)
