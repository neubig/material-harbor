import json
import os
import re
import urllib.request
from pathlib import Path


def extract_balanced_braces(text, start_pos):
    if not text or start_pos < 0 or start_pos >= len(text) or text[start_pos] != '{':
        return None
    stack = 0
    result = []
    for char in text[start_pos:]:
        if char == '{':
            stack += 1
            if stack > 1:
                result.append(char)
        elif char == '}':
            stack -= 1
            if stack == 0:
                return ''.join(result)
            if stack < 0:
                return None
            result.append(char)
        elif stack >= 1:
            result.append(char)
    return None


def extract_boxed_answer(text):
    if not text:
        return None
    matches = []
    for pattern in (r"\\boxed\{", r"\$\\boxed\{", r"boxed\{"):
        for match in re.finditer(pattern, text):
            value = extract_balanced_braces(text, match.end() - 1)
            if value is not None:
                matches.append(value)
    return matches[-1].strip() if matches else None


case = json.loads(Path('/app/data/case.json').read_text(encoding='utf-8'))['case']
answer_path = Path('/app/answer.txt')
response = answer_path.read_text(encoding='utf-8').strip() if answer_path.exists() else ''
model_answer = extract_boxed_answer(response)
standard_answer = case.get('answer')
if isinstance(standard_answer, list) and standard_answer:
    standard_answer = str(standard_answer[0])
elif isinstance(standard_answer, dict):
    standard_answer = json.dumps(standard_answer, ensure_ascii=False)
else:
    standard_answer = str(standard_answer or '')

reward = 0.0
api_key = os.environ.get('LLM_API_KEY')
base_url = os.environ.get('LLM_BASE_URL', '').rstrip('/')
if model_answer and standard_answer and api_key and base_url:
    prompt = f'''这是一个问题、一个标准答案，以及一个由AI模型生成的答案。请你判断AI模型的答案是否正确。
    评判要求：
    1. 忽略答案中的格式差异。
    2. 忽略无关的前缀或文字修饰。
    3. 如果标准答案是一个表达式或数值，只要模型答案在逻辑、计算或数值上等价，也视为正确。
    4. 若模型答案与标准答案**核心内容一致**，则判断为"正确"。

    请只输出：`正确` 或 `错误`。

    ---
    问题：
    {case.get('question', '')}
    ---
    标准答案：
    {standard_answer}
    ---
    AI模型的答案：
    {model_answer}
    ---
    '''
    payload = json.dumps({'model': os.environ.get('LLM_JUDGE_MODEL', 'openai/deepseek-v4-flash'), 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': 5}).encode()
    request = urllib.request.Request(f'{base_url}/v1/chat/completions', data=payload, headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as result:
            content = json.load(result)['choices'][0]['message']['content'].strip()
        reward = float(content == '正确')
    except Exception:
        reward = 0.0

Path('/logs/verifier/reward.txt').write_text(str(reward))
if reward < 1:
    raise SystemExit(1)
