import base64
import json
import os
import re
from pathlib import Path

import requests

info = json.loads(Path('/tests/data/info.json').read_text())
prediction = Path('/app/answer.txt').read_text(encoding='utf-8').strip() if Path('/app/answer.txt').exists() else ''
image = base64.b64encode(Path('/app/data/image.png').read_bytes()).decode()
model = os.environ.get('VLM_JUDGE_MODEL', 'gpt-5.5')
base = os.environ.get('VLM_JUDGE_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
key = os.environ.get('VLM_JUDGE_API_KEY') or os.environ.get('OPENAI_API_KEY') or os.environ.get('LLM_API_KEY')
if not key:
    raise RuntimeError('Set VLM_JUDGE_API_KEY, OPENAI_API_KEY, or LLM_API_KEY')
prompt = f'''Strictly judge this multimodal materials-science answer. Inspect the figure. The reference answer is evidence, not unquestionable truth. Score 1 only if the proposed answer substantively answers the question and is supported by the figure; otherwise score 0. Return only JSON {{"score":0 or 1,"rationale":"brief"}}.
Question: {info["question"]}
Reference answer: {info["answer"]}
Proposed answer: {prediction}'''
body = {'model': model, 'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image}}]}], 'max_tokens': 500, 'response_format': {'type': 'json_object'}}
response = requests.post(base + '/chat/completions', headers={'Authorization': 'Bearer ' + key}, json=body, timeout=240)
response.raise_for_status()
choices = response.json().get('choices', [])
if not choices:
    raise RuntimeError('judge returned no choices')
text = choices[0]['message'].get('content', '')
match = re.search(r'\{.*\}', text, re.S)
try:
    verdict = json.loads(match.group(0) if match else text)
except (TypeError, json.JSONDecodeError):
    verdict = {'score': 0, 'rationale': 'Judge returned malformed JSON.'}
score = int(verdict.get('score', 0))
Path('/logs/verifier/reward.txt').write_text(str(float(score)))
Path('/logs/verifier/details.json').write_text(json.dumps({'score': score, 'rationale': verdict.get('rationale', ''), 'model': model, 'prediction_available': bool(prediction)}))
if score != 1:
    raise SystemExit(1)
