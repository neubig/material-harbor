import json
from pathlib import Path

info = json.loads(Path('/tests/data/info.json').read_text())
answer_path = Path('/app/answer.txt')
reward = 0.0
if answer_path.exists():
    prediction = answer_path.read_text(encoding='utf-8').strip()
    reward = float(bool(prediction)) if info['qa_type'] == 'subjective' else float(prediction.upper() == info['answer'].upper())
Path('/logs/verifier/reward.txt').write_text(str(reward))
if reward < 1:
    raise SystemExit(1)
