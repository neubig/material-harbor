from pathlib import Path

expected = Path('/tests/data/answer.txt').read_text(encoding='utf-8').strip()
actual_path = Path('/app/answer.txt')
actual = actual_path.read_text(encoding='utf-8').strip() if actual_path.exists() else ''
reward = float(bool(actual) and actual == expected)
Path('/logs/verifier/reward.txt').write_text(str(reward))
if reward < 1:
    raise SystemExit(1)
