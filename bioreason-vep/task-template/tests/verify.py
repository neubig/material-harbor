import json
import re
from pathlib import Path

expected = json.loads(Path("/tests/data/info.json").read_text())["answer"]
answer_path = Path("/app/answer.txt")
prediction = (
    answer_path.read_text(encoding="utf-8").strip().lower()
    if answer_path.exists()
    else ""
)
labels = re.findall(r"\b(?:benign|pathogenic)\b", prediction)
reward = float(bool(labels) and labels[-1] == expected)
Path("/logs/verifier/reward.txt").write_text(str(reward))
if reward < 1:
    raise SystemExit(1)
