import ast
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path


def grade(record, submission):
    if submission is None:
        return {"reward": 0.0, "status": "missing"}
    match = re.fullmatch(r"\s*<answer>([^<>]+)</answer>\s*", submission)
    if not match:
        return {"reward": 0.0, "status": "malformed"}
    answer = match.group(1).strip()
    mode = record["eval_mode"]
    if mode == "range_verifier":
        low, high = (Decimal(str(x)) for x in ast.literal_eval(record["ideal"]))
        if not low.is_finite() or not high.is_finite() or low > high:
            raise ValueError("Invalid gold range")
        try:
            value = Decimal(answer)
            correct = value.is_finite() and low <= value <= high
        except InvalidOperation:
            correct = False
    elif mode == "str_verifier":
        correct = answer == record["ideal"].strip()
    else:
        return {"reward": 0.0, "status": "unsupported_judge_required"}
    return {"reward": float(correct), "status": "graded"}


def main():
    output = Path("/logs/verifier")
    output.mkdir(parents=True, exist_ok=True)
    try:
        record = json.loads(Path("/tests/gold.json").read_text())
        answer = Path("/workspace/answer.txt")
        result = grade(record, answer.read_text() if answer.exists() else None)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        SyntaxError,
        InvalidOperation,
    ) as exc:
        result = {
            "reward": 0.0,
            "status": "verifier_error",
            "error": f"{type(exc).__name__}: {exc}",
        }
    (output / "status.json").write_text(json.dumps(result))
    (output / "reward.json").write_text(json.dumps({"reward": result["reward"]}))


if __name__ == "__main__":
    main()
