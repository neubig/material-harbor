import argparse
import importlib.util
import json
from pathlib import Path

PROTOCOL = "matrix-full-credit-binary-v1"


def load_graded():
    spec = importlib.util.spec_from_file_location("matrix_graded", "/tests/verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def binaryize(result):
    if result["reward"] is not None:
        result["graded_reward"] = result["reward"]
        result["reward"] = int(result["graded_reward"] == 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", type=Path, default=Path("/logs/artifacts/answer.txt"))
    parser.add_argument("--gold", type=Path, default=Path("/tests/gold.json"))
    parser.add_argument("--logs", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    args.logs.mkdir(parents=True, exist_ok=True)
    (args.logs / "reward.txt").unlink(missing_ok=True)
    graded = load_graded()
    try:
        gold = json.loads(args.gold.read_text())
        result = binaryize(graded.evaluate(gold, args.answer))
    except Exception as exc:
        result = {"status": "infrastructure_error", "reward": None,
                  "error_type": type(exc).__name__}
    result["protocol"] = PROTOCOL
    result["fidelity"] = "Derived binary endpoint: exactly full rubric credit succeeds; every partial level fails. Paired graded_reward is retained."
    result["judge_model"] = graded.os.environ.get("MATRIX_JUDGE_MODEL")
    (args.logs / "result.json").write_text(json.dumps(result, indent=2))
    if result["reward"] is None:
        raise SystemExit(2)
    (args.logs / "reward.txt").write_text(str(result["reward"]))


if __name__ == "__main__":
    main()
