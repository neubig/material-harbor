import argparse
import json
import re
import unicodedata
from pathlib import Path


def disease_key(value: str) -> str:
    return " ".join(
        re.findall(
            r"\w+", unicodedata.normalize("NFKC", value).casefold().replace("_", " ")
        )
    )


def evaluate(prediction: object, expected: dict) -> dict:
    if expected["diseases"] is None:
        raise ValueError("Cannot grade an instance with missing disease gold")
    scores = {"reward": 0.0, "format": 0.0, "pathogenicity": 0.0, "diseases": 0.0}
    if not isinstance(prediction, dict) or set(prediction) != {
        "pathogenicity",
        "diseases",
    }:
        return scores
    label, diseases = prediction["pathogenicity"], prediction["diseases"]
    if not isinstance(label, str) or label not in {"benign", "pathogenic"}:
        return scores
    if not isinstance(diseases, list) or any(
        not isinstance(x, str) or not disease_key(x) for x in diseases
    ):
        return scores
    scores["format"] = 1.0
    scores["pathogenicity"] = float(label == expected["pathogenicity"])
    scores["diseases"] = float(
        {disease_key(x) for x in diseases}
        == {disease_key(x) for x in expected["diseases"]}
    )
    scores["reward"] = float(
        all(scores[key] == 1.0 for key in ("format", "pathogenicity", "diseases"))
    )
    return scores


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer-path", type=Path, default=Path("/app/answer.json"))
    parser.add_argument("--info-path", type=Path, default=Path("/tests/data/info.json"))
    parser.add_argument("--logs-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    expected = json.loads(args.info_path.read_text())["answer"]
    error = None
    try:
        prediction = json.loads(
            args.answer_path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
        )
    except (OSError, ValueError, UnicodeError) as exc:
        prediction, error = None, str(exc)
    scores = evaluate(prediction, expected)
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / "reward.json").write_text(json.dumps(scores))
    (args.logs_dir / "reward.txt").write_text(str(scores["reward"]))
    (args.logs_dir / "details.json").write_text(
        json.dumps(
            {
                "scores": scores,
                "prediction": prediction,
                "expected": expected,
                "error": error,
                "disease_matching": "normalized exact set; no inferred synonyms",
            }
        )
    )


if __name__ == "__main__":
    main()
