import argparse
import json
from pathlib import Path


def grade(prediction, gold, options):
    return int(isinstance(prediction, str) and prediction.strip() in options
               and len(prediction.strip()) == 1 and prediction.strip() == gold)


def verify(answer, reference, reward):
    reward.parent.mkdir(parents=True, exist_ok=True)
    reward.write_text('0\n')
    info = json.loads(reference.read_text())
    try:
        prediction = answer.read_text() if answer.stat().st_size <= 128 else None
    except (OSError, UnicodeError):
        prediction = None
    score = grade(prediction, info['answer'], info['options'])
    reward.write_text(f'{score}\n')
    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer', type=Path, default=Path('/app/answer.txt'))
    parser.add_argument('--reference', type=Path, default=Path('/tests/reference.json'))
    parser.add_argument('--reward', type=Path, default=Path('/logs/verifier/reward.txt'))
    args = parser.parse_args()
    verify(args.answer, args.reference, args.reward)
