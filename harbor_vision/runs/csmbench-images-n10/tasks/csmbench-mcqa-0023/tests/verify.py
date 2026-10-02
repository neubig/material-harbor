"""Strict final-artifact verifier; labels are only staged by Harbor at verification."""
import argparse
import json
from pathlib import Path


def grade(path, correct):
    if correct not in ('A', 'B', 'C', 'D'):
        raise ValueError('Invalid trusted label')
    try:
        if path.is_symlink() or not path.is_file():
            return 0.0
        with path.open('rb') as stream:
            raw = stream.read(3)
        return float(raw in (correct.encode(), (correct + '\n').encode()))
    except OSError:
        return 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--answer', type=Path, default=Path('/app/answer.txt'))
    p.add_argument('--label', type=Path, default=Path('/tests/label.json'))
    p.add_argument('--reward', type=Path, default=Path('/logs/verifier/reward.txt'))
    args = p.parse_args()
    correct = json.loads(args.label.read_text())['correct_answer']
    score = grade(args.answer, correct)
    args.reward.parent.mkdir(parents=True, exist_ok=True)
    args.reward.write_text(str(score) + '\n')


if __name__ == '__main__':
    main()
