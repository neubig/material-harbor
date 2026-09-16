import argparse
import json
import os
import stat
from pathlib import Path


def unique_object(pairs):
    if len(dict(pairs)) != len(pairs):
        raise ValueError('Duplicate JSON key')
    return dict(pairs)


def evaluate(text, expected, letters):
    scores = {'reward': 0.0, 'format': 0.0, 'answer': 0.0}
    try:
        prediction = json.loads(text, object_pairs_hook=unique_object)
    except (ValueError, TypeError):
        return scores
    if (not isinstance(prediction, dict) or set(prediction) != {'answer'}
            or not isinstance(prediction['answer'], str) or prediction['answer'] not in letters):
        return scores
    scores['format'] = 1.0
    scores['answer'] = float(prediction['answer'] == expected)
    scores['reward'] = scores['answer']
    return scores


def read_answer(path):
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Symlink ancestor')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1024:
            raise ValueError('Answer must be a regular file of at most 1024 bytes')
        content = stream.read(1025)
        if len(content) > 1024:
            raise ValueError('Oversize answer')
        return content.decode('utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer-path', type=Path, default=Path('/logs/artifacts/answer.json'))
    parser.add_argument('--info-path', type=Path, default=Path('/tests/gold.json'))
    parser.add_argument('--logs-dir', type=Path, default=Path('/logs/verifier'))
    args = parser.parse_args()
    gold = json.loads(args.info_path.read_text())
    try:
        text = read_answer(args.answer_path)
    except (OSError, UnicodeError, ValueError):
        text = ''
    scores = evaluate(text, gold['answer'], gold['letters'])
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / 'reward.txt').write_text(str(scores['reward']))
    (args.logs_dir / 'reward.json').write_text(json.dumps(scores))


if __name__ == '__main__':
    main()
