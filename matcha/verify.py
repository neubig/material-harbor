import argparse
import json
import os
import re
import stat
from pathlib import Path


def read_answer(path):
    if any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Symlink ancestor')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 2:
            raise ValueError('Expected a regular file containing one letter')
        content = stream.read(3)
        if len(content) > 2:
            raise ValueError('Oversize answer')
        return content.decode('ascii')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer-path', type=Path, default=Path('/app/answer.txt'))
    parser.add_argument('--gold-path', type=Path, default=Path('/tests/gold.json'))
    parser.add_argument('--logs-dir', type=Path, default=Path('/logs/verifier'))
    args = parser.parse_args()
    gold = json.loads(args.gold_path.read_text())
    reward = 0.0
    try:
        text = read_answer(args.answer_path)
        if re.fullmatch(r'[A-Z]\n?', text) and text[0] in gold['letters']:
            reward = float(text[0] == gold['answer'])
    except (OSError, UnicodeError, ValueError):
        pass
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    (args.logs_dir / 'reward.txt').write_text(str(reward))


if __name__ == '__main__':
    main()
