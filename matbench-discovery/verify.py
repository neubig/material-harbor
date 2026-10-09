"""Standard-library-only verifier; never import agent-writable code."""
import json
import os
import stat
from pathlib import Path

MAX_BYTES = 1024


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def read_answer(directory: Path) -> bool:
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open('answer.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    with os.fdopen(fd, 'rb') as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BYTES or info.st_nlink != 1:
            raise ValueError('invalid answer file')
        raw = handle.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError('oversized answer')
    answer = json.loads(raw, object_pairs_hook=unique_object)
    if type(answer) is not dict or set(answer) != {'stable'} or type(answer['stable']) is not bool:
        raise ValueError('expected exactly one boolean stable')
    return answer['stable']


def evaluate(directory: Path, gold: bool) -> dict:
    if type(gold) is not bool:
        raise ValueError('invalid verifier key')
    try:
        answer = read_answer(directory)
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        return {'reward': 0, 'status': 'invalid', 'error': type(exc).__name__}
    return {'reward': int(answer == gold), 'status': 'correct' if answer == gold else 'incorrect'}


if __name__ == '__main__':
    gold = json.loads(Path('/tests/gold.json').read_text())['stable']
    result = evaluate(Path('/output'), gold)
    logs = Path('/logs/verifier')
    logs.mkdir(parents=True, exist_ok=True)
    (logs / 'reward.txt').write_text(str(result['reward']) + '\n')
    (logs / 'result.json').write_text(json.dumps(result) + '\n')
