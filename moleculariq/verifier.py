"""Finite exact JSON verifier, executed only in a separate environment."""
import json
import os
import stat
from pathlib import Path

MAX_BYTES = 4096


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate key')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError('nonfinite JSON')


def accepts(text, reference):
    try:
        if len(text.encode('utf-8')) > MAX_BYTES:
            return False
        answer = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        key = reference['key']
        return (type(answer) is dict and set(answer) == {key}
                and type(answer[key]) is int
                and 0 <= answer[key] <= reference['upper_bound']
                and answer[key] == reference['value'])
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return False


def read_answer(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('answer must be a regular file')
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('oversize answer')
    return data.decode('utf-8')


def main():
    reward = False
    try:
        reference = json.loads(Path('/tests/reference.json').read_text())
        reward = accepts(read_answer('/app/answer.json'), reference)
    except (OSError, ValueError, UnicodeError):
        pass
    Path('/logs/verifier').mkdir(parents=True, exist_ok=True)
    Path('/logs/verifier/reward.txt').write_text(str(int(reward)))


if __name__ == '__main__':
    main()
