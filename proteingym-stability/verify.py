"""Strict artifact verifier for a separate trusted Harbor container."""
import json
from pathlib import Path
import sys


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def grade(answer_path, gold_path):
    try:
        path = Path(answer_path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024:
            return 0
        answer = json.loads(path.read_text(), object_pairs_hook=unique_object)
        gold = json.loads(Path(gold_path).read_text())
        return int(isinstance(answer, dict) and set(answer) == {'answer'} and isinstance(answer['answer'], str) and answer['answer'] in ('A', 'B') and answer['answer'] == gold['answer'])
    except (ValueError, OSError, UnicodeError):
        return 0


if __name__ == '__main__':
    Path(sys.argv[3]).write_text(str(grade(sys.argv[1], sys.argv[2])) + '\n')
