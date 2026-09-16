"""Retrieve pinned public inputs without credentials; verify before accepting files."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent


def checksum(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sources():
    provenance = json.loads((ROOT / 'provenance.json').read_text())
    return [(name, info['url'], info['sha256']) for name, info in provenance['archives'].items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name, url, expected in sources():
        destination = args.output / name
        if destination.exists():
            if checksum(destination) != expected:
                raise ValueError(f'Existing source checksum mismatch: {name}')
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.part')
        try:
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if checksum(temporary) != expected:
                raise ValueError(f'Download checksum mismatch: {name}')
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        print(f'Verified {name}')


if __name__ == '__main__':
    main()
