"""Fetch pinned public sources, validating each file before writing it."""
import hashlib
import json
from pathlib import Path
from pathlib import Path
import urllib.request

from .main import REVISION, SOURCE


def fetch(destination: Path = SOURCE, base_url: str | None = None):
    base_url = base_url or f'https://raw.githubusercontent.com/wanhaoliu/OmniMatBench/{REVISION}/'
    hashes = json.loads((SOURCE / 'sha256.json').read_text())
    for name, expected in hashes.items():
        target = destination / name
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        url = base_url.rstrip('/') + '/' + name
        with urllib.request.urlopen(url, timeout=120) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f'Source hash mismatch: {name}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        print(f'Fetched {name}')


if __name__ == '__main__':
    fetch()
