#!/bin/sh
set -eu
python - <<'PYTHON'
from pathlib import Path
Path("/app/answer.json").write_bytes(bytes.fromhex("{{ answer_hex }}"))
PYTHON
