#!/bin/sh
set -eu
ANSWER_HEX='{{ answer }}' python -c 'import os; from pathlib import Path; Path("/app/answer.txt").write_bytes(bytes.fromhex(os.environ["ANSWER_HEX"]))'
