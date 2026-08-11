#!/bin/sh
set -eu
printf '%s' '{{ answer }}' | base64 -d > /app/answer.txt
