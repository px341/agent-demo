#!/usr/bin/env bash
# Verified Mini defaults to a read-only setup check, never a benchmark run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python="${MYAGENT_PYTHON:-$PWD/agentvenv/bin/python}"
exec "$python" -m myagent.swebench_mini "$@"
