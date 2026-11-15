#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Run bash scripts/setup.sh first.' >&2
  exit 1
fi
.venv/bin/python -m app.manage check
# One owner process for Milvus Lite. Do not add --reload or extra workers.
exec .venv/bin/python -m uvicorn app.main:app \
  --host "${BIND_HOST:-127.0.0.1}" --port "${PORT:-8000}" \
  --workers 1 --timeout-graceful-shutdown 30
