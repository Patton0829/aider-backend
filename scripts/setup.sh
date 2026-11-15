#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 12), "Server lock requires Python >= 3.12 (3.12 / 3.13 recommended)"'
if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-server.lock
.venv/bin/python -m pip install --no-deps -e '.[milvus]'
if [[ ! -f .env ]]; then
  (umask 077; cp .env.example .env)
  echo 'Created .env. Fill DASHSCOPE_API_KEY on this server before starting.'
else
  echo 'Existing .env preserved.'
fi
echo 'Setup complete. Run .venv/bin/python -m app.manage check, then bash scripts/start.sh.'
