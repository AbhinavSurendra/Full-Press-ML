#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv}"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR[dev]"

cat <<EOF
Project virtual environment created at:
  $VENV_DIR

Activate it with:
  source "$VENV_DIR/bin/activate"

Or run commands directly with:
  "$VENV_DIR/bin/python" scripts/build_possessions.py --help
EOF
