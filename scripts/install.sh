#!/bin/sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-"$ROOT_DIR/.venv"}

"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade-strategy only-if-needed setuptools wheel
"$VENV_DIR/bin/python" -m pip install --no-build-isolation "$ROOT_DIR"

printf '\nInstalled VORTEX into %s\n' "$VENV_DIR"
printf 'Run it with:\n'
printf '  %s/bin/vortex\n' "$VENV_DIR"
