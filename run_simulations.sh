#!/bin/bash
#
# Thin wrapper kept for backward compatibility: bootstraps the venv and runs
# the full invsim pipeline. All real logic lives in the invsim package —
# see `invsim run --help` for options.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install -e "$SCRIPT_DIR"
fi

exec "$VENV_DIR/bin/python" -m invsim run "$@"
