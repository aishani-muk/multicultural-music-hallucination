#!/usr/bin/env bash
# Non-intrusive env loader. Usage:   source setup_env.sh
# Loads .env, reports which keys are set, and exports PYTHONPATH to src/.
# Does NOT modify ~/.bashrc.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
    echo "Loaded: $SCRIPT_DIR/.env"
else
    echo "No .env found. Copy .env.example to .env and fill in keys."
fi

export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

_show() { [ -z "${!1}" ] && echo "[ ] $1 not set" || echo "[x] $1 set (${!1:0:8}...)"; }
_show GOOGLE_API_KEY
_show HF_TOKEN

echo "PYTHONPATH includes $SCRIPT_DIR/src"
