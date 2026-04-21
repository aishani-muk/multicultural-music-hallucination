#!/usr/bin/env bash
# One-command onboarding. Usage:  bash scripts/bootstrap.sh
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$REPO"

echo "[1/4] .env"
[ -f .env ] || { cp .env.example .env; echo "  created .env from .env.example — please fill in keys"; }

echo "[2/4] Python deps (pip install -e .[dev,gemini])"
python -m pip install --upgrade pip >/dev/null
python -m pip install -e ".[dev,gemini]" || python -m pip install -r requirements.txt

echo "[3/4] fixtures + seed prompts"
python fixtures/build_fixtures.py
python scripts/regenerate_seed_prompts.py

echo "[4/4] smoke test"
PYTHONPATH="$REPO/src:${PYTHONPATH:-}" python -m mhb smoke

echo "Done. Next:  source setup_env.sh  then  python -m mhb --help"
