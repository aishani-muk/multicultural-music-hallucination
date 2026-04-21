#!/usr/bin/env bash
# Shared preamble for all sbatch scripts.
set -euo pipefail
REPO="${MHB_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)}"
cd "$REPO"
[ -f .env ] && { set -a; . ./.env; set +a; }
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
mkdir -p logs outputs
echo "[mhb] host=$(hostname) job=${SLURM_JOB_ID:-local} repo=$REPO"
