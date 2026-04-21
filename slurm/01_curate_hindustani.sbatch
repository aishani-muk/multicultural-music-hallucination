#!/usr/bin/env bash
#SBATCH --job-name=mhb-curate-hindustani
#SBATCH --partition=tron
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
source "$(dirname "$0")/_common.sh"
RAW="${SARAGA_HINDUSTANI_ROOT:-outputs/data/raw/saraga_hindustani}"
OUT="${MHB_CURATED_ROOT:-outputs/data/curated}"
python -m mhb data curate --source saraga_hindustani --raw "$RAW" --out "$OUT"
python -m mhb data audit  --curated "$OUT"
