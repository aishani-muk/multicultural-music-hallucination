#!/usr/bin/env bash
#SBATCH --job-name=mhb-unpack-saraga
#SBATCH --partition=tron
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
source "$(dirname "$0")/_common.sh"
ZIP="${SARAGA_HINDUSTANI_ZIP:-/fs/classhomes/amukherj/research/saraga1.5_hindustani.zip}"
OUT="${SARAGA_HINDUSTANI_ROOT:-outputs/data/raw/saraga_hindustani}"
python -m mhb data unpack --zip "$ZIP" --out "$OUT"
