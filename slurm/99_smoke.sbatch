#!/usr/bin/env bash
#SBATCH --job-name=mhb-smoke
#SBATCH --partition=tron
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
source "$(dirname "$0")/_common.sh"
python -m mhb smoke
