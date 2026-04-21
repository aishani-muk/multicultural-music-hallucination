#!/usr/bin/env bash
#SBATCH --job-name=mhb-download
#SBATCH --partition=class
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
source "$(dirname "$0")/_common.sh"
bash scripts/download_datasets.sh "${@:-all}"
