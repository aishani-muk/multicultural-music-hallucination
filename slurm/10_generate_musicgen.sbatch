#!/usr/bin/env bash
#SBATCH --job-name=mhb-musicgen
#SBATCH --partition=tron
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
source "$(dirname "$0")/_common.sh"
python -m mhb generate --config configs/generation.yaml
