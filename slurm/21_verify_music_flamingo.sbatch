#!/usr/bin/env bash
#SBATCH --job-name=mhb-verify-mf
#SBATCH --partition=tron
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
source "$(dirname "$0")/_common.sh"
python -m mhb verify \
    --config configs/understanding.yaml \
    --verifier music_flamingo \
    --target prompts \
    --manifest outputs/generated_audio/generation_manifest.json
