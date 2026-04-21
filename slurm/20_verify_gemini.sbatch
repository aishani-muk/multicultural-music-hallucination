#!/usr/bin/env bash
#SBATCH --job-name=mhb-verify-gemini
#SBATCH --partition=tron
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
source "$(dirname "$0")/_common.sh"
# Target prompts (generation track). Swap to --target clips for understanding track.
python -m mhb verify \
    --config configs/understanding.yaml \
    --verifier gemini \
    --target prompts \
    --manifest outputs/generated_audio/generation_manifest.json
