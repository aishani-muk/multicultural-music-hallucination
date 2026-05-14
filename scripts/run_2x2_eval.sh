#!/bin/bash
# Full 2×2 evaluation: (MusicGen × Stable Audio) × (OpenAI × Music Flamingo)
# Run from repo root: bash scripts/run_2x2_eval.sh

set -e
cd "$(dirname "$0")/.."

CONFIG=configs/default.yaml
MUSICGEN_MANIFEST=outputs/generated_audio/musicgen/generation_manifest.json
STABLE_MANIFEST=outputs/generated_audio/stable_audio/generation_manifest.json

# # ── Verify all 4 cells ─────────────────────────────────────────────

# echo "=== MusicGen × OpenAI ==="
# python -m mhb verify --config $CONFIG --verifier openai \
#   --target prompts --manifest $MUSICGEN_MANIFEST \
#   --out outputs/verification_results/musicgen/openai

# echo "=== MusicGen × Music Flamingo ==="
# python -m mhb verify --config $CONFIG --verifier music_flamingo \
#   --target prompts --manifest $MUSICGEN_MANIFEST \
#   --out outputs/verification_results/musicgen/music_flamingo

# echo "=== Stable Audio × OpenAI ==="
# python -m mhb verify --config $CONFIG --verifier openai \
#   --target prompts --manifest $STABLE_MANIFEST \
#   --out outputs/verification_results/stable_audio/openai

# echo "=== Stable Audio × Music Flamingo ==="
# python -m mhb verify --config $CONFIG --verifier music_flamingo \
#   --target prompts --manifest $STABLE_MANIFEST \
#   --out outputs/verification_results/stable_audio/music_flamingo

# ── Score all 4 cells ──────────────────────────────────────────────

echo "=== Scoring ==="
mkdir -p outputs/scores

echo "=== MusicGen × OpenAI ==="
python -m mhb score \
  --results outputs/verification_results/musicgen/openai/openai/openai__gpt-4o-audio-preview_all_results.json \
  --out outputs/scores/musicgen_x_openai.json

echo "=== MusicGen × Music Flamingo ==="
python -m mhb score \
  --results outputs/verification_results/musicgen/music_flamingo/music_flamingo/music_flamingo__nvidia__audio-flamingo-3-hf_all_results.json \
  --out outputs/scores/musicgen_x_flamingo.json

echo "=== Stable Audio × OpenAI ==="
python -m mhb score \
  --results outputs/verification_results/stable_audio/openai/openai/openai__gpt-4o-audio-preview_all_results.json \
  --out outputs/scores/stable_audio_x_openai.json

echo "=== Stable Audio × Music Flamingo ==="
python -m mhb score \
  --results outputs/verification_results/stable_audio/music_flamingo/music_flamingo/music_flamingo__nvidia__audio-flamingo-3-hf_all_results.json \
  --out outputs/scores/stable_audio_x_flamingo.json

# ── Cross-model analysis + figures ─────────────────────────────────

echo "=== Analysis ==="
python -m mhb analyze \
  --scores "musicgen_x_openai=outputs/scores/musicgen_x_openai.json" \
           "musicgen_x_flamingo=outputs/scores/musicgen_x_flamingo.json" \
           "stable_audio_x_openai=outputs/scores/stable_audio_x_openai.json" \
           "stable_audio_x_flamingo=outputs/scores/stable_audio_x_flamingo.json" \
  --out outputs/figures

echo "=== Done ==="
echo "Scores:  outputs/scores/"
echo "Figures: outputs/figures/"
