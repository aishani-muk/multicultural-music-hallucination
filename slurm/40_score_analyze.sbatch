#!/usr/bin/env bash
#SBATCH --job-name=mhb-score-analyze
#SBATCH --partition=tron
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
source "$(dirname "$0")/_common.sh"

VER_DIR=outputs/verification_results
SCORES=outputs/scores; mkdir -p "$SCORES"

for V in gemini music_flamingo; do
    R="$VER_DIR/$V"/*_all_results.json
    for f in $R; do
        [ -f "$f" ] || continue
        python -m mhb score --results "$f" --out "$SCORES/${V}_scores.json"
    done
done

ARGS=()
[ -f "$SCORES/gemini_scores.json" ]         && ARGS+=("gemini=$SCORES/gemini_scores.json")
[ -f "$SCORES/music_flamingo_scores.json" ] && ARGS+=("music_flamingo=$SCORES/music_flamingo_scores.json")
python -m mhb analyze --scores "${ARGS[@]}" --out outputs/figures
