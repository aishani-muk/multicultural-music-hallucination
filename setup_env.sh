#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Environment setup for the hallucination benchmark pipeline
#  Run once:  source setup_env.sh
# ═══════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Load .env if it exists ────────────────────────────────
if [ -f "$SCRIPT_DIR/.env" ]; then
    echo "Loading .env ..."
    set -a
    source "$SCRIPT_DIR/.env"
    set +a
fi

# ── Validate required keys ────────────────────────────────
if [ -z "$GOOGLE_API_KEY" ]; then
    echo ""
    echo "WARNING: GOOGLE_API_KEY is not set."
    echo "  1. Go to https://aistudio.google.com/apikey"
    echo "  2. Create an API key"
    echo "  3. Add it to $SCRIPT_DIR/.env:"
    echo "     echo 'GOOGLE_API_KEY=your-key' >> $SCRIPT_DIR/.env"
    echo ""
else
    echo "GOOGLE_API_KEY is set (${GOOGLE_API_KEY:0:8}...)"
fi

if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "WARNING: HF_TOKEN is not set (needed for Music Flamingo)."
    echo "  1. Go to https://huggingface.co/settings/tokens"
    echo "  2. Create a read token"
    echo "  3. Accept the license at https://huggingface.co/nvidia/music-flamingo-hf"
    echo "  4. Add it to $SCRIPT_DIR/.env:"
    echo "     echo 'HF_TOKEN=hf_your_token' >> $SCRIPT_DIR/.env"
    echo ""
else
    echo "HF_TOKEN is set (${HF_TOKEN:0:8}...)"
fi

# ── Also append to ~/.bashrc for persistence ──────────────
MARKER="# >>> music-hallucination-benchmark env >>>"
if ! grep -q "$MARKER" ~/.bashrc 2>/dev/null; then
    echo ""
    echo "Adding env loader to ~/.bashrc for persistent sessions..."
    cat >> ~/.bashrc << BASHEOF

$MARKER
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a; source "$SCRIPT_DIR/.env"; set +a
fi
# <<< music-hallucination-benchmark env <<<
BASHEOF
    echo "Done. Future shell sessions will auto-load .env."
fi

echo ""
echo "Environment setup complete."
echo "  Pipeline dir: $SCRIPT_DIR"
echo "  Config:       $SCRIPT_DIR/configs/benchmark_config.yaml"
