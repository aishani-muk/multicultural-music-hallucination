#!/usr/bin/env bash
# Download multicultural music corpora directly to NEXUS classhomes storage.
# Idempotent and resumable: re-running skips files that already exist or
# continues interrupted transfers via `wget -c`.
#
# Usage:
#   bash scripts/download_datasets.sh [TARGET ...]
#
# Targets:  carnatic | arab | makam | jamendo | all (default)
#
# Data root: $MHB_DATA_ROOT (default: /fs/classhomes/amukherj/research)

set -euo pipefail
DATA_ROOT="${MHB_DATA_ROOT:-/fs/classhomes/amukherj/research}"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

# wget flags: resume, retries, sensible timeouts, concise progress, no clobber.
WGET=(wget --continue --tries=10 --timeout=60 --waitretry=5 --progress=dot:giga)

targets=("$@")
[ "${#targets[@]}" -eq 0 ] && targets=(all)
want() { for t in "${targets[@]}"; do [ "$t" = all ] || [ "$t" = "$1" ] && return 0; done; return 1; }

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }


# ─── Saraga 1.5 Carnatic  (~11 GB) ───────────────────────────────────────
if want carnatic; then
    log "→ Saraga 1.5 Carnatic"
    "${WGET[@]}" -O saraga1.5_carnatic.zip \
        "https://zenodo.org/records/4301737/files/saraga1.5_carnatic.zip?download=1"
    log "   done: $(du -h saraga1.5_carnatic.zip | cut -f1)"
fi


# ─── Arab-Andalusian  (8.9 GB) ───────────────────────────────────────────
if want arab; then
    log "→ Arab-Andalusian (CompMusic)"
    "${WGET[@]}" -O ArabAndalusianDataset.tar.gz \
        "https://zenodo.org/records/1291776/files/ArabAndalusianDataset.tar.gz?download=1"
    log "   done: $(du -h ArabAndalusianDataset.tar.gz | cut -f1)"
fi


# ─── OTMM Turkish Makam metadata  (100 MB, audio is copyright-restricted) ─
if want makam; then
    log "→ OTMM Turkish Makam (metadata + pitch tracks; NO audio)"
    "${WGET[@]}" -O otmm_makam_recognition_dataset.zip \
        "https://zenodo.org/records/58408/files/otmm_makam_recognition_dataset-dlfm2016.zip?download=1" \
        || "${WGET[@]}" -O otmm_makam_recognition_dataset.zip \
           "https://zenodo.org/records/58408/files/MTG/otmm_makam_recognition_dataset-dlfm2016.zip?download=1"
    log "   done: $(du -h otmm_makam_recognition_dataset.zip | cut -f1)"
fi


# ─── MTG-Jamendo reggaeton subset  (~3-5 GB, est.) ───────────────────────
# MTG-Jamendo's audio mirror needs their download.py; we clone the repo and
# run their script filtered to tracks with genre=reggaeton.
if want jamendo; then
    log "→ MTG-Jamendo (cloning metadata repo)"
    REPO="$DATA_ROOT/mtg-jamendo-dataset"
    if [ ! -d "$REPO" ]; then
        git clone --depth 1 https://github.com/MTG/mtg-jamendo-dataset.git "$REPO"
    fi

    AUDIO_OUT="$DATA_ROOT/mtg_jamendo_reggaeton/audio"
    META_OUT="$DATA_ROOT/mtg_jamendo_reggaeton/metadata"
    mkdir -p "$AUDIO_OUT" "$META_OUT"

    # Build a list of TRACK_IDs with genre=reggaeton from their autotagging_genre TSV.
    TSV="$REPO/data/autotagging_genre.tsv"
    if [ ! -f "$TSV" ]; then
        log "   ERROR: $TSV missing after clone"; exit 1
    fi

    # Format: TRACK_ID<TAB>ARTIST_ID<TAB>...<TAB>TAGS(space-separated)
    # Extract track_ids where tags contain 'genre---reggaeton'.
    python3 - "$TSV" "$META_OUT/reggaeton_ids.txt" <<'PY'
import sys, csv
tsv, out = sys.argv[1], sys.argv[2]
ids = []
with open(tsv, newline="") as f:
    r = csv.reader(f, delimiter="\t")
    header = next(r, None)
    for row in r:
        tags = " ".join(row[5:]) if len(row) >= 6 else ""
        if "genre---reggaeton" in tags:
            ids.append(row[0])
with open(out, "w") as g:
    g.write("\n".join(ids) + "\n")
print(f"  found {len(ids)} reggaeton tracks")
PY

    N=$(wc -l < "$META_OUT/reggaeton_ids.txt")
    log "   $N reggaeton TRACK_IDs extracted → $META_OUT/reggaeton_ids.txt"

    # Run MTG's download.py with a custom subset. Their script expects a metadata TSV;
    # we build one filtered down to reggaeton only, so we only download those files.
    SUBSET_TSV="$META_OUT/autotagging_genre_reggaeton.tsv"
    head -n1 "$TSV" > "$SUBSET_TSV"
    grep -F -f "$META_OUT/reggaeton_ids.txt" "$TSV" >> "$SUBSET_TSV" || true

    log "   downloading reggaeton audio subset via MTG's script…"
    # Their downloader: python3 scripts/download/download.py --type audio <tsv> <outdir>
    # It uses essentia.upf.edu as the mirror. `--unpack` extracts from tar.
    python3 "$REPO/scripts/download/download.py" \
        --type audio \
        "$SUBSET_TSV" \
        "$AUDIO_OUT" \
        --unpack \
        --remove || log "   (note: downloader reported errors; re-run to resume)"

    log "   done. Contents: $(du -sh $AUDIO_OUT 2>/dev/null | cut -f1)"
fi


log "ALL DONE"
log "  Data root: $DATA_ROOT"
ls -lh "$DATA_ROOT" 2>/dev/null | tail -n +2
