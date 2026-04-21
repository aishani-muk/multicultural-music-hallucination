"""Reggaeton loader using MTG-Jamendo (CC-licensed) `reggaeton` tag.

The MTG-Jamendo dataset is distributed with TSV annotations (`autotagging_moodtheme.tsv`,
`autotagging_genre.tsv`, etc.) and per-track MP3s. This loader expects a
normalized manifest.jsonl with at minimum:

    {"track_id": "jamendo_123456",
     "audio_path": "abs/path/to/123456.mp3",
     "genre_tags": ["reggaeton", "reggae"],
     "instrument_tags": ["electronic", "drums", "bass"],
     "duration_sec": 210.0,
     "artist_id": "optional_jamendo_artist",
     "tempo_bpm": 95.0}

Prepare it with `scripts/prepare_jamendo_reggaeton.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

from ..schema import (
    ClipItem, GroundTruth, InstrumentTag, LicenseTag,
    MetricCycle, ModalLabel, SectionSpan,
)
from .facts import synthesize_facts
from .segmentation import assign_tier, segment_track
from .vocab import load_vocab, normalize

log = logging.getLogger("mhb.data.jamendo_reggaeton")


JAMENDO_LICENSE = LicenseTag(
    spdx="CC-BY-NC-SA-4.0",
    source_url="https://mtg.github.io/mtg-jamendo-dataset/",
    redistribute=False,
)


DEMBOW_CYCLE = MetricCycle(
    name="dembow", system="dembow", beats=8,
    accent_pattern="3+3+2 (boom-ch-boom-chick)",
    aliases=["dem_bow", "dembow_riddim"],
)

SPANISH_MODAL_STUB = ModalLabel(
    name="reggaeton_minor_pop",
    system="popular_western",
    aliases=[],
    notes="Reggaeton typically uses minor/phrygian-flavored pop progressions; "
          "no canonical modal label, so modal_system is a stub for tiering only.",
)


def curate_jamendo_reggaeton(
    manifest_path: Path,
    window_sec: float = 25.0,
    max_windows_per_track: int = 3,
    max_negative_instruments: int = 3,
    sample_rate: int = 16000,
    seed: int = 848,
) -> tuple[List[ClipItem], dict]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Jamendo reggaeton manifest not found: {manifest_path}. "
            "Prepare it via scripts/prepare_jamendo_reggaeton.py."
        )

    instruments_vocab = load_vocab("instruments")
    clips: List[ClipItem] = []
    vocab_misses: Dict[str, List[str]] = {}
    dropped: List[dict] = []
    kept_tracks = 0
    considered = 0

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            considered += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                dropped.append({"line": line[:120], "reason": "json_parse"})
                continue

            genre_tags = [t.lower() for t in (row.get("genre_tags") or [])]
            if "reggaeton" not in genre_tags:
                dropped.append({"track_id": row.get("track_id"),
                                "reason": "not_reggaeton"})
                continue

            audio_path = Path(row.get("audio_path", ""))
            if not audio_path.exists():
                dropped.append({"track_id": row.get("track_id"),
                                "reason": "audio_missing"})
                continue

            # Instruments: normalize jamendo instrument tags to our vocab where possible.
            inst_tags = row.get("instrument_tags") or []
            instruments: List[InstrumentTag] = []
            seen = set()
            for raw in inst_tags:
                canonical = normalize(str(raw), instruments_vocab)
                if canonical and canonical not in seen:
                    seen.add(canonical)
                    instruments.append(InstrumentTag(canonical=canonical, source="tag"))
                elif canonical is None:
                    vocab_misses.setdefault("instruments", []).append(raw)

            # Always add electronic/kick/clap/synth_bass as implicit dembow-stack if absent.
            for implicit in ("dembow_kick", "clap", "synth_bass"):
                if implicit not in seen:
                    instruments.append(InstrumentTag(canonical=implicit, source="auto"))
                    seen.add(implicit)

            gt = GroundTruth(
                modal_system=SPANISH_MODAL_STUB,
                metric_cycle=DEMBOW_CYCLE,
                form="reggaeton_song",
                sections=[],  # usually no per-section annotations
                instruments=instruments,
                tempo_bpm=row.get("tempo_bpm"),
                language="es" if row.get("language", "es") in ("es", "spanish", None) else row["language"],
                artist_id=(str(row["artist_id"]).lower().replace(" ", "_")
                           if row.get("artist_id") else None),
                recording_id=row.get("track_id"),
                provenance={
                    "source_corpus": "mtg_jamendo_reggaeton",
                    "genre_tags": genre_tags,
                    "instrument_tags": inst_tags,
                },
            )

            total_dur = row.get("duration_sec") or 210.0
            windows = segment_track(
                total_duration_sec=total_dur,
                sections=[],
                window_sec=window_sec,
                max_windows=max_windows_per_track,
                seed=seed,
            )
            if not windows:
                dropped.append({"track_id": row.get("track_id"),
                                "reason": "no_usable_windows"})
                continue

            kept_tracks += 1
            for s, e, label in windows:
                facts = synthesize_facts("reggaeton", gt,
                                         max_negative_instruments=max_negative_instruments)
                clip_id = f"reggaeton_{row.get('track_id', 'x')}_{int(s):06d}_{int(e):06d}"
                clips.append(ClipItem(
                    clip_id=clip_id,
                    tradition="reggaeton",
                    audio_path=str(audio_path),
                    sr=sample_rate,
                    start_sec=float(s), end_sec=float(e),
                    tier=assign_tier(gt),
                    ground_truth=gt,
                    derived_facts=facts,
                    license=JAMENDO_LICENSE,
                    source_corpus="mtg_jamendo_reggaeton",
                    audit={"segmentation": "fixed"},
                ))

    return clips, {
        "source": "mtg_jamendo_reggaeton",
        "manifest": str(manifest_path),
        "tracks_considered": considered,
        "tracks_kept": kept_tracks,
        "clips_produced": len(clips),
        "dropped": dropped,
        "vocab_misses": vocab_misses,
    }
