"""CompMusic Turkish-makam and Arab-Andalusian (Moroccan) loaders.

Skeletons: these corpora have heterogeneous metadata formats across their
Zenodo/Dunya releases. The loaders below read a normalized manifest file
(`manifest.jsonl`) that the user prepares with `scripts/prepare_compmusic.py`
(adapter), so the curation code remains stable even as source layouts change.

Expected `manifest.jsonl` row (one per track):

    {"track_id": "makam_turkish_0001",
     "audio_path": "abs/path/to/file.mp3",
     "makam": "Hicaz",
     "usul":  "Aksak",
     "form":  "taksim",
     "instruments": ["ney"],
     "artist_id": "optional",
     "duration_sec": 320.0,
     "sections": [{"label": "taksim", "start": 0.0, "end": 320.0}],
     "provenance": {...}}

Absent fields are treated as missing GT.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..schema import (
    ClipItem, GroundTruth, InstrumentTag, LicenseTag,
    MetricCycle, ModalLabel, SectionSpan,
)
from .facts import synthesize_facts
from .segmentation import assign_tier, is_tier_sufficient, segment_track
from .vocab import load_vocab, normalize

log = logging.getLogger("mhb.data.compmusic")


COMPMUSIC_LICENSES = {
    "turkish_makam": LicenseTag(
        spdx="CC-BY-NC-SA-4.0",
        source_url="https://compmusic.upf.edu/node/318",
        redistribute=False,
    ),
    "arab_andalusian": LicenseTag(
        spdx="CC-BY-NC-SA-4.0",
        source_url="https://compmusic.upf.edu/corpora",
        redistribute=False,
    ),
}


def _modal_from_row(row: dict, tradition: str, vocab_misses: dict) -> Optional[ModalLabel]:
    if tradition == "turkish_makam":
        raw = row.get("makam")
        vocab_name = "makam"
        system = "makam"
    elif tradition == "arab_andalusian":
        raw = row.get("tab") or row.get("tab3") or row.get("mode")
        vocab_name = "tab3"
        system = "tab3"
    else:
        return None
    if not raw:
        return None
    vocab = load_vocab(vocab_name)
    canonical = normalize(raw, vocab)
    if canonical is None:
        vocab_misses.setdefault(vocab_name, []).append(raw)
        return ModalLabel(name=str(raw).lower(), system=system)
    entry = vocab[canonical]
    return ModalLabel(name=canonical, system=system, aliases=entry.get("aliases", []))


def _cycle_from_row(row: dict, tradition: str, vocab_misses: dict) -> Optional[MetricCycle]:
    if tradition == "turkish_makam":
        raw = row.get("usul")
        vocab_name = "usul"
        system = "usul"
    elif tradition == "arab_andalusian":
        raw = row.get("mizan")
        vocab_name = "mizan"
        system = "mizan"
    else:
        return None
    if not raw:
        return None
    vocab = load_vocab(vocab_name)
    canonical = normalize(raw, vocab)
    if canonical is None:
        vocab_misses.setdefault(vocab_name, []).append(raw)
        return MetricCycle(name=str(raw).lower(), system=system)
    entry = vocab[canonical]
    return MetricCycle(
        name=canonical, system=system,
        beats=entry.get("beats"),
        accent_pattern=entry.get("accent_pattern"),
        aliases=entry.get("aliases", []),
    )


def _instruments_from_row(row: dict, vocab_misses: dict) -> List[InstrumentTag]:
    raw_list = row.get("instruments") or []
    if isinstance(raw_list, str):
        raw_list = [raw_list]
    vocab = load_vocab("instruments")
    out: List[InstrumentTag] = []
    seen = set()
    for raw in raw_list:
        canonical = normalize(str(raw), vocab)
        if canonical is None:
            vocab_misses.setdefault("instruments", []).append(raw)
            continue
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(InstrumentTag(canonical=canonical, source="metadata"))
    return out


def _sections_from_row(row: dict) -> List[SectionSpan]:
    out: List[SectionSpan] = []
    for s in row.get("sections") or []:
        try:
            out.append(SectionSpan(
                label=str(s.get("label", "")),
                start_sec=float(s.get("start", 0)),
                end_sec=float(s.get("end", 0)),
            ))
        except Exception:
            continue
    return out


def curate_compmusic(
    manifest_path: Path,
    tradition: str,
    window_sec: float = 25.0,
    max_windows_per_track: int = 4,
    max_negative_instruments: int = 3,
    sample_rate: int = 16000,
    seed: int = 848,
) -> tuple[List[ClipItem], dict]:
    """Curate ClipItems from a normalized CompMusic manifest.jsonl."""
    assert tradition in ("turkish_makam", "arab_andalusian"), tradition
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"CompMusic manifest not found: {manifest_path}. "
            "Prepare it via scripts/prepare_compmusic.py (see DATA.md)."
        )

    license_tag = COMPMUSIC_LICENSES[tradition]
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

            audio_path = Path(row.get("audio_path", ""))
            if not audio_path.exists():
                dropped.append({"track_id": row.get("track_id"), "reason": "audio_missing"})
                continue

            gt = GroundTruth(
                modal_system=_modal_from_row(row, tradition, vocab_misses),
                metric_cycle=_cycle_from_row(row, tradition, vocab_misses),
                form=row.get("form"),
                sections=_sections_from_row(row),
                instruments=_instruments_from_row(row, vocab_misses),
                artist_id=(str(row["artist_id"]).lower().replace(" ", "_")
                           if row.get("artist_id") else None),
                recording_id=row.get("track_id"),
                provenance=row.get("provenance") or {"row": row},
            )
            if not is_tier_sufficient(gt):
                dropped.append({"track_id": row.get("track_id"),
                                "reason": "tier_insufficient"})
                continue

            total_dur = row.get("duration_sec") or (
                max((s.end_sec for s in gt.sections), default=300.0))

            windows = segment_track(
                total_duration_sec=total_dur,
                sections=gt.sections,
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
                clip_sections = [sec for sec in gt.sections
                                 if sec.start_sec <= s < sec.end_sec]
                clip_gt = GroundTruth(
                    modal_system=gt.modal_system,
                    metric_cycle=gt.metric_cycle,
                    form=gt.form,
                    sections=clip_sections,
                    instruments=list(gt.instruments),
                    tonic_hz=gt.tonic_hz,
                    tempo_bpm=gt.tempo_bpm,
                    language=gt.language,
                    artist_id=gt.artist_id,
                    recording_id=gt.recording_id,
                    provenance={**gt.provenance,
                                "window_section_label": label,
                                "window_start_sec": s,
                                "window_end_sec": e},
                )
                facts = synthesize_facts(
                    tradition, clip_gt,
                    max_negative_instruments=max_negative_instruments,
                )
                clip_id = f"{tradition}_{row.get('track_id', 'x')}_{int(s):06d}_{int(e):06d}"
                clips.append(ClipItem(
                    clip_id=clip_id,
                    tradition=tradition,
                    audio_path=str(audio_path),
                    sr=sample_rate,
                    start_sec=float(s), end_sec=float(e),
                    tier=assign_tier(clip_gt),
                    ground_truth=clip_gt,
                    derived_facts=facts,
                    license=license_tag,
                    source_corpus=f"compmusic_{tradition}",
                    audit={"segmentation": "section" if label else "fixed"},
                ))

    report = {
        "source": f"compmusic_{tradition}",
        "manifest": str(manifest_path),
        "tracks_considered": considered,
        "tracks_kept": kept_tracks,
        "clips_produced": len(clips),
        "dropped": dropped,
        "vocab_misses": vocab_misses,
    }
    return clips, report
