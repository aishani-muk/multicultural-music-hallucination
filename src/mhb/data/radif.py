"""Iranian radif skeleton loader.

No fully-open corpus with per-clip dastgāh labels is wired in yet. Candidate
sources (to be evaluated manually for license/availability):

  - Mahoor radif corpus (commercial reference recordings; not redistributable)
  - ShirazRadifDataset (academic; limited)
  - Individual scholar releases on Zenodo

If `$DATA_ROOT/raw/radif/manifest.jsonl` exists with the same normalized row
format as CompMusic, the loader uses it. Otherwise it returns an empty list
and prints a single informational notice.
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
from .segmentation import assign_tier, is_tier_sufficient, segment_track
from .vocab import load_vocab, normalize

log = logging.getLogger("mhb.data.radif")


RADIF_LICENSE = LicenseTag(spdx="unknown", redistribute=False)


def _modal_from_row(row: dict, vocab_misses: dict):
    raw = row.get("dastgah") or row.get("avaz") or row.get("mode")
    if not raw:
        return None
    vocab = load_vocab("dastgah")
    canonical = normalize(str(raw), vocab)
    if canonical is None:
        vocab_misses.setdefault("dastgah", []).append(raw)
        return ModalLabel(name=str(raw).lower(), system="dastgah")
    entry = vocab[canonical]
    return ModalLabel(name=canonical, system="dastgah", aliases=entry.get("aliases", []))


def curate_radif(
    manifest_path: Path,
    window_sec: float = 25.0,
    max_windows_per_track: int = 3,
    max_negative_instruments: int = 3,
    sample_rate: int = 16000,
    seed: int = 848,
) -> tuple[List[ClipItem], dict]:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        log.info(
            "Radif manifest not found at %s; skipping. See DATA.md for candidate sources.",
            manifest_path,
        )
        return [], {"source": "radif", "manifest": str(manifest_path),
                    "tracks_considered": 0, "tracks_kept": 0,
                    "clips_produced": 0, "dropped": [], "vocab_misses": {},
                    "note": "no_manifest"}

    clips: List[ClipItem] = []
    vocab_misses: Dict[str, List[str]] = {}
    dropped = []
    kept = 0
    considered = 0

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            considered += 1
            row = json.loads(line)
            audio_path = Path(row.get("audio_path", ""))
            if not audio_path.exists():
                dropped.append({"track_id": row.get("track_id"), "reason": "audio_missing"})
                continue

            gt = GroundTruth(
                modal_system=_modal_from_row(row, vocab_misses),
                metric_cycle=None,   # avaz is typically free-rhythm
                form=row.get("form"),
                sections=[SectionSpan(**s) for s in (row.get("sections") or [])
                          if all(k in s for k in ("label", "start_sec", "end_sec"))],
                instruments=[InstrumentTag(canonical=str(i), source="metadata")
                             for i in (row.get("instruments") or [])],
                artist_id=row.get("artist_id"),
                recording_id=row.get("track_id"),
                provenance={"source_corpus": "radif", "row": row},
            )

            if not is_tier_sufficient(gt):
                dropped.append({"track_id": row.get("track_id"), "reason": "tier_insufficient"})
                continue

            total_dur = row.get("duration_sec") or max(
                (s.end_sec for s in gt.sections), default=300.0)
            windows = segment_track(total_duration_sec=total_dur, sections=gt.sections,
                                    window_sec=window_sec, max_windows=max_windows_per_track, seed=seed)
            if not windows:
                dropped.append({"track_id": row.get("track_id"), "reason": "no_usable_windows"})
                continue

            kept += 1
            for s, e, label in windows:
                facts = synthesize_facts("radif", gt, max_negative_instruments=max_negative_instruments)
                clips.append(ClipItem(
                    clip_id=f"radif_{row.get('track_id','x')}_{int(s):06d}_{int(e):06d}",
                    tradition="radif",
                    audio_path=str(audio_path),
                    sr=sample_rate,
                    start_sec=float(s), end_sec=float(e),
                    tier=assign_tier(gt),
                    ground_truth=gt, derived_facts=facts,
                    license=RADIF_LICENSE, source_corpus="radif",
                    audit={"segmentation": "section" if label else "fixed"},
                ))

    return clips, {
        "source": "radif",
        "manifest": str(manifest_path),
        "tracks_considered": considered,
        "tracks_kept": kept,
        "clips_produced": len(clips),
        "dropped": dropped,
        "vocab_misses": vocab_misses,
    }
