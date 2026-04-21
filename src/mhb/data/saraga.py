"""Saraga 1.5 loader (shared by Hindustani and Carnatic corpora).

Saraga 1.5 directory layout (both Hindustani and Carnatic follow this):

    saraga1.5_{tradition}/
        {artist}/
            {album or concert}/
                {track}/
                    {track}.mp3
                    {track}.json              # metadata: raags, taals, form, artists...
                    {track}.ctonic.txt        # tonic in Hz
                    {track}.sections-manual-p.txt   # section boundaries (start, end, label)
                    {track}.tempo-manual.txt  # BPM
                    {track}.pitch-manual.txt  # optional
                    ...

If the archive is a zip (as on NEXUS at saraga1.5_hindustani.zip), we handle
both extracted and zipped layouts: the `unpack` CLI extracts the zip first.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from ..schema import (
    ClipItem, GroundTruth, InstrumentTag, LicenseTag,
    MetricCycle, ModalLabel, SectionSpan,
)
from .facts import synthesize_facts
from .segmentation import assign_tier, is_tier_sufficient, segment_track
from .vocab import load_vocab, normalize

log = logging.getLogger("mhb.data.saraga")


SARAGA_LICENSE = LicenseTag(
    spdx="CC-BY-NC-SA-4.0",
    source_url="https://mtg.github.io/saraga/",
    redistribute=False,
)


def _read_ctonic(path: Path) -> Optional[float]:
    try:
        txt = path.read_text().strip().split()
        return float(txt[0]) if txt else None
    except Exception:
        return None


def _read_tempo(path: Path) -> Optional[float]:
    try:
        txt = path.read_text().strip().split()
        return float(txt[0]) if txt else None
    except Exception:
        return None


def _read_sections(path: Path) -> List[SectionSpan]:
    """Saraga section annotations: tab-separated 'start\\tend\\tlabel' lines."""
    out: List[SectionSpan] = []
    if not path.exists():
        return out
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(",")
            if len(parts) >= 3:
                try:
                    out.append(SectionSpan(
                        label=parts[2].strip(),
                        start_sec=float(parts[0]),
                        end_sec=float(parts[1]),
                    ))
                except ValueError:
                    continue
    except Exception as e:
        log.warning(f"Could not read sections from {path}: {e}")
    return out


def _parse_track_metadata(meta_path: Path) -> dict:
    """Parse the per-track JSON metadata. Layout differs slightly between Hindustani and Carnatic."""
    try:
        return json.loads(meta_path.read_text())
    except Exception as e:
        log.warning(f"Bad metadata {meta_path}: {e}")
        return {}


def _modal_from_metadata(meta: dict, tradition: str,
                         vocab_misses: dict) -> Optional[ModalLabel]:
    """Extract raga from Saraga metadata, normalized to canonical vocab."""
    raw = None
    # Saraga formats observed: {'raags': [{'name': 'Yaman', 'uuid': '...'}]}
    for key in ("raags", "raagas", "ragas"):
        if key in meta and meta[key]:
            item = meta[key][0]
            if isinstance(item, dict):
                raw = item.get("name") or item.get("common_name")
            else:
                raw = str(item)
            if raw:
                break
    if not raw:
        return None
    vocab_name = {
        "hindustani": "ragas_hindustani",
        "carnatic": "ragas_carnatic",
    }[tradition]
    vocab = load_vocab(vocab_name)
    canonical = normalize(raw, vocab)
    if canonical is None:
        vocab_misses.setdefault(vocab_name, []).append(raw)
        # Still return a label with raw name so scoring can see it.
        return ModalLabel(name=raw.lower().replace(" ", "_"),
                          system=f"raga_{tradition}", aliases=[])
    entry = vocab[canonical]
    return ModalLabel(
        name=canonical,
        system=f"raga_{tradition}",
        aliases=entry.get("aliases", []),
        parent=entry.get("parent_raga") or (
            str(entry["melakarta"]) if "melakarta" in entry else None
        ),
    )


def _cycle_from_metadata(meta: dict, tradition: str,
                         vocab_misses: dict) -> Optional[MetricCycle]:
    raw = None
    for key in ("taals", "taalas", "talas"):
        if key in meta and meta[key]:
            item = meta[key][0]
            if isinstance(item, dict):
                raw = item.get("name") or item.get("common_name")
            else:
                raw = str(item)
            if raw:
                break
    if not raw:
        return None
    vocab_name = {
        "hindustani": "talas_hindustani",
        "carnatic": "talas_carnatic",
    }[tradition]
    vocab = load_vocab(vocab_name)
    canonical = normalize(raw, vocab)
    if canonical is None:
        vocab_misses.setdefault(vocab_name, []).append(raw)
        return MetricCycle(name=raw.lower().replace(" ", "_"),
                           system=f"tala_{tradition}")
    entry = vocab[canonical]
    return MetricCycle(
        name=canonical,
        system=f"tala_{tradition}",
        beats=entry.get("beats"),
        accent_pattern=entry.get("division") or entry.get("angas"),
        aliases=entry.get("aliases", []),
    )


def _instruments_from_metadata(meta: dict, tradition: str,
                               vocab_misses: dict) -> List[InstrumentTag]:
    """Saraga lists instruments under 'artists' -> each with 'instrument'."""
    out: List[InstrumentTag] = []
    instruments_vocab = load_vocab("instruments")

    for artist_entry in (meta.get("artists") or []):
        if not isinstance(artist_entry, dict):
            continue
        raw = artist_entry.get("instrument") or artist_entry.get("role")
        if not raw:
            continue
        canonical = normalize(raw, instruments_vocab)
        if canonical is None:
            vocab_misses.setdefault("instruments", []).append(raw)
            continue
        out.append(InstrumentTag(canonical=canonical, source="metadata"))

    # Dedup while preserving order.
    seen = set()
    dedup = []
    for inst in out:
        if inst.canonical not in seen:
            seen.add(inst.canonical)
            dedup.append(inst)
    return dedup


def _form_from_metadata(meta: dict) -> Optional[str]:
    for key in ("form", "forms"):
        v = meta.get(key)
        if isinstance(v, list) and v:
            item = v[0]
            if isinstance(item, dict):
                return (item.get("name") or item.get("common_name") or "").lower() or None
            return str(item).lower()
        if isinstance(v, str):
            return v.lower()
    return None


def _artist_id_from_metadata(meta: dict) -> Optional[str]:
    # Use first 'lead' artist if present, else first artist.
    artists = meta.get("artists") or []
    leads = [a for a in artists if isinstance(a, dict) and a.get("lead")]
    pool = leads or [a for a in artists if isinstance(a, dict)]
    for a in pool:
        name = a.get("name") or a.get("mbid") or a.get("uuid")
        if name:
            return str(name).strip().lower().replace(" ", "_")
    return None


def iter_saraga_tracks(root: Path, tradition: str) -> Iterable[Path]:
    """Yield per-track metadata JSON files (one per track)."""
    root = Path(root)
    for meta_path in sorted(root.rglob("*.json")):
        # Heuristic: track metadata files co-locate with the mp3
        sibling_audio = list(meta_path.parent.glob(meta_path.stem + ".mp3"))
        if sibling_audio:
            yield meta_path


def build_ground_truth(
    meta_path: Path, tradition: str, vocab_misses: dict,
) -> tuple[GroundTruth, Path, dict]:
    """Return (ground_truth, audio_path, meta_dict)."""
    meta = _parse_track_metadata(meta_path)
    audio_path = meta_path.with_suffix(".mp3")
    track_dir = meta_path.parent
    stem = meta_path.stem

    tonic = _read_ctonic(track_dir / f"{stem}.ctonic.txt")
    tempo = _read_tempo(track_dir / f"{stem}.tempo-manual.txt")
    sections = _read_sections(track_dir / f"{stem}.sections-manual-p.txt")

    gt = GroundTruth(
        modal_system=_modal_from_metadata(meta, tradition, vocab_misses),
        metric_cycle=_cycle_from_metadata(meta, tradition, vocab_misses),
        form=_form_from_metadata(meta),
        sections=sections,
        instruments=_instruments_from_metadata(meta, tradition, vocab_misses),
        tonic_hz=tonic,
        tempo_bpm=tempo,
        artist_id=_artist_id_from_metadata(meta),
        recording_id=stem,
        provenance={
            "source_corpus": f"saraga1.5_{tradition}",
            "metadata_file": str(meta_path),
            "sections_file": str(track_dir / f"{stem}.sections-manual-p.txt"),
        },
    )
    return gt, audio_path, meta


def curate_saraga(
    root: Path,
    tradition: str,
    window_sec: float = 25.0,
    max_windows_per_track: int = 4,
    max_negative_instruments: int = 3,
    sample_rate: int = 16000,
    seed: int = 848,
) -> tuple[List[ClipItem], dict]:
    """Produce ClipItems from an already-unpacked Saraga tree."""
    assert tradition in ("hindustani", "carnatic"), tradition
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Saraga root not found: {root}")

    clips: List[ClipItem] = []
    vocab_misses: Dict[str, List[str]] = {}
    dropped: List[dict] = []
    kept_tracks = 0
    considered_tracks = 0

    for meta_path in iter_saraga_tracks(root, tradition):
        considered_tracks += 1
        try:
            gt, audio_path, meta = build_ground_truth(meta_path, tradition, vocab_misses)
        except Exception as e:
            dropped.append({"track": str(meta_path), "reason": f"metadata_parse:{e}"})
            continue

        if not audio_path.exists():
            dropped.append({"track": str(meta_path), "reason": "audio_missing"})
            continue

        if not is_tier_sufficient(gt):
            dropped.append({"track": str(meta_path), "reason": "tier_insufficient"})
            continue

        # Estimate track duration from sections if available, else best-effort via librosa.
        if gt.sections:
            total_dur = max(s.end_sec for s in gt.sections)
        else:
            try:
                import librosa
                total_dur = librosa.get_duration(path=str(audio_path))
            except Exception:
                total_dur = 300.0  # fallback

        windows = segment_track(
            total_duration_sec=total_dur,
            sections=gt.sections,
            window_sec=window_sec,
            max_windows=max_windows_per_track,
            seed=seed,
        )
        if not windows:
            dropped.append({"track": str(meta_path), "reason": "no_usable_windows"})
            continue

        kept_tracks += 1
        for i, (s, e, label) in enumerate(windows):
            # Per-clip GT may narrow sections to the one that contained this window.
            clip_sections = [sec for sec in gt.sections
                             if sec.start_sec <= s < sec.end_sec]
            # A shallow-copy GT limited to this clip.
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
            clip = ClipItem(
                clip_id=f"saraga_{tradition}_{meta_path.stem}_{int(s):06d}_{int(e):06d}",
                tradition=tradition,
                audio_path=str(audio_path),
                sr=sample_rate,
                start_sec=float(s),
                end_sec=float(e),
                tier=assign_tier(clip_gt),
                ground_truth=clip_gt,
                derived_facts=facts,
                license=SARAGA_LICENSE,
                source_corpus=f"saraga1.5_{tradition}",
                audit={"segmentation": "section" if label else "fixed"},
            )
            clips.append(clip)

    report = {
        "source": f"saraga1.5_{tradition}",
        "root": str(root),
        "tracks_considered": considered_tracks,
        "tracks_kept": kept_tracks,
        "clips_produced": len(clips),
        "dropped": dropped,
        "vocab_misses": vocab_misses,
    }
    return clips, report
