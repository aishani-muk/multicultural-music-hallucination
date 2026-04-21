"""Unified 'mhb data curate' entry point across all traditions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..schema import ClipItem, save_clip_items
from ..utils.io import atomic_write_json
from .splits import artist_disjoint_split, coverage_summary

log = logging.getLogger("mhb.data.curate")


SOURCE_ALIASES = {
    "saraga_hindustani": ("hindustani", "saraga"),
    "saraga_carnatic":   ("carnatic", "saraga"),
    "compmusic_turkish_makam":    ("turkish_makam", "compmusic"),
    "compmusic_arab_andalusian":  ("arab_andalusian", "compmusic"),
    "mtg_jamendo_reggaeton":      ("reggaeton", "jamendo"),
    "radif":             ("radif", "radif"),
}


def curate_source(
    source: str,
    raw_root: Path,
    out_dir: Path,
    window_sec: float = 25.0,
    max_windows_per_track: int = 4,
    max_negative_instruments: int = 3,
    sample_rate: int = 16000,
    seed: int = 848,
) -> dict:
    """Curate a single source into `out_dir/<tradition>/clips.jsonl`."""
    if source not in SOURCE_ALIASES:
        raise ValueError(
            f"Unknown source: {source}. Known: {sorted(SOURCE_ALIASES.keys())}"
        )
    tradition, loader = SOURCE_ALIASES[source]

    if loader == "saraga":
        from .saraga import curate_saraga
        clips, report = curate_saraga(
            raw_root, tradition,
            window_sec=window_sec,
            max_windows_per_track=max_windows_per_track,
            max_negative_instruments=max_negative_instruments,
            sample_rate=sample_rate, seed=seed,
        )
    elif loader == "compmusic":
        from .compmusic import curate_compmusic
        clips, report = curate_compmusic(
            raw_root / "manifest.jsonl" if raw_root.is_dir() else raw_root,
            tradition,
            window_sec=window_sec,
            max_windows_per_track=max_windows_per_track,
            max_negative_instruments=max_negative_instruments,
            sample_rate=sample_rate, seed=seed,
        )
    elif loader == "jamendo":
        from .jamendo_reggaeton import curate_jamendo_reggaeton
        clips, report = curate_jamendo_reggaeton(
            raw_root / "manifest.jsonl" if raw_root.is_dir() else raw_root,
            window_sec=window_sec,
            max_windows_per_track=max_windows_per_track,
            max_negative_instruments=max_negative_instruments,
            sample_rate=sample_rate, seed=seed,
        )
    elif loader == "radif":
        from .radif import curate_radif
        clips, report = curate_radif(
            raw_root / "manifest.jsonl" if raw_root.is_dir() else raw_root,
            window_sec=window_sec,
            max_windows_per_track=max_windows_per_track,
            max_negative_instruments=max_negative_instruments,
            sample_rate=sample_rate, seed=seed,
        )
    else:
        raise RuntimeError(f"Unhandled loader: {loader}")

    out_dir = Path(out_dir) / tradition
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_path = out_dir / "clips.jsonl"
    save_clip_items(clips, clips_path)

    coverage = coverage_summary(clips)
    audit = {**report, "coverage": coverage}
    atomic_write_json(out_dir / "audit_report.json", audit)

    log.info(
        "Curated %d clips from %s → %s (coverage: %s tiers)",
        len(clips), source, clips_path, coverage["tiers"],
    )
    return {
        "source": source,
        "tradition": tradition,
        "clips_path": str(clips_path),
        "audit_path": str(out_dir / "audit_report.json"),
        "coverage": coverage,
    }


def merge_all_clips(curated_root: Path) -> List[ClipItem]:
    """Load every clips.jsonl under curated_root and return merged list."""
    from ..schema import load_clip_items
    out: List[ClipItem] = []
    for p in Path(curated_root).rglob("clips.jsonl"):
        out.extend(load_clip_items(p))
    return out


def audit_all(curated_root: Path) -> dict:
    """Top-level audit: union coverage and license summary."""
    clips = merge_all_clips(curated_root)
    cov = coverage_summary(clips)
    # Also list per-tradition breakdowns by loading each file.
    per_tradition = {}
    for p in Path(curated_root).rglob("clips.jsonl"):
        tradition = p.parent.name
        from ..schema import load_clip_items
        t_clips = load_clip_items(p)
        per_tradition[tradition] = coverage_summary(t_clips)
    return {"total": cov, "per_tradition": per_tradition}


def make_splits(curated_root: Path, out_path: Path,
                eval_fraction: float = 0.6,
                retrieval_fraction: float = 0.3,
                max_clips_per_artist: int = 3,
                seed: int = 848) -> dict:
    clips = merge_all_clips(curated_root)
    split = artist_disjoint_split(
        clips,
        eval_fraction=eval_fraction,
        retrieval_fraction=retrieval_fraction,
        max_clips_per_artist=max_clips_per_artist,
        seed=seed,
    )
    atomic_write_json(out_path, split)
    log.info("Splits written → %s  (%d eval, %d retrieval, %d held-back)",
             out_path, len(split["eval"]), len(split["retrieval"]),
             len(split["held_back"]))
    return split
