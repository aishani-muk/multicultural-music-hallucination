"""Artist-disjoint train/eval/retrieval splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional

from ..schema import ClipItem


def artist_disjoint_split(
    clips: List[ClipItem],
    eval_fraction: float = 0.6,
    retrieval_fraction: float = 0.3,
    max_clips_per_artist: int = 3,
    seed: int = 848,
) -> Dict[str, List[str]]:
    """Split clips by artist, disjoint between eval and retrieval corpora.

    - Clips without an artist_id are treated as unique artists (one per clip).
    - Eval set has per-artist cap; retrieval corpus does not.
    - Any artist that would straddle is forced fully into retrieval.

    Returns {'eval': [...], 'retrieval': [...], 'held_back': [...]}.
    """
    rng = random.Random(seed)

    # Group clips by artist.
    by_artist: Dict[str, List[ClipItem]] = defaultdict(list)
    for c in clips:
        aid = c.ground_truth.artist_id or f"__solo__::{c.clip_id}"
        by_artist[aid].append(c)

    artists = sorted(by_artist.keys())
    rng.shuffle(artists)

    n = len(artists)
    n_eval = max(1, int(round(eval_fraction * n)))
    n_retr = max(1, int(round(retrieval_fraction * n)))

    eval_artists = set(artists[:n_eval])
    retrieval_artists = set(artists[n_eval:n_eval + n_retr])
    held_artists = set(artists[n_eval + n_retr:])

    eval_ids: List[str] = []
    retrieval_ids: List[str] = []
    held_ids: List[str] = []

    for aid, items in by_artist.items():
        items_sorted = sorted(items, key=lambda ci: ci.clip_id)
        if aid in eval_artists:
            # Cap clips per artist.
            keep = items_sorted[:max_clips_per_artist]
            # Overflow goes to retrieval (artist is eval-side, so force overflow to held to keep disjoint).
            overflow = items_sorted[max_clips_per_artist:]
            eval_ids.extend(c.clip_id for c in keep)
            held_ids.extend(c.clip_id for c in overflow)
        elif aid in retrieval_artists:
            retrieval_ids.extend(c.clip_id for c in items_sorted)
        else:
            held_ids.extend(c.clip_id for c in items_sorted)

    return {
        "eval": sorted(eval_ids),
        "retrieval": sorted(retrieval_ids),
        "held_back": sorted(held_ids),
    }


def coverage_summary(clips: List[ClipItem]) -> dict:
    """Compute a coverage/audit summary for `mhb data audit`."""
    n = len(clips)
    tiers = Counter(c.tier for c in clips)
    traditions = Counter(c.tradition for c in clips)
    ragas = Counter()
    talas = Counter()
    forms = Counter()
    instruments = Counter()
    licenses = Counter()
    sr_counts = Counter()
    durations: List[float] = []
    artists = Counter()

    for c in clips:
        if c.ground_truth.modal_system:
            ragas[c.ground_truth.modal_system.name] += 1
        if c.ground_truth.metric_cycle:
            talas[c.ground_truth.metric_cycle.name] += 1
        if c.ground_truth.form:
            forms[c.ground_truth.form] += 1
        for inst in c.ground_truth.instruments:
            instruments[inst.canonical] += 1
        licenses[c.license.spdx] += 1
        sr_counts[c.sr] += 1
        durations.append(c.duration_sec)
        if c.ground_truth.artist_id:
            artists[c.ground_truth.artist_id] += 1

    avg_dur = sum(durations) / max(1, len(durations))
    return {
        "n_clips": n,
        "tiers": dict(tiers.most_common()),
        "traditions": dict(traditions.most_common()),
        "modal_systems": dict(ragas.most_common()),
        "metric_cycles": dict(talas.most_common()),
        "forms": dict(forms.most_common()),
        "instruments": dict(instruments.most_common()),
        "licenses": dict(licenses.most_common()),
        "sr_counts": dict(sr_counts.most_common()),
        "avg_duration_sec": round(avg_dur, 2),
        "n_artists": len(artists),
    }
