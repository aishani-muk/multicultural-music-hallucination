"""Hybrid segmentation: section-aligned when annotated, else fixed windows.

Also defines tier assignment (A/B/C) based on how much GT is grounded.
"""

from __future__ import annotations

import random
from typing import List, Optional

from ..schema import ClipItem, GroundTruth, SectionSpan


# ─────────────────────────────────────────────────────────────────────────
#  Tier assignment
# ─────────────────────────────────────────────────────────────────────────


def assign_tier(gt: GroundTruth) -> str:
    """Return 'A' | 'B' | 'C' based on GT completeness."""
    has_modal = gt.modal_system is not None
    has_cycle = gt.metric_cycle is not None and gt.metric_cycle.beats is not None
    has_form = bool(gt.form)
    n_instruments = len(gt.instruments)
    has_sections = len(gt.sections) > 0

    if has_modal and has_cycle and has_form and n_instruments >= 2 and has_sections:
        return "A"
    if has_modal and has_cycle and n_instruments >= 2:
        return "B"
    if (has_modal or has_cycle) and n_instruments >= 1:
        return "C"
    return "C"  # will be dropped by curation if below min threshold


def is_tier_sufficient(gt: GroundTruth) -> bool:
    """Curation drops clips below this threshold (weaker than Tier C)."""
    has_modal = gt.modal_system is not None
    has_cycle = gt.metric_cycle is not None
    has_instruments = len(gt.instruments) >= 1
    return (has_modal or has_cycle) and has_instruments


# ─────────────────────────────────────────────────────────────────────────
#  Windowing
# ─────────────────────────────────────────────────────────────────────────


def section_aligned_windows(
    sections: List[SectionSpan],
    window_sec: float = 25.0,
    min_section_sec: float = 10.0,
) -> List[tuple[float, float, Optional[str]]]:
    """Extract one representative window per annotated section.

    Returns list of (start_sec, end_sec, section_label).
    """
    out = []
    for sec in sections:
        dur = sec.end_sec - sec.start_sec
        if dur < min_section_sec:
            continue
        # Center the window in the section, clipped to section bounds.
        if dur <= window_sec:
            out.append((sec.start_sec, sec.end_sec, sec.label))
        else:
            mid = (sec.start_sec + sec.end_sec) / 2.0
            s = max(sec.start_sec, mid - window_sec / 2.0)
            e = min(sec.end_sec, s + window_sec)
            out.append((float(s), float(e), sec.label))
    return out


def fixed_stride_windows(
    total_duration_sec: float,
    window_sec: float = 30.0,
    stride_sec: float = 60.0,
    edge_skip_sec: float = 30.0,
    max_windows: int = 4,
    seed: int = 848,
) -> List[tuple[float, float, Optional[str]]]:
    """Fixed windows with stride; first/last edge_skip trimmed; capped at max_windows."""
    if total_duration_sec <= 2 * edge_skip_sec + window_sec:
        return []

    rng = random.Random(seed)
    starts = []
    t = edge_skip_sec
    while t + window_sec <= total_duration_sec - edge_skip_sec:
        starts.append(t)
        t += stride_sec
    if len(starts) > max_windows:
        # Deterministic downsample: evenly spaced pick.
        indices = [int(round(i * (len(starts) - 1) / (max_windows - 1)))
                   for i in range(max_windows)]
        starts = [starts[i] for i in indices]
    return [(s, s + window_sec, None) for s in starts]


def segment_track(
    total_duration_sec: float,
    sections: List[SectionSpan],
    window_sec: float = 25.0,
    stride_sec: float = 60.0,
    edge_skip_sec: float = 30.0,
    max_windows: int = 4,
    seed: int = 848,
) -> List[tuple[float, float, Optional[str]]]:
    """Hybrid: section-aligned if sections exist AND usable, else fixed windows."""
    if sections:
        wins = section_aligned_windows(sections, window_sec=window_sec)
        if wins:
            return wins[:max_windows]
    return fixed_stride_windows(
        total_duration_sec=total_duration_sec,
        window_sec=window_sec,
        stride_sec=stride_sec,
        edge_skip_sec=edge_skip_sec,
        max_windows=max_windows,
        seed=seed,
    )
