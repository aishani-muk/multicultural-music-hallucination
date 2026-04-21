from mhb.data.segmentation import (
    assign_tier, fixed_stride_windows, is_tier_sufficient, section_aligned_windows,
    segment_track,
)
from mhb.data.splits import artist_disjoint_split, coverage_summary
from mhb.schema import (
    ClipItem, GroundTruth, InstrumentTag, MetricCycle, ModalLabel, SectionSpan,
    LicenseTag,
)


def _make_clip(clip_id, artist, tradition="hindustani", tier="B"):
    gt = GroundTruth(
        modal_system=ModalLabel(name="yaman", system="raga_hindustani"),
        metric_cycle=MetricCycle(name="teental", system="tala_hindustani", beats=16),
        instruments=[InstrumentTag(canonical="sitar"),
                     InstrumentTag(canonical="tabla")],
        artist_id=artist,
    )
    return ClipItem(
        clip_id=clip_id, tradition=tradition,
        audio_path="x.wav", sr=16000,
        start_sec=0.0, end_sec=25.0, tier=tier,
        ground_truth=gt, derived_facts=[],
        license=LicenseTag(spdx="CC0"),
    )


def test_assign_tier_levels():
    gt_c = GroundTruth(modal_system=ModalLabel(name="yaman", system="raga_hindustani"),
                       instruments=[InstrumentTag(canonical="sitar")])
    assert assign_tier(gt_c) == "C"

    gt_b = GroundTruth(
        modal_system=ModalLabel(name="yaman", system="raga_hindustani"),
        metric_cycle=MetricCycle(name="teental", system="tala_hindustani", beats=16),
        instruments=[InstrumentTag(canonical="sitar"),
                     InstrumentTag(canonical="tabla")],
    )
    assert assign_tier(gt_b) == "B"

    gt_a = GroundTruth(
        modal_system=ModalLabel(name="yaman", system="raga_hindustani"),
        metric_cycle=MetricCycle(name="teental", system="tala_hindustani", beats=16),
        form="khayal",
        sections=[SectionSpan(label="bandish", start_sec=0, end_sec=10)],
        instruments=[InstrumentTag(canonical="sitar"),
                     InstrumentTag(canonical="tabla")],
    )
    assert assign_tier(gt_a) == "A"


def test_fixed_windows_deterministic():
    a = fixed_stride_windows(600, window_sec=30, stride_sec=60,
                             edge_skip_sec=30, max_windows=4, seed=42)
    b = fixed_stride_windows(600, window_sec=30, stride_sec=60,
                             edge_skip_sec=30, max_windows=4, seed=42)
    assert a == b
    assert len(a) == 4
    for s, e, _ in a:
        assert e - s == 30


def test_section_aligned_respects_short_sections():
    sections = [SectionSpan(label="alap", start_sec=0, end_sec=5),
                SectionSpan(label="bandish", start_sec=10, end_sec=80)]
    wins = section_aligned_windows(sections, window_sec=20, min_section_sec=10)
    # 'alap' (5s) dropped; 'bandish' retained.
    assert [w[2] for w in wins] == ["bandish"]


def test_artist_disjoint_split():
    clips = [_make_clip(f"c{i}", artist=f"a{i % 4}") for i in range(20)]
    split = artist_disjoint_split(clips, eval_fraction=0.5,
                                  retrieval_fraction=0.25,
                                  max_clips_per_artist=2, seed=0)
    # Eval + retrieval + held_back partitions all original ids.
    total = sorted(split["eval"] + split["retrieval"] + split["held_back"])
    assert total == sorted(c.clip_id for c in clips)
    # No shared artist across eval and retrieval.
    by_id = {c.clip_id: c.ground_truth.artist_id for c in clips}
    eval_artists = {by_id[i] for i in split["eval"]}
    retr_artists = {by_id[i] for i in split["retrieval"]}
    assert eval_artists.isdisjoint(retr_artists)


def test_coverage_summary_counts():
    clips = [_make_clip(f"c{i}", artist=f"a{i % 3}") for i in range(9)]
    cov = coverage_summary(clips)
    assert cov["n_clips"] == 9
    assert cov["traditions"]["hindustani"] == 9
    assert "yaman" in cov["modal_systems"]
    assert "teental" in cov["metric_cycles"]
