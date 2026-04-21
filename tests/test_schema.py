from pathlib import Path

from mhb.schema import (
    AtomicFact, BenchmarkPrompt, ClipItem, GroundTruth, InstrumentTag,
    MetricCycle, ModalLabel, SectionSpan, load_clip_items, save_clip_items,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_clipitem_roundtrip(tmp_path):
    clip = ClipItem(
        clip_id="t_001", tradition="hindustani",
        audio_path="a.wav", sr=16000,
        start_sec=0.0, end_sec=10.0,
        tier="B",
        ground_truth=GroundTruth(
            modal_system=ModalLabel(name="yaman", system="raga_hindustani"),
            metric_cycle=MetricCycle(name="teental", system="tala_hindustani", beats=16),
            form="khayal",
            sections=[SectionSpan(label="bandish", start_sec=0.0, end_sec=10.0)],
            instruments=[InstrumentTag(canonical="sitar"),
                         InstrumentTag(canonical="tabla")],
        ),
        derived_facts=[AtomicFact(
            fact_id="f1", category="instrument", claim="sitar present",
            verification_question="Is sitar audible?",
            expected_answer=True, difficulty="must_have",
        )],
    )
    out = tmp_path / "clips.jsonl"
    save_clip_items([clip], out)
    loaded = load_clip_items(out)
    assert len(loaded) == 1
    got = loaded[0]
    assert got.clip_id == "t_001"
    assert got.tradition == "hindustani"
    assert got.ground_truth.modal_system.name == "yaman"
    assert got.ground_truth.metric_cycle.beats == 16
    assert got.ground_truth.sections[0].label == "bandish"
    assert [i.canonical for i in got.ground_truth.instruments] == ["sitar", "tabla"]
    assert got.derived_facts[0].expected_answer is True


def test_benchmarkprompt_from_dict():
    bp = BenchmarkPrompt.from_dict({
        "prompt_id": "p1", "tradition": "carnatic",
        "generation_prompt": "...",
        "description": "",
        "atomic_facts": [{
            "fact_id": "x", "category": "instrument",
            "claim": "c", "verification_question": "q",
            "expected_answer": True, "difficulty": "must_have",
        }],
        "reference_metadata": {},
    })
    assert bp.prompt_id == "p1"
    assert len(bp.atomic_facts) == 1
    assert bp.must_have_facts[0].expected_answer is True


def test_fixture_clips_load():
    clips = load_clip_items(FIXTURES / "clips_mini.jsonl")
    assert len(clips) == 2
    hindustani = clips[0]
    assert hindustani.ground_truth.modal_system.name == "yaman"
    assert any(i.canonical == "tabla" for i in hindustani.ground_truth.instruments)
