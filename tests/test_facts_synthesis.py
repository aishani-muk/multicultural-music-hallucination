from mhb.data.facts import synthesize_facts
from mhb.schema import GroundTruth, InstrumentTag, MetricCycle, ModalLabel


def _hindustani_gt():
    return GroundTruth(
        modal_system=ModalLabel(name="yaman", system="raga_hindustani",
                                aliases=["kalyan"]),
        metric_cycle=MetricCycle(name="teental", system="tala_hindustani",
                                 beats=16, accent_pattern="4+4+4+4"),
        form="khayal",
        instruments=[InstrumentTag(canonical="sitar"),
                     InstrumentTag(canonical="tabla"),
                     InstrumentTag(canonical="tanpura")],
    )


def test_synthesize_emits_positive_modal_and_cycle():
    gt = _hindustani_gt()
    facts = synthesize_facts("hindustani", gt)
    # Should have modal_system and metric_cycle positives.
    assert any(f.category == "modal_system" and f.expected_answer for f in facts)
    assert any(f.category == "metric_cycle" and f.expected_answer for f in facts)


def test_synthesize_per_instrument_positive():
    gt = _hindustani_gt()
    facts = synthesize_facts("hindustani", gt)
    inst_names = {f.grounding.split("[")[-1].rstrip("]")
                  for f in facts
                  if f.source == "auto:instrument_present"}
    assert {"sitar", "tabla", "tanpura"}.issubset(inst_names)


def test_synthesize_negative_instruments_out_of_culture():
    gt = _hindustani_gt()
    facts = synthesize_facts("hindustani", gt, max_negative_instruments=3)
    negs = [f for f in facts if f.source == "auto:instrument_negative"]
    assert len(negs) == 3
    for f in negs:
        assert f.expected_answer is False
        assert f.difficulty == "easy_to_get_wrong"


def test_culture_discriminative_facts_present():
    gt = _hindustani_gt()
    facts = synthesize_facts("hindustani", gt)
    kinds = {f.source for f in facts}
    # meend + tanpura drone templates must apply.
    assert "auto:template:tanpura_drone" in kinds
    assert "auto:template:meend_ornamentation" in kinds


def test_fact_ids_unique():
    gt = _hindustani_gt()
    facts = synthesize_facts("hindustani", gt)
    ids = [f.fact_id for f in facts]
    assert len(ids) == len(set(ids))
