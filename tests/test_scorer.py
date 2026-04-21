import json
from pathlib import Path

from mhb.evaluate.scorer import score_clip, aggregate, score_verification_file


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _fact(correct, difficulty="must_have", category="instrument"):
    return {
        "fact_id": f"f_{correct}_{difficulty}",
        "category": category,
        "claim": "x", "verification_question": "q?",
        "expected_answer": True, "difficulty": difficulty,
        "model_answer": (True if correct is True else (False if correct is False else None)),
        "confidence": 0.9 if correct is not None else 0.0,
        "evidence": "",
        "correct": correct,
    }


def test_score_single_clip_rates():
    clip = {
        "clip_id": "c1", "tradition": "hindustani",
        "verifier": "x", "source": "clip",
        "fact_results": [
            _fact(True, "must_have"), _fact(True, "must_have"),
            _fact(False, "must_have"),
            _fact(None, "must_have"),
            _fact(False, "easy_to_get_wrong"),
            _fact(True, "culture_discriminative"),
        ],
    }
    s = score_clip(clip)
    # 3 verified of must_have (2 correct), 1 abstained.
    assert s["n_verified"] == 5
    assert s["n_correct"] == 3   # two must_have + 1 culture_disc
    assert s["n_incorrect"] == 2  # one must_have wrong + one easy_wrong
    assert s["n_abstained"] == 1
    assert abs(s["correctness_rate"] - 3/5) < 1e-9
    assert abs(s["hallucination_rate"] - 2/5) < 1e-9
    # must_have subset: 2 correct, 1 incorrect, 1 abstain -> correctness = 2/3
    assert abs(s["must_have_correctness"] - 2/3) < 1e-9


def test_aggregate_handles_empty_gracefully():
    assert aggregate([])["n_clips"] == 0
    assert aggregate([{"error": "x"}])["n_clips"] == 0


def test_fixture_scoring_end_to_end(tmp_path):
    out = tmp_path / "scores.json"
    payload = score_verification_file(
        FIXTURES / "verification_results_gemini.json", out,
        bootstrap_n=50, seed=0,
    )
    assert out.exists()
    assert payload["aggregate"]["n_clips"] == 2
    overall = payload["aggregate"]["overall"]
    assert 0.0 <= overall["correctness_rate"]["mean"] <= 1.0
    assert "ci_low" in overall["correctness_rate"]
    assert "ci_high" in overall["correctness_rate"]
