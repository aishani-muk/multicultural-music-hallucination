from pathlib import Path

from mhb.evaluate.analysis import (
    load_scores, pairwise_agreement, run_analysis, scores_to_rows,
)
from mhb.evaluate.scorer import score_verification_file


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_run_analysis_emits_summary_and_csv(tmp_path):
    sa = tmp_path / "a.json"
    sb = tmp_path / "b.json"
    score_verification_file(FIXTURES / "verification_results_gemini.json", sa, bootstrap_n=20)
    score_verification_file(FIXTURES / "verification_results_music_flamingo.json", sb, bootstrap_n=20)

    out = tmp_path / "analysis"
    res = run_analysis({"gemini": sa, "mf": sb}, out)
    assert "pairwise_agreement" in res
    assert (out / "analysis_summary.json").exists()
    # CSV is produced if pandas is available; either way summary exists.
    assert res["verifiers"] == ["gemini", "mf"]


def test_agreement_kappa_in_range(tmp_path):
    sa = tmp_path / "a.json"
    sb = tmp_path / "b.json"
    score_verification_file(FIXTURES / "verification_results_gemini.json", sa, bootstrap_n=20)
    score_verification_file(FIXTURES / "verification_results_music_flamingo.json", sb, bootstrap_n=20)

    a = load_scores(sa); b = load_scores(sb)
    agr = pairwise_agreement(a, b)
    assert 0.0 <= agr["agreement_rate"] <= 1.0
    assert -1.0 <= agr["kappa"] <= 1.0
