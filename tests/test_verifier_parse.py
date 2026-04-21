from mhb.verify.base import parse_json_response


def test_parse_strict_json_true():
    r = parse_json_response('{"answer": true, "confidence": 0.8, "evidence": "x"}')
    assert r.answer is True
    assert r.confidence == 0.8
    assert r.evidence == "x"


def test_parse_strict_json_null():
    r = parse_json_response('{"answer": null, "confidence": 0, "evidence": "unsure"}')
    assert r.answer is None


def test_parse_fenced_json():
    r = parse_json_response('```json\n{"answer": false, "confidence": 0.5, "evidence": "e"}\n```')
    assert r.answer is False


def test_parse_loose_json_with_preamble():
    text = 'Sure, my answer is: {"answer": true, "confidence": 0.7, "evidence": "e"} thanks.'
    r = parse_json_response(text)
    assert r.answer is True


def test_parse_string_answer_keywords():
    r = parse_json_response('{"answer": "yes", "confidence": 0.6, "evidence": "e"}')
    assert r.answer is True


def test_parse_garbage_returns_abstain():
    r = parse_json_response("completely unparseable text without json")
    assert r.answer is None
    assert r.confidence == 0.0


def test_parse_truncated_fenced_json_recovers_answer():
    # Mimics Gemini hitting max_output_tokens mid-response.
    truncated = '```json\n{\n  "answer": null,\n  "confidence": 0.0,\n  "evidence": "The clip is too short'
    r = parse_json_response(truncated)
    assert r.answer is None
    assert "Parse error" not in r.evidence


def test_parse_truncated_json_with_false_answer():
    truncated = '{"answer": false, "confidence": 0.85, "evidence": "No tabla; only a sustained tone'
    r = parse_json_response(truncated)
    assert r.answer is False
    assert r.confidence == 0.85
    assert "Parse error" not in r.evidence


def test_parse_fenced_true_with_trailing_prose():
    text = 'Sure!\n```json\n{"answer": true, "confidence": 0.9, "evidence": "e"}\n```\nHope that helps.'
    r = parse_json_response(text)
    assert r.answer is True
    assert r.confidence == 0.9
