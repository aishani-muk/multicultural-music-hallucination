from mhb.mitigations.abstain import AbstainWrapper
from mhb.mitigations.cove import CoVeVerifier
from mhb.mitigations.rarr import RARRVerifier, NoOpRetriever
from mhb.verify.base import VerifierResponse


class _Scripted:
    """Mock verifier returning canned responses."""
    def __init__(self, responses):
        self._resp = list(responses)
        self.name = "scripted"
        self.calls = 0

    def verify(self, audio_path, question, start_sec=None, end_sec=None):
        r = self._resp[min(self.calls, len(self._resp) - 1)]
        self.calls += 1
        return r


def test_abstain_wrapper_thresholds():
    inner = _Scripted([VerifierResponse(True, 0.2, "e")])
    w = AbstainWrapper(inner, threshold=0.5)
    r = w.verify("a.wav", "q?")
    assert r.answer is None
    assert "Abstained" in r.evidence


def test_abstain_wrapper_passes_confident():
    inner = _Scripted([VerifierResponse(True, 0.9, "e")])
    w = AbstainWrapper(inner, threshold=0.5)
    r = w.verify("a.wav", "q?")
    assert r.answer is True


def test_cove_downgrades_on_recheck_negative():
    inner = _Scripted([
        VerifierResponse(True, 0.8, "I hear a tabla"),
        VerifierResponse(False, 0.7, "actually no tabla present"),
    ])
    w = CoVeVerifier(inner)
    r = w.verify("a.wav", "Is there tabla?")
    assert r.answer is None  # downgraded


def test_cove_keeps_when_recheck_confirms():
    inner = _Scripted([
        VerifierResponse(True, 0.8, "tabla"),
        VerifierResponse(True, 0.9, "confirmed tabla"),
    ])
    w = CoVeVerifier(inner)
    r = w.verify("a.wav", "q?")
    assert r.answer is True


def test_rarr_noop_retriever_passes_through():
    inner = _Scripted([VerifierResponse(True, 0.8, "e")])
    w = RARRVerifier(inner, retriever=NoOpRetriever())
    r = w.verify("a.wav", "q?")
    assert r.answer is True
    assert inner.calls == 1
