"""Chain-of-Verification (CoVe) wrapper.

Two-pass mitigation:
  1. Ask the inner verifier the original yes/no question.
  2. If the answer is not already an abstention, ask a re-verification
     question that explicitly references the claimed evidence:
     "Your prior answer was X because: '<evidence>'. Listen again. Is that
      evidence actually present in the audio?"
     If the re-check flips to "no", we downgrade the original answer to
     abstention.

This is a cheap-and-simple baseline; a full CoVe setup decomposes facts
and cross-checks, which is left for future work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..verify.base import VerifierResponse


RECHECK_TEMPLATE = (
    "Your prior answer to the yes/no question below was \"{prev}\", "
    "with this stated evidence: \"{evidence}\".\n\n"
    "Prior question: {question}\n\n"
    "Listen to the audio again. Is that stated audible evidence ACTUALLY present? "
    "Respond with the same strict JSON format (true/false/null)."
)


class CoVeVerifier:
    def __init__(self, inner, recheck_when_confident_above: float = 0.0):
        self.inner = inner
        self.name = f"{inner.name}+cove"
        self.min_conf = float(recheck_when_confident_above)

    def verify(
        self,
        audio_path: Union[str, Path],
        question: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> VerifierResponse:
        r1 = self.inner.verify(audio_path, question,
                               start_sec=start_sec, end_sec=end_sec)
        if r1.answer is None or r1.confidence < self.min_conf:
            return r1  # leave abstention as-is
        recheck_q = RECHECK_TEMPLATE.format(
            prev=("yes" if r1.answer else "no"),
            evidence=r1.evidence, question=question,
        )
        r2 = self.inner.verify(audio_path, recheck_q,
                               start_sec=start_sec, end_sec=end_sec)
        # If the re-check says the evidence is NOT present, downgrade.
        if r2.answer is False:
            return VerifierResponse(
                answer=None,
                confidence=min(r1.confidence, r2.confidence),
                evidence=(f"CoVe downgraded: initial='{r1.evidence}'  "
                          f"recheck='{r2.evidence}'"),
                raw=r1.raw + "\n\n---RECHECK---\n" + (r2.raw or ""),
                latency_sec=(r1.latency_sec or 0) + (r2.latency_sec or 0),
            )
        return r1
