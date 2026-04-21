"""Confidence-threshold abstention wrapper.

Wraps any `Verifier` so that responses with confidence below `threshold`
are converted to abstentions (answer=None). Useful as a baseline
"abstain-when-unsure" mitigation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ..verify.base import VerifierResponse


class AbstainWrapper:
    def __init__(self, inner, threshold: float = 0.5):
        self.inner = inner
        self.threshold = float(threshold)
        self.name = f"{inner.name}+abstain@{threshold:.2f}"

    def verify(
        self,
        audio_path: Union[str, Path],
        question: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> VerifierResponse:
        r = self.inner.verify(audio_path, question,
                              start_sec=start_sec, end_sec=end_sec)
        if r.answer is not None and r.confidence < self.threshold:
            return VerifierResponse(
                answer=None,
                confidence=r.confidence,
                evidence=(f"Abstained (conf {r.confidence:.2f} < {self.threshold:.2f}). "
                          f"Original: {r.evidence}"),
                raw=r.raw,
                latency_sec=r.latency_sec,
            )
        return r
