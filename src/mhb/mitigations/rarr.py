"""RARR (Retrieval-Augmented Revision) skeleton.

Not runnable end-to-end without a retrieval corpus; left as a stub with a
pluggable retriever interface so future work can drop in a corpus of
culture-specific musical metadata (raga/tala/makam reference pages).

Interface:

    retriever(question: str, k: int) -> List[{"text": ..., "url": ...}]
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Union

from ..verify.base import VerifierResponse


class NoOpRetriever:
    """Returns an empty context. Use as the default when no corpus is available."""

    def __call__(self, question: str, k: int = 3) -> List[dict]:
        return []


RARR_TEMPLATE = (
    "{question}\n\n"
    "Relevant reference context (may or may not apply; use only if directly "
    "relevant to what you hear):\n{ctx}"
)


class RARRVerifier:
    def __init__(self, inner, retriever: Optional[Callable] = None, top_k: int = 3):
        self.inner = inner
        self.retriever = retriever or NoOpRetriever()
        self.top_k = top_k
        self.name = f"{inner.name}+rarr(k={top_k})"

    def verify(
        self,
        audio_path: Union[str, Path],
        question: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> VerifierResponse:
        ctx_items = self.retriever(question, k=self.top_k) or []
        ctx = "\n".join(f"- {c.get('text','')}" for c in ctx_items) or "(no context)"
        q = RARR_TEMPLATE.format(question=question, ctx=ctx)
        return self.inner.verify(audio_path, q,
                                 start_sec=start_sec, end_sec=end_sec)
