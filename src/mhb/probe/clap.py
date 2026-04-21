"""LAION-CLAP zero-shot probe — a model-independent yes/no signal for
instrument and genre presence facts.

Usage idea: for an instrument fact "Is a tabla audible?" we compute the
CLAP similarity between the clip's audio embedding and the text embedding
of "a recording with tabla" vs "a recording without tabla", and threshold.

This probe is *optional*: it requires `laion-clap` to be installed. If the
package is missing, `is_available()` returns False and the CLI skips it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from ..schema import AtomicFact
from ..verify.base import VerifierResponse

log = logging.getLogger("mhb.probe.clap")


def is_available() -> bool:
    try:
        import laion_clap  # noqa: F401
        return True
    except ImportError:
        return False


class ClapProbe:
    """Zero-shot binary audio-text matcher using LAION-CLAP."""

    def __init__(
        self,
        ckpt: Optional[str] = None,          # path to a LAION-CLAP ckpt
        device: str = "cuda",
        threshold: float = 0.05,             # margin on (pos_sim - neg_sim)
    ):
        if not is_available():
            raise ImportError(
                "laion_clap not installed. `pip install laion-clap`."
            )
        import laion_clap
        import torch

        self.device = device
        self.model = laion_clap.CLAP_Module(enable_fusion=False).to(device)
        if ckpt:
            self.model.load_ckpt(ckpt)
        else:
            self.model.load_ckpt()  # default HF ckpt
        self.threshold = threshold
        self.name = f"probe/clap"

    def _embed_audio(self, audio_path: Path,
                     start_sec: Optional[float],
                     end_sec: Optional[float]) -> np.ndarray:
        import librosa
        offset = float(start_sec or 0.0)
        duration = (float(end_sec) - offset) if end_sec is not None else None
        y, sr = librosa.load(str(audio_path), sr=48000, mono=True,
                             offset=offset, duration=duration)
        emb = self.model.get_audio_embedding_from_data(
            x=y[None, :], use_tensor=False,
        )
        return np.asarray(emb).reshape(-1)

    def _embed_text(self, text: str) -> np.ndarray:
        emb = self.model.get_text_embedding([text], use_tensor=False)
        return np.asarray(emb).reshape(-1)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        an = np.linalg.norm(a)
        bn = np.linalg.norm(b)
        if an == 0 or bn == 0:
            return 0.0
        return float(np.dot(a, b) / (an * bn))

    def verify(
        self,
        audio_path: Union[str, Path],
        question: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> VerifierResponse:
        """Map yes/no question to a positive/negative text prompt pair.

        Heuristic: the probe treats `question` as asking about a concept; the
        positive probe is the question's affirmative form, negative is its
        explicit denial.
        """
        pos_text = question.replace("?", "").strip()
        neg_text = f"a recording where {pos_text} is NOT true"
        pos_text = f"a recording where {pos_text} is true"

        a = self._embed_audio(Path(audio_path), start_sec, end_sec)
        sp = self._embed_text(pos_text)
        sn = self._embed_text(neg_text)
        pos_sim = self._cosine(a, sp)
        neg_sim = self._cosine(a, sn)
        margin = pos_sim - neg_sim
        if margin > self.threshold:
            answer = True
        elif margin < -self.threshold:
            answer = False
        else:
            answer = None   # abstain when indecisive
        confidence = float(min(1.0, abs(margin) * 5.0))
        evidence = (
            f"CLAP cosine: pos={pos_sim:.3f}, neg={neg_sim:.3f}, "
            f"margin={margin:.3f}, threshold={self.threshold:.3f}"
        )
        return VerifierResponse(answer=answer, confidence=confidence,
                                evidence=evidence, raw=evidence)
