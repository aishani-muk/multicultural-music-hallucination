"""Base Verifier protocol + shared JSON-parse + prompt.

All verifiers take (audio_path, yes/no question) and return a `VerifierResponse`
with fields: answer (bool|None), confidence (float), evidence (str).

The verifier layer is decoupled from `ClipItem` / `BenchmarkPrompt`:
the verify orchestrator handles iteration over facts/clips and converts
responses to `FactVerification` records (see `mhb.evaluate.scorer.run_verification`).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Protocol, Union


SYSTEM_PROMPT = """\
You are an expert music analyst. You will be given an audio clip and a yes/no \
question about its musical content. Answer ONLY based on what you can hear in \
the audio. Do NOT use prior knowledge about specific artists, albums, concerts, \
or recordings.

Respond with STRICT JSON in this format (no other text):
{
  "answer": true or false,
  "confidence": 0.0 to 1.0,
  "evidence": "Brief description of the audible evidence supporting your answer"
}

If you genuinely cannot determine the answer from the audio, respond:
{
  "answer": null,
  "confidence": 0.0,
  "evidence": "Insufficient evidence: <reason>"
}
"""


USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Answer with strict JSON as specified in your instructions."
)


@dataclass
class VerifierResponse:
    answer: Optional[bool]          # None = abstained / parse error
    confidence: float
    evidence: str
    raw: str = ""                   # raw model text (for debugging)
    latency_sec: Optional[float] = None


class Verifier(Protocol):
    """Minimal verifier interface."""

    name: str                       # e.g., "gemini/gemini-2.5-flash"

    def verify(
        self,
        audio_path: Union[str, Path],
        question: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> VerifierResponse: ...


# ─────────────────────────────────────────────────────────────────────────
#  JSON extraction
# ─────────────────────────────────────────────────────────────────────────


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_BLOCK_OPEN_RE = re.compile(r"```(?:json)?\s*(\{.*)", re.DOTALL)  # unterminated fence
_LOOSE_JSON_RE = re.compile(r"(\{.*\})", re.DOTALL)
_ANSWER_FIELD_RE = re.compile(
    r'"?answer"?\s*:\s*(true|false|null|"(?:true|false|yes|no|y|n|1|0)")',
    re.IGNORECASE,
)
_CONFIDENCE_FIELD_RE = re.compile(r'"?confidence"?\s*:\s*([0-9]*\.?[0-9]+)', re.IGNORECASE)
_EVIDENCE_FIELD_RE = re.compile(r'"?evidence"?\s*:\s*"((?:[^"\\]|\\.)*)', re.IGNORECASE)


def _coerce_answer(raw) -> Optional[bool]:
    if raw is None or isinstance(raw, bool):
        return raw
    s = str(raw).strip().strip('"').lower()
    if s in ("true", "yes", "y", "1"):
        return True
    if s in ("false", "no", "n", "0"):
        return False
    if s in ("null", "none", ""):
        return None
    return None


def parse_json_response(text: str) -> VerifierResponse:
    """Best-effort parse of a verifier response. Handles:

    (1) strict JSON, (2) fenced ```json blocks (open or closed),
    (3) JSON embedded in prose, (4) *truncated* JSON where the closing
    brace/quote was cut by max_output_tokens — via field-level regex.
    """
    text = (text or "").strip()

    candidates: List[str] = []
    m = _JSON_BLOCK_RE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    m2 = _LOOSE_JSON_RE.search(text)
    if m2:
        candidates.append(m2.group(1))
    # Unterminated fence — try both with and without closing brace added.
    m3 = _JSON_BLOCK_OPEN_RE.search(text)
    if m3:
        body = m3.group(1).rstrip("` \n")
        candidates.extend([body, body + "}"])

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(obj, dict):
            continue
        answer = _coerce_answer(obj.get("answer"))
        try:
            conf = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        evidence = str(obj.get("evidence", "")).strip()[:1000]
        return VerifierResponse(answer=answer, confidence=conf,
                                evidence=evidence, raw=text)

    # Field-level rescue for truncated responses.
    m_a = _ANSWER_FIELD_RE.search(text)
    if m_a:
        answer = _coerce_answer(m_a.group(1))
        m_c = _CONFIDENCE_FIELD_RE.search(text)
        conf = float(m_c.group(1)) if m_c else 0.0
        m_e = _EVIDENCE_FIELD_RE.search(text)
        evidence = m_e.group(1).strip()[:1000] if m_e else "(truncated response)"
        return VerifierResponse(answer=answer, confidence=conf,
                                evidence=evidence, raw=text)

    low = text.lower()
    if "yes" in low and "no" not in low:
        return VerifierResponse(True, 0.25, "heuristic: 'yes' keyword", raw=text)
    if "no" in low and "yes" not in low:
        return VerifierResponse(False, 0.25, "heuristic: 'no' keyword", raw=text)
    return VerifierResponse(
        None, 0.0, f"Parse error: {text[:200]}", raw=text,
    )


# ─────────────────────────────────────────────────────────────────────────
#  Retry wrapper
# ─────────────────────────────────────────────────────────────────────────


def with_retries(fn, max_retries: int = 3, base_wait: float = 2.0,
                 logger=None):
    """Call fn() with exponential backoff; return last exception if all fail."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if logger:
                logger.warning("Verifier call failed (attempt %d/%d): %s",
                               attempt + 1, max_retries, e)
            time.sleep(base_wait ** (attempt + 1))
    raise last_err  # type: ignore[misc]
