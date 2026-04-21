"""MuChoMusic MCQ evaluator.

Reuses the generic yes/no verifier interface by asking one prompt that
includes the 4 candidate answers and instructs the model to pick A/B/C/D.
Scoring is pure accuracy (stratified where metadata allows).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Union

from ..data.muchomusic import MCQItem, load_muchomusic
from ..utils.io import atomic_write_json

log = logging.getLogger("mhb.evaluate.muchomusic_eval")


MCQ_SYSTEM_PROMPT = """\
You are an expert music analyst. You will hear a short audio clip and be asked \
a multiple-choice question with four options (A, B, C, D). Choose the single \
best answer based on what you hear.

Respond with STRICT JSON only:
{
  "answer": "A" or "B" or "C" or "D",
  "confidence": 0.0 to 1.0,
  "evidence": "Brief description of audible evidence"
}
"""


MCQ_USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Options:\n"
    "  A. {a}\n"
    "  B. {b}\n"
    "  C. {c}\n"
    "  D. {d}\n\n"
    "Respond with strict JSON as specified."
)


_LETTER_RE = re.compile(r'"answer"\s*:\s*"?([A-Da-d])"?')


def _parse_mcq(text: str) -> Optional[int]:
    if not text:
        return None
    m = _LETTER_RE.search(text)
    if m:
        return {"a": 0, "b": 1, "c": 2, "d": 3}[m.group(1).lower()]
    # Fallback: scan for "A." / "B." / "(C)" / "final answer: D"
    for letter, idx in (("a", 0), ("b", 1), ("c", 2), ("d", 3)):
        if re.search(rf"\b{letter}\b", text, re.IGNORECASE):
            return idx
    return None


def _ask(verifier, item: MCQItem) -> tuple[Optional[int], str]:
    # Pack MCQ as a single free-form question; parse letter out of response.
    q = MCQ_USER_TEMPLATE.format(
        question=item.question,
        a=(item.choices[0] if len(item.choices) > 0 else ""),
        b=(item.choices[1] if len(item.choices) > 1 else ""),
        c=(item.choices[2] if len(item.choices) > 2 else ""),
        d=(item.choices[3] if len(item.choices) > 3 else ""),
    )
    resp = verifier.verify(item.audio_path, q)
    raw = resp.raw or resp.evidence or ""
    idx = _parse_mcq(raw)
    return idx, raw


def evaluate_mcq(
    verifier,
    items: List[MCQItem],
    out_path: Union[str, Path],
) -> dict:
    n_correct = 0
    rows = []
    for it in items:
        pred, raw = _ask(verifier, it)
        correct = (pred is not None and pred == it.correct_index)
        if correct:
            n_correct += 1
        rows.append({
            "item_id": it.item_id,
            "question": it.question,
            "correct_index": it.correct_index,
            "predicted_index": pred,
            "correct": bool(correct),
            "raw": raw[:600],
        })

    accuracy = n_correct / max(1, len(items))
    payload = {
        "verifier": getattr(verifier, "name", str(verifier)),
        "n_items": len(items),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "items": rows,
    }
    atomic_write_json(out_path, payload)
    log.info("MuChoMusic [%s]: accuracy = %.3f (%d/%d)",
             verifier.name, accuracy, n_correct, len(items))
    return payload


def run_muchomusic(
    verifier,
    out_dir: Union[str, Path],
    split: str = "test",
    max_items: Optional[int] = None,
) -> dict:
    items = load_muchomusic(split=split, max_items=max_items)
    out_path = Path(out_dir) / f"{_safe(verifier.name)}_muchomusic.json"
    return evaluate_mcq(verifier, items, out_path)


def _safe(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")
