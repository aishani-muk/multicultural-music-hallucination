"""MuChoMusic loader — HuggingFace datasets adapter.

MuChoMusic is a multiple-choice music understanding benchmark:
    https://huggingface.co/datasets/mulab-mir/muchomusic

Each example has an audio clip, a question, and 4 candidate answers with
exactly one correct choice. This loader exposes it in a form that the
`verify` layer can consume uniformly with our yes/no atomic facts, plus a
native MCQ evaluator in `mhb.evaluate.muchomusic_eval`.

Requires `pip install datasets` and network access on first use (unless
`HF_DATASETS_CACHE` already has it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

log = logging.getLogger("mhb.data.muchomusic")


@dataclass
class MCQItem:
    item_id: str
    audio_path: str
    question: str
    choices: List[str]
    correct_index: int          # 0..3
    metadata: dict = field(default_factory=dict)


DEFAULT_REPO = "mulab-mir/muchomusic"

def load_muchomusic(
    hf_repo: str = DEFAULT_REPO,
    split: str = "test",
    max_items: Optional[int] = None,
    cache_dir: Optional[str] = None,
) -> List[MCQItem]:
    """Load MuChoMusic from HuggingFace datasets. Returns MCQItems."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets package required: pip install datasets"
        ) from e

    ds = load_dataset(hf_repo, split=split, cache_dir=cache_dir)
    out: List[MCQItem] = []
    for i, ex in enumerate(ds):
        if max_items is not None and i >= max_items:
            break
        audio = ex.get("audio") or {}
        audio_path = audio.get("path") if isinstance(audio, dict) else str(audio)
        choices = ex.get("choices") or [
            ex.get("answer_a"), ex.get("answer_b"),
            ex.get("answer_c"), ex.get("answer_d"),
        ]
        choices = [c for c in choices if c is not None]
        correct = ex.get("correct_index")
        if correct is None:
            correct_letter = (ex.get("correct_answer") or ex.get("label") or "").strip().upper()
            correct = {"A": 0, "B": 1, "C": 2, "D": 3}.get(correct_letter, 0)
        out.append(MCQItem(
            item_id=str(ex.get("id", f"mcm_{i:06d}")),
            audio_path=str(audio_path) if audio_path else "",
            question=str(ex.get("question", "")),
            choices=[str(c) for c in choices],
            correct_index=int(correct),
            metadata={k: v for k, v in ex.items()
                      if k not in ("audio", "question", "choices",
                                   "correct_index", "correct_answer", "label",
                                   "answer_a", "answer_b", "answer_c", "answer_d")},
        ))
    log.info("Loaded %d MuChoMusic items (split=%s)", len(out), split)
    return out
