"""Multicultural Music-Hallucination Benchmark (mhb).

A dual-track benchmark for measuring hallucination in audio-language models
(Gemini, Music Flamingo, Audio Flamingo 3, Qwen2.5-Omni, Qwen2-Audio) on:

  1. Understanding: real multicultural audio (Saraga, CompMusic, MTG-Jamendo,
     MuChoMusic) with culture-specific ground truth (raga, tala, makam, usul,
     tab', mizan, dastgah, dembow, form, instruments).
  2. Generation: MusicGen outputs checked against culture-grounded atomic
     facts via the same oracle stack.

See README.md + QUICKSTART.md for usage.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
