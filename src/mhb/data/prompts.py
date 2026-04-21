"""Thin re-export of prompt IO from the schema module.

Kept for readability in downstream modules that work with
`BenchmarkPrompt` objects (generation track).
"""

from __future__ import annotations

from ..schema import (  # noqa: F401
    AtomicFact, BenchmarkPrompt, load_prompts, save_prompts,
)
