"""Controlled vocabularies for modal systems, metric cycles, instruments, languages.

Each vocab file is a YAML mapping `canonical_name -> {aliases, ...fields}`.
Loaders here normalize free-form metadata strings to canonical names.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

VOCAB_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_vocab(name: str) -> Dict[str, Dict[str, Any]]:
    """Load a YAML vocab by basename (without extension)."""
    path = VOCAB_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    # Normalize: ensure every entry has an 'aliases' list
    for k, v in data.items():
        if v is None:
            data[k] = {"aliases": []}
        elif "aliases" not in v:
            v["aliases"] = []
    return data


def normalize(name: str, vocab: Dict[str, Dict[str, Any]]) -> Optional[str]:
    """Return the canonical vocab key for a free-form name, or None if unknown.

    Matches case-insensitively against canonical keys and aliases; strips
    whitespace and common diacritic variants.
    """
    if not name:
        return None
    n = _simplify(name)
    for canonical, entry in vocab.items():
        if _simplify(canonical) == n:
            return canonical
        for a in entry.get("aliases") or []:
            if _simplify(a) == n:
                return canonical
    return None


def _simplify(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower().strip()
    for junk in ["-", "_", ".", " ", "'", "`", "\u2019"]:
        s = s.replace(junk, "")
    return s


def list_vocabs() -> List[str]:
    return sorted(p.stem for p in VOCAB_DIR.glob("*.yaml"))
