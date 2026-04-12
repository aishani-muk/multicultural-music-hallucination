"""
Prompt schema for the Multicultural Music Hallucination Benchmark.

Each prompt carries:
  - A text-to-music generation instruction
  - Tradition / genre metadata
  - Atomic facts split into must-have and easy-to-get-wrong sets
  - Optional reference metadata for ground-truth verification
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class AtomicFact:
    """A single verifiable claim about the expected audio."""
    fact_id: str                        # Unique within the prompt
    category: str                       # instrument | scale | rhythm | structure | timbre | tuning | cultural
    claim: str                          # Natural-language claim, e.g. "A mridangam is present"
    verification_question: str          # Binary question for the verifier
    expected_answer: bool               # True = should be present/correct
    difficulty: str = "must_have"       # must_have | easy_to_get_wrong


@dataclass
class BenchmarkPrompt:
    """One benchmark entry: a generation prompt + its atomic fact annotations."""
    prompt_id: str
    tradition: str                      # carnatic | hindustani | turkish_makam | arab_andalusian | western_control
    generation_prompt: str              # The text fed to MusicGen / Suno / Udio
    description: str                    # Human-readable description of what the prompt expects
    atomic_facts: List[AtomicFact] = field(default_factory=list)
    reference_metadata: dict = field(default_factory=dict)   # Optional ground-truth keys

    @property
    def must_have_facts(self) -> List[AtomicFact]:
        return [f for f in self.atomic_facts if f.difficulty == "must_have"]

    @property
    def easy_to_get_wrong_facts(self) -> List[AtomicFact]:
        return [f for f in self.atomic_facts if f.difficulty == "easy_to_get_wrong"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkPrompt:
        facts = [AtomicFact(**f) for f in d.pop("atomic_facts", [])]
        return cls(atomic_facts=facts, **d)


def load_prompts(path: str | Path) -> List[BenchmarkPrompt]:
    """Load benchmark prompts from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [BenchmarkPrompt.from_dict(p) for p in data]


def save_prompts(prompts: List[BenchmarkPrompt], path: str | Path) -> None:
    """Save benchmark prompts to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([p.to_dict() for p in prompts], f, indent=2, ensure_ascii=False)
