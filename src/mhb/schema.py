"""Core dataclasses for the benchmark.

Everything downstream (loaders, verifiers, scorer, analysis) consumes these
types, so keep them stable and JSON-serializable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional, Literal, Union


# ═════════════════════════════════════════════════════════════════════════
#  Ground-truth types
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class ModalLabel:
    """Modal/melodic system: raga | makam | tab' | dastgah | (generic)."""
    name: str                       # canonical name in the controlled vocab
    system: str                     # raga_carnatic | raga_hindustani | makam | tab3 | dastgah | western_key | dembow_free
    aliases: List[str] = field(default_factory=list)
    parent: Optional[str] = None    # e.g., melakarta for janya ragas
    notes: Optional[str] = None


@dataclass
class MetricCycle:
    """Metric cycle: tala | usul | mizan | dembow | western_meter."""
    name: str                       # canonical name in controlled vocab
    system: str                     # tala_carnatic | tala_hindustani | usul | mizan | dembow | western
    beats: Optional[int] = None     # total beats per cycle (None = free rhythm / taksim / alap)
    accent_pattern: Optional[str] = None   # e.g., "2+2+2+3" for aksak, "3+3+2" for dembow
    aliases: List[str] = field(default_factory=list)


@dataclass
class SectionSpan:
    """A labeled time range within the source recording."""
    label: str                      # alap | vilambit | drut | bandish | kriti | alapana | taksim | ...
    start_sec: float
    end_sec: float


@dataclass
class InstrumentTag:
    """A single instrument attribution with optional confidence."""
    canonical: str                  # from vocab/instruments.yaml
    source: Literal["metadata", "tag", "auto"] = "metadata"
    confidence: Optional[float] = None   # [0, 1] if available


@dataclass
class GroundTruth:
    """Culture-specific ground truth for a single clip.

    All fields are optional so loaders can populate progressively; tier
    assignment (see data.segmentation.assign_tier) encodes how complete
    this is.
    """
    modal_system: Optional[ModalLabel] = None
    metric_cycle: Optional[MetricCycle] = None
    form: Optional[str] = None
    sections: List[SectionSpan] = field(default_factory=list)
    instruments: List[InstrumentTag] = field(default_factory=list)
    tonic_hz: Optional[float] = None
    tempo_bpm: Optional[float] = None
    language: Optional[str] = None
    artist_id: Optional[str] = None
    recording_id: Optional[str] = None
    provenance: dict = field(default_factory=dict)   # loader-specific raw metadata pointer


@dataclass
class LicenseTag:
    """License + rights provenance for a single clip."""
    spdx: str                       # e.g., "CC-BY-NC-SA-4.0" | "restricted" | "unknown"
    source_url: Optional[str] = None
    redistribute: bool = False      # True iff license allows redistribution


# ═════════════════════════════════════════════════════════════════════════
#  Atomic facts (prompts + derived-from-GT)
# ═════════════════════════════════════════════════════════════════════════


FactCategory = Literal[
    "instrument", "scale", "rhythm", "structure", "timbre",
    "tuning", "cultural", "modal_system", "metric_cycle",
    "form", "tonic", "tempo", "language",
]

FactDifficulty = Literal[
    "must_have",               # required positive: presence of canonical attribute
    "culture_discriminative",  # positive that distinguishes the tradition
    "easy_to_get_wrong",       # negative: out-of-culture distractor
    "expert_only",             # needs expert judgment; excluded from default scoring
]


@dataclass
class AtomicFact:
    """A single verifiable yes/no claim about the audio."""
    fact_id: str
    category: str                   # FactCategory (kept str for legacy JSON)
    claim: str
    verification_question: str
    expected_answer: bool
    difficulty: str = "must_have"   # FactDifficulty
    source: str = "manual"          # manual | auto:<template_id>
    grounding: Optional[str] = None # e.g., "ground_truth.modal_system.name"


# ═════════════════════════════════════════════════════════════════════════
#  Benchmark prompts (generation track) and clip items (understanding track)
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkPrompt:
    """One generation-track entry: a text prompt + atomic-fact annotations."""
    prompt_id: str
    tradition: str
    generation_prompt: str
    description: str = ""
    atomic_facts: List[AtomicFact] = field(default_factory=list)
    reference_metadata: dict = field(default_factory=dict)

    @property
    def must_have_facts(self) -> List[AtomicFact]:
        return [f for f in self.atomic_facts if f.difficulty == "must_have"]

    @property
    def easy_to_get_wrong_facts(self) -> List[AtomicFact]:
        return [f for f in self.atomic_facts if f.difficulty == "easy_to_get_wrong"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkPrompt":
        d = dict(d)
        facts = [AtomicFact(**f) for f in d.pop("atomic_facts", [])]
        return cls(atomic_facts=facts, **d)


@dataclass
class ClipItem:
    """One understanding-track entry: an audio clip + culture-specific GT + derived facts."""
    clip_id: str
    tradition: str
    audio_path: str                     # absolute path on disk
    sr: int
    start_sec: float
    end_sec: float
    tier: str                           # "A" | "B" | "C"
    ground_truth: GroundTruth = field(default_factory=GroundTruth)
    derived_facts: List[AtomicFact] = field(default_factory=list)
    license: LicenseTag = field(default_factory=lambda: LicenseTag(spdx="unknown"))
    source_corpus: str = "unknown"
    audit: dict = field(default_factory=dict)   # clip-QC results

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClipItem":
        d = dict(d)
        gt = d.pop("ground_truth", {}) or {}
        lic = d.pop("license", {}) or {}
        facts = d.pop("derived_facts", []) or []

        # ground_truth nested
        ml = gt.get("modal_system")
        mc = gt.get("metric_cycle")
        gt_obj = GroundTruth(
            modal_system=ModalLabel(**ml) if ml else None,
            metric_cycle=MetricCycle(**mc) if mc else None,
            form=gt.get("form"),
            sections=[SectionSpan(**s) for s in gt.get("sections", [])],
            instruments=[InstrumentTag(**i) for i in gt.get("instruments", [])],
            tonic_hz=gt.get("tonic_hz"),
            tempo_bpm=gt.get("tempo_bpm"),
            language=gt.get("language"),
            artist_id=gt.get("artist_id"),
            recording_id=gt.get("recording_id"),
            provenance=gt.get("provenance", {}),
        )
        return cls(
            ground_truth=gt_obj,
            derived_facts=[AtomicFact(**f) for f in facts],
            license=LicenseTag(**lic) if lic else LicenseTag(spdx="unknown"),
            **d,
        )


# ═════════════════════════════════════════════════════════════════════════
#  Verification results
# ═════════════════════════════════════════════════════════════════════════


@dataclass
class FactVerification:
    """Result of verifying one atomic fact against one clip by one verifier."""
    fact_id: str
    category: str
    claim: str
    verification_question: str
    expected_answer: bool
    difficulty: str
    model_answer: Optional[bool]     # None = abstained / parse error
    confidence: float
    evidence: str
    correct: Optional[bool]          # None = abstained
    verifier: str                    # "gemini/gemini-2.5-flash", "music_flamingo/…", etc.
    latency_sec: Optional[float] = None


@dataclass
class ClipVerification:
    """All fact-level verifications for a single clip by a single verifier."""
    clip_id: str
    tradition: str
    audio_path: str
    verifier: str
    fact_results: List[FactVerification] = field(default_factory=list)
    # Optional pointers back to the source of the facts
    source: Literal["prompt", "clip"] = "clip"


# ═════════════════════════════════════════════════════════════════════════
#  JSON I/O helpers
# ═════════════════════════════════════════════════════════════════════════


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/Paths to JSON-safe primitives."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return _to_jsonable(asdict(obj))
    return str(obj)


def dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(_to_jsonable(obj), indent=indent, ensure_ascii=False)


def dump_json(obj: Any, path: Union[str, Path], indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(dumps(obj, indent=indent))
    tmp.replace(path)
    return path


def load_json(path: Union[str, Path]) -> Any:
    with open(path) as f:
        return json.load(f)


def load_prompts(path: Union[str, Path]) -> List[BenchmarkPrompt]:
    data = load_json(path)
    return [BenchmarkPrompt.from_dict(p) for p in data]


def save_prompts(prompts: List[BenchmarkPrompt], path: Union[str, Path]) -> Path:
    return dump_json([p.to_dict() for p in prompts], path)


def load_clip_items(path: Union[str, Path]) -> List[ClipItem]:
    """Load a JSONL file of ClipItems."""
    items: List[ClipItem] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(ClipItem.from_dict(json.loads(line)))
    return items


def save_clip_items(items: List[ClipItem], path: Union[str, Path]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for ci in items:
            f.write(dumps(ci, indent=None) + "\n")
    tmp.replace(path)
    return path
