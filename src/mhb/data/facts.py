"""Auto-synthesize AtomicFacts from GroundTruth + templates + vocab.

Usage:
    from mhb.data.facts import synthesize_facts
    facts = synthesize_facts(tradition, ground_truth)
"""

from __future__ import annotations

import re
from typing import List, Optional

from .auto_templates import load_templates
from .vocab import load_vocab
from ..schema import AtomicFact, GroundTruth, InstrumentTag


# ─────────────────────────────────────────────────────────────────────────
#  Safe expression evaluation for 'applies_if'
# ─────────────────────────────────────────────────────────────────────────

# Whitelist: comparison/logic + attribute access on a local `gt`.
# No arbitrary builtins.
_ALLOWED_NAMES = {"gt", "True", "False", "None", "any", "all", "len", "range"}
_RE_FORBIDDEN = re.compile(r"\b(import|open|__|eval|exec|lambda|os\.|sys\.)\b")


def _eval_applies_if(expr: str, gt: GroundTruth) -> bool:
    if not expr or expr == "True":
        return True
    if _RE_FORBIDDEN.search(expr):
        raise ValueError(f"Unsafe expression in template: {expr!r}")
    try:
        return bool(eval(expr, {"__builtins__": {}}, {
            "gt": gt,
            "any": any, "all": all, "len": len, "range": range,
            "True": True, "False": False, "None": None,
        }))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────
#  Format string rendering (attribute-only, no code execution)
# ─────────────────────────────────────────────────────────────────────────


def _resolve(path: str, obj) -> Optional[str]:
    """Resolve 'gt.modal_system.name'-style path on obj."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        if hasattr(cur, part):
            cur = getattr(cur, part)
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _render(tmpl: str, gt: GroundTruth) -> str:
    """Replace {gt.x.y} occurrences in tmpl with safe attribute lookups."""
    def _sub(match: re.Match) -> str:
        path = match.group(1).strip()
        val = _resolve(path, None if not path.startswith("gt") else gt
                       if path == "gt" else gt)
        # Generic: always start at `gt` for gt.* paths
        if path.startswith("gt."):
            val = _resolve(path[3:], gt)
        return "?" if val is None else str(val)

    return re.sub(r"\{([^{}]+)\}", _sub, tmpl)


# ─────────────────────────────────────────────────────────────────────────
#  Synthesis
# ─────────────────────────────────────────────────────────────────────────


def _instrument_positive_facts(tradition: str, gt: GroundTruth, idx: int) -> List[AtomicFact]:
    """One positive 'is X present?' fact per annotated instrument."""
    out: List[AtomicFact] = []
    for inst in gt.instruments:
        fact_id = f"auto_inst_pos_{idx:03d}_{inst.canonical}"
        out.append(AtomicFact(
            fact_id=fact_id,
            category="instrument",
            claim=f"The instrument '{inst.canonical}' is present.",
            verification_question=(
                f"Is '{inst.canonical.replace('_', ' ')}' audible in the clip?"
            ),
            expected_answer=True,
            difficulty="must_have",
            source="auto:instrument_present",
            grounding=f"ground_truth.instruments[{inst.canonical}]",
        ))
        idx += 1
    return out


def _negative_instrument_facts(tradition: str, gt: GroundTruth, idx: int,
                               max_negatives: int = 3) -> List[AtomicFact]:
    """Sample negative 'is X present?' facts from culturally-implausible instruments.

    Draws from instruments vocab where `tradition` is listed in
    `out_of_culture_for`. Deterministic given the tradition string.
    """
    vocab = load_vocab("instruments")
    candidates: List[str] = []
    for name, entry in vocab.items():
        ooc = entry.get("out_of_culture_for") or []
        if tradition in ooc:
            candidates.append(name)

    # Sort for determinism, then pick top-N.
    candidates.sort()
    picks = candidates[:max_negatives]

    out: List[AtomicFact] = []
    for inst in picks:
        fact_id = f"auto_inst_neg_{idx:03d}_{inst}"
        out.append(AtomicFact(
            fact_id=fact_id,
            category="instrument",
            claim=f"'{inst}' is NOT present.",
            verification_question=(
                f"Is a '{inst.replace('_', ' ')}' audible in the clip?"
            ),
            expected_answer=False,
            difficulty="easy_to_get_wrong",
            source="auto:instrument_negative",
            grounding=f"vocab/instruments.yaml#out_of_culture_for~{tradition}",
        ))
        idx += 1
    return out


def synthesize_facts(tradition: str, gt: GroundTruth,
                     max_negative_instruments: int = 3) -> List[AtomicFact]:
    """Synthesize the full atomic-fact list for a clip."""
    facts: List[AtomicFact] = []
    idx = 0

    # 1. Template-driven facts
    for tmpl in load_templates(tradition):
        if not _eval_applies_if(tmpl.get("applies_if", "True"), gt):
            continue
        try:
            claim = _render(tmpl["claim"], gt)
            question = _render(tmpl["question"], gt)
        except Exception:
            continue  # skip templates that fail to render
        facts.append(AtomicFact(
            fact_id=f"auto_tpl_{idx:03d}_{tmpl['id']}",
            category=tmpl.get("category", "cultural"),
            claim=claim,
            verification_question=question,
            expected_answer=bool(tmpl.get("expected", True)),
            difficulty=tmpl.get("difficulty", "must_have"),
            source=f"auto:template:{tmpl['id']}",
            grounding=tmpl.get("grounding"),
        ))
        idx += 1

    # 2. Per-instrument positive facts
    facts.extend(_instrument_positive_facts(tradition, gt, idx))
    idx += len(gt.instruments)

    # 3. Out-of-culture negative instrument facts
    facts.extend(_negative_instrument_facts(
        tradition, gt, idx, max_negatives=max_negative_instruments,
    ))

    # Deduplicate by fact_id (keep first occurrence)
    seen = set()
    out: List[AtomicFact] = []
    for f in facts:
        if f.fact_id in seen:
            continue
        seen.add(f.fact_id)
        out.append(f)
    return out
