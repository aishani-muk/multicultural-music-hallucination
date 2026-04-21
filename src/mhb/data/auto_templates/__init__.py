"""Per-tradition auto-fact templates (YAML) + synthesizer.

The YAML schema per template entry:

  - id: unique within the template file
    category: instrument|scale|rhythm|structure|timbre|tuning|cultural|modal_system|metric_cycle|form|...
    difficulty: must_have|culture_discriminative|easy_to_get_wrong|expert_only
    applies_if: python-like expression over GroundTruth (see _eval_applies_if)
    claim: f-string using {gt.modal_system.name}, {gt.metric_cycle.beats}, ...
    question: f-string verification question
    expected: true|false
    grounding: dotted path in GroundTruth

Negative-fact pools (out-of-culture instruments) are built from
instruments.yaml's `out_of_culture_for` field at synthesis time, not
hand-listed per template file, so adding a new instrument automatically
propagates.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import yaml

TEMPLATE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def load_templates(tradition: str) -> List[dict]:
    path = TEMPLATE_DIR / f"{tradition}.yaml"
    if not path.exists():
        return []
    with open(path) as f:
        data = yaml.safe_load(f) or []
    return data
