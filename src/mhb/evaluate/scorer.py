"""Score per-clip and aggregate hallucination metrics + bootstrap CIs.

Headline metrics:
  - correctness_rate   = n_correct / n_verified
  - hallucination_rate = n_incorrect / n_verified
  - abstention_rate    = n_abstained / n_total
  - must_have_correctness
  - culture_discriminative_correctness
  - easy_wrong_error_rate
  - per-category breakdowns
  - per-tier / per-tradition aggregates
  - bootstrap 95% CIs (stratified by tradition+tier)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ..utils.io import atomic_write_json

log = logging.getLogger("mhb.evaluate.scorer")


# ─────────────────────────────────────────────────────────────────────────
#  Per-clip scoring
# ─────────────────────────────────────────────────────────────────────────


def _subset_correctness(facts: List[dict]) -> float:
    verified = [f for f in facts if f.get("correct") is not None]
    if not verified:
        return 0.0
    correct = [f for f in verified if f["correct"] is True]
    return len(correct) / len(verified)


def score_clip(clip_result: dict) -> dict:
    """Per-clip metrics from a single ClipVerification dict."""
    facts = clip_result.get("fact_results", [])
    total = len(facts)
    if total == 0:
        return {"clip_id": clip_result.get("clip_id"), "error": "no_facts"}

    verified = [f for f in facts if f.get("correct") is not None]
    correct = [f for f in verified if f["correct"] is True]
    incorrect = [f for f in verified if f["correct"] is False]
    abstained = [f for f in facts if f.get("correct") is None]

    n_verified = len(verified)

    correctness_rate = len(correct) / n_verified if n_verified else 0.0
    hallucination_rate = len(incorrect) / n_verified if n_verified else 0.0
    abstention_rate = len(abstained) / total

    must_have = [f for f in facts if f.get("difficulty") == "must_have"]
    culture_disc = [f for f in facts if f.get("difficulty") == "culture_discriminative"]
    easy_wrong = [f for f in facts if f.get("difficulty") == "easy_to_get_wrong"]

    category_scores: Dict[str, dict] = {}
    cats: Dict[str, List[dict]] = defaultdict(list)
    for f in facts:
        cats[f.get("category", "unknown")].append(f)
    for cat, cat_facts in cats.items():
        category_scores[cat] = {
            "correctness": _subset_correctness(cat_facts),
            "n_facts": len(cat_facts),
        }

    return {
        "clip_id": clip_result.get("clip_id"),
        "tradition": clip_result.get("tradition"),
        "verifier": clip_result.get("verifier"),
        "source": clip_result.get("source", "clip"),
        "n_total_facts": total,
        "n_verified": n_verified,
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "n_abstained": len(abstained),
        "correctness_rate": correctness_rate,
        "hallucination_rate": hallucination_rate,
        "abstention_rate": abstention_rate,
        "must_have_correctness": _subset_correctness(must_have),
        "culture_discriminative_correctness": _subset_correctness(culture_disc),
        "easy_wrong_error_rate": 1.0 - _subset_correctness(easy_wrong) if easy_wrong else 0.0,
        "category_scores": category_scores,
        "fact_details": [
            {"fact_id": f.get("fact_id"), "category": f.get("category"),
             "difficulty": f.get("difficulty"),
             "expected": f.get("expected_answer"),
             "model_answer": f.get("model_answer"),
             "correct": f.get("correct"),
             "confidence": f.get("confidence", None)}
            for f in facts
        ],
    }


# ─────────────────────────────────────────────────────────────────────────
#  Aggregation
# ─────────────────────────────────────────────────────────────────────────


_RATE_KEYS = (
    "correctness_rate", "hallucination_rate", "abstention_rate",
    "must_have_correctness", "culture_discriminative_correctness",
    "easy_wrong_error_rate",
)


def _stats(values: List[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
    }


def _bootstrap_ci(
    values: List[float], n_resamples: int = 1000, alpha: float = 0.05,
    seed: int = 848,
) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = arr[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def aggregate(
    clip_scores: List[dict],
    bootstrap_n: int = 1000,
    alpha: float = 0.05,
    seed: int = 848,
) -> dict:
    valid = [cs for cs in clip_scores if "error" not in cs]
    if not valid:
        return {"n_clips": 0}

    # Overall stats + CIs
    overall: Dict[str, Any] = {}
    for key in _RATE_KEYS:
        vals = [cs[key] for cs in valid]
        s = _stats(vals)
        lo, hi = _bootstrap_ci(vals, n_resamples=bootstrap_n, alpha=alpha, seed=seed)
        overall[key] = {**s, "ci_low": lo, "ci_high": hi}

    # By tradition
    by_tradition: Dict[str, Dict[str, Any]] = defaultdict(dict)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for cs in valid:
        groups[cs.get("tradition", "unknown")].append(cs)
    for trad, items in groups.items():
        entry: Dict[str, Any] = {"n_clips": len(items)}
        for key in _RATE_KEYS:
            vals = [cs[key] for cs in items]
            entry[key] = {**_stats(vals),
                          **dict(zip(("ci_low", "ci_high"),
                                     _bootstrap_ci(vals, bootstrap_n, alpha, seed)))}
        by_tradition[trad] = entry

    # By category (pool across clips)
    cat_pool: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for cs in valid:
        for fd in cs.get("fact_details", []):
            cat = fd.get("category", "unknown")
            if fd.get("correct") is None:
                continue
            cat_pool[cat]["total"] += 1
            if fd["correct"]:
                cat_pool[cat]["correct"] += 1
    category_summary = {
        c: {"correctness": (d["correct"] / d["total"]) if d["total"] else 0.0,
            "n_facts": d["total"]}
        for c, d in cat_pool.items()
    }

    return {
        "n_clips": len(valid),
        "overall": overall,
        "by_tradition": by_tradition,
        "by_category": category_summary,
    }


def score_verification_file(
    results_path: Union[str, Path],
    output_path: Union[str, Path],
    bootstrap_n: int = 1000,
    alpha: float = 0.05,
    seed: int = 848,
) -> dict:
    """Score a `*_all_results.json` file from the verifier; write JSON output."""
    with open(results_path) as f:
        all_results = json.load(f)

    clip_scores = [score_clip(r) for r in all_results]
    agg = aggregate(clip_scores, bootstrap_n=bootstrap_n, alpha=alpha, seed=seed)

    payload = {
        "per_clip": clip_scores,
        "aggregate": agg,
        "source_results": str(results_path),
        "config": {
            "bootstrap_n": bootstrap_n,
            "alpha": alpha,
            "seed": seed,
        },
    }
    atomic_write_json(output_path, payload)
    _print_summary(payload)
    return payload


# ─────────────────────────────────────────────────────────────────────────
#  Pretty summary
# ─────────────────────────────────────────────────────────────────────────


def _print_summary(payload: dict) -> None:
    agg = payload.get("aggregate", {})
    n = agg.get("n_clips", 0)
    print()
    print("=" * 64)
    print(f"Hallucination Scoring Summary  (n_clips = {n})")
    print("=" * 64)
    overall = agg.get("overall", {})
    for key in _RATE_KEYS:
        entry = overall.get(key, {})
        if not entry:
            continue
        m = entry.get("mean", 0.0)
        lo = entry.get("ci_low", 0.0)
        hi = entry.get("ci_high", 0.0)
        print(f"  {key:<42s}  {m:.3f}   [{lo:.3f}, {hi:.3f}]")

    by_trad = agg.get("by_tradition", {})
    if by_trad:
        print("\nBy tradition:")
        for t, entry in by_trad.items():
            m = entry.get("correctness_rate", {}).get("mean", 0.0)
            h = entry.get("hallucination_rate", {}).get("mean", 0.0)
            print(f"  {t:<22s}  correctness={m:.3f}  hallucination={h:.3f}  (n={entry.get('n_clips')})")

    by_cat = agg.get("by_category", {})
    if by_cat:
        print("\nBy fact category:")
        for c, entry in by_cat.items():
            print(f"  {c:<22s}  correctness={entry['correctness']:.3f}  (n={entry['n_facts']})")
    print()
