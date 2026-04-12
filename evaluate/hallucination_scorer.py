"""
Hallucination scorer: compute per-clip and aggregate hallucination metrics
from verification results.

Metrics (aligned with the proposal and deep-research-report):
  - Correctness Rate  = (# correct facts) / (# verified facts)
  - Hallucination Rate = (# incorrect facts) / (# verified facts)
  - Abstention Rate   = (# null/abstained facts) / (# total facts)
  - Must-Have Correctness   (subset: difficulty == must_have)
  - Easy-to-Get-Wrong Error (subset: difficulty == easy_to_get_wrong)
  - Per-category breakdown  (instrument, scale, rhythm, etc.)

Usage:
    python -m evaluate.hallucination_scorer \
        --results outputs/verification_results/gemini/gemini_all_results.json \
        --output outputs/scores/gemini_scores.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def score_single_clip(clip_result: dict) -> dict:
    """Compute hallucination metrics for a single clip's verification results."""
    facts = clip_result["fact_results"]
    total = len(facts)
    if total == 0:
        return {"prompt_id": clip_result["prompt_id"], "error": "no facts"}

    verified = [f for f in facts if f["correct"] is not None]
    correct = [f for f in verified if f["correct"] is True]
    incorrect = [f for f in verified if f["correct"] is False]
    abstained = [f for f in facts if f["correct"] is None]

    n_verified = len(verified)

    # Overall rates
    correctness_rate = len(correct) / n_verified if n_verified > 0 else 0.0
    hallucination_rate = len(incorrect) / n_verified if n_verified > 0 else 0.0
    abstention_rate = len(abstained) / total

    # By difficulty
    must_have = [f for f in facts if f["difficulty"] == "must_have"]
    easy_wrong = [f for f in facts if f["difficulty"] == "easy_to_get_wrong"]

    def _subset_rate(subset):
        v = [f for f in subset if f["correct"] is not None]
        c = [f for f in v if f["correct"] is True]
        return len(c) / len(v) if v else 0.0

    must_have_correctness = _subset_rate(must_have)
    easy_wrong_error = 1.0 - _subset_rate(easy_wrong)

    # By category
    category_scores = {}
    cats = defaultdict(list)
    for f in facts:
        cats[f["category"]].append(f)
    for cat, cat_facts in cats.items():
        category_scores[cat] = {
            "correctness": _subset_rate(cat_facts),
            "n_facts": len(cat_facts),
        }

    # Per-fact detail (for drill-down)
    fact_details = []
    for f in facts:
        fact_details.append({
            "fact_id": f["fact_id"],
            "category": f["category"],
            "difficulty": f["difficulty"],
            "expected": f["expected_answer"],
            "model_answer": f["model_answer"],
            "correct": f["correct"],
            "confidence": f.get("confidence", None),
        })

    return {
        "prompt_id": clip_result["prompt_id"],
        "tradition": clip_result["tradition"],
        "verifier": clip_result["verifier"],
        "n_total_facts": total,
        "n_verified": n_verified,
        "n_correct": len(correct),
        "n_incorrect": len(incorrect),
        "n_abstained": len(abstained),
        "correctness_rate": correctness_rate,
        "hallucination_rate": hallucination_rate,
        "abstention_rate": abstention_rate,
        "must_have_correctness": must_have_correctness,
        "easy_wrong_error_rate": easy_wrong_error,
        "category_scores": category_scores,
        "fact_details": fact_details,
    }


def aggregate_scores(clip_scores: list[dict]) -> dict:
    """Compute aggregate statistics across all clips."""
    if not clip_scores:
        return {}

    # Overall
    rates = {
        "correctness_rate": [],
        "hallucination_rate": [],
        "abstention_rate": [],
        "must_have_correctness": [],
        "easy_wrong_error_rate": [],
    }
    for cs in clip_scores:
        if "error" in cs:
            continue
        for key in rates:
            rates[key].append(cs[key])

    agg = {}
    for key, vals in rates.items():
        if vals:
            agg[f"mean_{key}"] = float(np.mean(vals))
            agg[f"std_{key}"] = float(np.std(vals))
            agg[f"median_{key}"] = float(np.median(vals))

    # By tradition
    tradition_agg = defaultdict(lambda: defaultdict(list))
    for cs in clip_scores:
        if "error" in cs:
            continue
        t = cs["tradition"]
        for key in rates:
            tradition_agg[t][key].append(cs[key])

    tradition_summary = {}
    for t, t_rates in tradition_agg.items():
        tradition_summary[t] = {
            "n_clips": len(t_rates["correctness_rate"]),
        }
        for key, vals in t_rates.items():
            tradition_summary[t][f"mean_{key}"] = float(np.mean(vals))
            tradition_summary[t][f"std_{key}"] = float(np.std(vals))

    # By category (pooled across clips)
    cat_pool = defaultdict(lambda: {"correct": 0, "total": 0})
    for cs in clip_scores:
        if "error" in cs:
            continue
        for fd in cs.get("fact_details", []):
            cat = fd["category"]
            if fd["correct"] is not None:
                cat_pool[cat]["total"] += 1
                if fd["correct"]:
                    cat_pool[cat]["correct"] += 1

    category_summary = {}
    for cat, counts in cat_pool.items():
        category_summary[cat] = {
            "correctness": counts["correct"] / counts["total"] if counts["total"] else 0.0,
            "n_facts": counts["total"],
        }

    agg["n_clips"] = len([c for c in clip_scores if "error" not in c])
    agg["by_tradition"] = tradition_summary
    agg["by_category"] = category_summary

    return agg


def run_scoring(
    results_path: str,
    output_path: str,
):
    """Score all verification results and save."""
    with open(results_path) as f:
        all_results = json.load(f)

    clip_scores = [score_single_clip(r) for r in all_results]
    agg = aggregate_scores(clip_scores)

    output = {
        "per_clip": clip_scores,
        "aggregate": agg,
        "source_results": results_path,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Hallucination Scoring Summary ({agg.get('n_clips', 0)} clips)")
    print(f"{'='*60}")
    for key in ["mean_correctness_rate", "mean_hallucination_rate",
                 "mean_abstention_rate", "mean_must_have_correctness",
                 "mean_easy_wrong_error_rate"]:
        if key in agg:
            print(f"  {key}: {agg[key]:.4f}")

    if "by_tradition" in agg:
        print(f"\nBy Tradition:")
        for t, ts in agg["by_tradition"].items():
            print(f"  {t}: correctness={ts['mean_correctness_rate']:.3f}  "
                  f"hallucination={ts['mean_hallucination_rate']:.3f}  "
                  f"(n={ts['n_clips']})")

    if "by_category" in agg:
        print(f"\nBy Fact Category:")
        for cat, cs in agg["by_category"].items():
            print(f"  {cat}: correctness={cs['correctness']:.3f} (n={cs['n_facts']})")

    print(f"\nSaved → {output_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score hallucination results")
    parser.add_argument("--results", required=True,
                        help="Path to *_all_results.json from a verifier")
    parser.add_argument("--output", default="outputs/scores/scores.json")
    args = parser.parse_args()

    run_scoring(args.results, args.output)
