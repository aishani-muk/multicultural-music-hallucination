"""Cross-verifier analysis + figures.

Given two or more `*_scores.json` files, computes:
  - Pairwise per-fact agreement + Cohen's kappa
  - Per-tradition / per-category breakdown table
  - Figures: hallucination-by-tradition bar, category-correctness heatmap,
    must-have vs easy-to-get-wrong bar.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

log = logging.getLogger("mhb.evaluate.analysis")


def load_scores(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return json.load(f)


def scores_to_rows(scores: dict, verifier_label: str) -> List[dict]:
    rows = []
    for cs in scores.get("per_clip", []):
        if "error" in cs:
            continue
        rows.append({
            "clip_id": cs["clip_id"],
            "tradition": cs.get("tradition", "unknown"),
            "verifier": verifier_label,
            "correctness_rate": cs["correctness_rate"],
            "hallucination_rate": cs["hallucination_rate"],
            "abstention_rate": cs["abstention_rate"],
            "must_have_correctness": cs["must_have_correctness"],
            "culture_discriminative_correctness": cs.get(
                "culture_discriminative_correctness", 0.0),
            "easy_wrong_error_rate": cs["easy_wrong_error_rate"],
        })
    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Cross-verifier agreement
# ─────────────────────────────────────────────────────────────────────────


def _build_fact_lookup(scores: dict) -> Dict[Tuple[str, str], Optional[bool]]:
    lookup = {}
    for cs in scores.get("per_clip", []):
        if "error" in cs:
            continue
        for fd in cs.get("fact_details", []):
            lookup[(cs["clip_id"], fd["fact_id"])] = fd.get("correct")
    return lookup


def pairwise_agreement(a: dict, b: dict) -> dict:
    la = _build_fact_lookup(a)
    lb = _build_fact_lookup(b)
    keys = set(la) & set(lb)
    # Exclude abstentions on either side.
    both = [(la[k], lb[k]) for k in keys
            if la[k] is not None and lb[k] is not None]
    if not both:
        return {"n_common": len(keys), "n_both_answered": 0,
                "agreement_rate": 0.0, "kappa": 0.0}

    agree = sum(1 for x, y in both if x == y)
    n = len(both)
    p_a = agree / n

    # Cohen's kappa for binary correct/incorrect.
    p_true_a = sum(1 for x, _ in both if x) / n
    p_true_b = sum(1 for _, y in both if y) / n
    p_e = p_true_a * p_true_b + (1 - p_true_a) * (1 - p_true_b)
    kappa = 0.0 if (1 - p_e) == 0 else (p_a - p_e) / (1 - p_e)

    return {
        "n_common": len(keys),
        "n_both_answered": n,
        "agreement_rate": p_a,
        "kappa": float(kappa),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Plotting
# ─────────────────────────────────────────────────────────────────────────


def _try_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        log.warning("matplotlib not available; skipping plots")
        return None


def plot_hallucination_by_tradition(rows: List[dict], out_path: Path) -> Optional[Path]:
    plt = _try_plot()
    if plt is None:
        return None
    verifiers = sorted({r["verifier"] for r in rows})
    traditions = sorted({r["tradition"] for r in rows})
    width = 0.8 / max(1, len(verifiers))

    fig, ax = plt.subplots(figsize=(max(6, len(traditions) * 1.3), 4.5))
    for i, v in enumerate(verifiers):
        means = []
        for t in traditions:
            xs = [r["hallucination_rate"] for r in rows
                  if r["tradition"] == t and r["verifier"] == v]
            means.append(float(np.mean(xs)) if xs else 0.0)
        xpos = np.arange(len(traditions)) + i * width - (len(verifiers) - 1) * width / 2
        ax.bar(xpos, means, width=width, label=v)

    ax.set_xticks(np.arange(len(traditions)))
    ax.set_xticklabels(traditions, rotation=20, ha="right")
    ax.set_ylabel("Hallucination rate (lower is better)")
    ax.set_ylim(0, 1)
    ax.set_title("Hallucination rate by tradition × verifier")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_category_heatmap(scores_list: List[dict], labels: List[str],
                          out_path: Path) -> Optional[Path]:
    plt = _try_plot()
    if plt is None:
        return None

    # Build category × verifier correctness matrix.
    categories = set()
    for s in scores_list:
        categories.update(s.get("aggregate", {}).get("by_category", {}).keys())
    categories = sorted(categories)
    if not categories:
        return None

    mat = np.zeros((len(categories), len(scores_list)))
    for j, s in enumerate(scores_list):
        cats = s.get("aggregate", {}).get("by_category", {})
        for i, c in enumerate(categories):
            mat[i, j] = cats.get(c, {}).get("correctness", np.nan)

    fig, ax = plt.subplots(figsize=(1.5 + 1.5 * len(labels),
                                    0.4 + 0.35 * len(categories)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=8)
    ax.set_title("Correctness by category × verifier")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────
#  Top-level
# ─────────────────────────────────────────────────────────────────────────


def run_analysis(
    score_paths: Dict[str, Path],   # {verifier_label: Path}
    out_dir: Union[str, Path],
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = {k: load_scores(v) for k, v in score_paths.items()}
    rows: List[dict] = []
    for label, s in scores.items():
        rows.extend(scores_to_rows(s, label))

    # Pairwise agreements
    labels = list(scores.keys())
    agreement: Dict[str, dict] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            agreement[f"{la}__vs__{lb}"] = pairwise_agreement(scores[la], scores[lb])

    summary = {
        "verifiers": labels,
        "n_rows": len(rows),
        "pairwise_agreement": agreement,
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV for easy pandas/spreadsheet use.
    try:
        import pandas as pd
        pd.DataFrame(rows).to_csv(out_dir / "per_clip.csv", index=False)
    except ImportError:
        pass

    plot_hallucination_by_tradition(rows, out_dir / "hallucination_by_tradition.png")
    plot_category_heatmap(list(scores.values()), labels,
                          out_dir / "category_correctness.png")

    log.info("Analysis written → %s", out_dir)
    return summary
