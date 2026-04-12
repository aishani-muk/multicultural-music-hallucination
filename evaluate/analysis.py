"""
Analysis and visualization: per-genre, per-model breakdowns, cross-verifier
agreement, and publication-ready figures.

Usage:
    python -m evaluate.analysis \
        --gemini_scores outputs/scores/gemini_scores.json \
        --audio_llm_scores outputs/scores/qwen2_audio_scores.json \
        --figures_dir outputs/figures/
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def load_scores(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def scores_to_dataframe(scores: dict, verifier_label: str) -> pd.DataFrame:
    """Convert per-clip scores to a flat DataFrame."""
    rows = []
    for cs in scores["per_clip"]:
        if "error" in cs:
            continue
        rows.append({
            "prompt_id": cs["prompt_id"],
            "tradition": cs["tradition"],
            "verifier": verifier_label,
            "correctness_rate": cs["correctness_rate"],
            "hallucination_rate": cs["hallucination_rate"],
            "abstention_rate": cs["abstention_rate"],
            "must_have_correctness": cs["must_have_correctness"],
            "easy_wrong_error_rate": cs["easy_wrong_error_rate"],
        })
    return pd.DataFrame(rows)


def cross_verifier_agreement(gemini_scores: dict, audio_llm_scores: dict) -> dict:
    """Compute per-fact agreement between Gemini and audio-LLM verifiers."""
    # Build lookup: (prompt_id, fact_id) -> correct
    def build_lookup(scores):
        lookup = {}
        for cs in scores["per_clip"]:
            if "error" in cs:
                continue
            for fd in cs.get("fact_details", []):
                lookup[(cs["prompt_id"], fd["fact_id"])] = fd["correct"]
        return lookup

    g_lookup = build_lookup(gemini_scores)
    a_lookup = build_lookup(audio_llm_scores)

    common_keys = set(g_lookup.keys()) & set(a_lookup.keys())
    if not common_keys:
        return {"n_common": 0, "agreement": 0.0}

    agree = sum(
        1 for k in common_keys
        if g_lookup[k] is not None and a_lookup[k] is not None
        and g_lookup[k] == a_lookup[k]
    )
    both_answered = sum(
        1 for k in common_keys
        if g_lookup[k] is not None and a_lookup[k] is not None
    )

    return {
        "n_common_facts": len(common_keys),
        "n_both_answered": both_answered,
        "agreement_rate": agree / both_answered if both_answered else 0.0,
    }


def plot_hallucination_by_tradition(df: pd.DataFrame, output_path: str):
    """Bar chart: hallucination rate by tradition, grouped by verifier."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available, skipping plots")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df, x="tradition", y="hallucination_rate",
        hue="verifier", ax=ax, palette="Set2",
    )
    ax.set_title("Hallucination Rate by Musical Tradition")
    ax.set_ylabel("Hallucination Rate (lower is better)")
    ax.set_xlabel("Tradition")
    ax.set_ylim(0, 1)
    ax.legend(title="Verifier")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {output_path}")


def plot_category_correctness(df_facts: pd.DataFrame, output_path: str):
    """Heatmap: correctness rate by fact category × tradition."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        return

    pivot = df_facts.pivot_table(
        values="correct", index="category", columns="tradition",
        aggfunc="mean"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=0, vmax=1, ax=ax)
    ax.set_title("Fact Correctness by Category × Tradition")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {output_path}")


def plot_must_have_vs_easy_wrong(df: pd.DataFrame, output_path: str):
    """Grouped bar: must-have correctness vs easy-to-get-wrong error by tradition."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    traditions = df["tradition"].unique()
    x = np.arange(len(traditions))
    width = 0.35

    mh = [df[df["tradition"] == t]["must_have_correctness"].mean() for t in traditions]
    ew = [df[df["tradition"] == t]["easy_wrong_error_rate"].mean() for t in traditions]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, mh, width, label="Must-Have Correctness", color="#4CAF50")
    ax.bar(x + width / 2, ew, width, label="Easy-to-Get-Wrong Error", color="#F44336")
    ax.set_xticks(x)
    ax.set_xticklabels(traditions, rotation=15)
    ax.set_ylabel("Rate")
    ax.set_title("Must-Have Correctness vs Easy-to-Get-Wrong Error Rate")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure → {output_path}")


def run_analysis(
    gemini_scores_path: str | None = None,
    audio_llm_scores_path: str | None = None,
    figures_dir: str = "outputs/figures/",
):
    """Run full analysis and generate figures."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    gemini_scores = None
    audio_llm_scores = None

    if gemini_scores_path and Path(gemini_scores_path).exists():
        gemini_scores = load_scores(gemini_scores_path)
        dfs.append(scores_to_dataframe(gemini_scores, "Gemini"))

    if audio_llm_scores_path and Path(audio_llm_scores_path).exists():
        audio_llm_scores = load_scores(audio_llm_scores_path)
        dfs.append(scores_to_dataframe(audio_llm_scores, "Audio-LLM"))

    if not dfs:
        print("No score files found. Run scoring first.")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Summary table
    summary = df.groupby(["verifier", "tradition"]).agg(
        mean_hallucination=("hallucination_rate", "mean"),
        std_hallucination=("hallucination_rate", "std"),
        mean_correctness=("correctness_rate", "mean"),
        n_clips=("prompt_id", "count"),
    ).round(4)
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    print(summary.to_string())
    summary_path = figures_dir / "summary_table.csv"
    summary.to_csv(summary_path)
    print(f"\nSaved summary → {summary_path}")

    # Cross-verifier agreement
    if gemini_scores and audio_llm_scores:
        agreement = cross_verifier_agreement(gemini_scores, audio_llm_scores)
        print(f"\nCross-verifier agreement: {agreement}")
        with open(figures_dir / "cross_verifier_agreement.json", "w") as f:
            json.dump(agreement, f, indent=2)

    # Plots
    plot_hallucination_by_tradition(
        df, str(figures_dir / "hallucination_by_tradition.png")
    )
    plot_must_have_vs_easy_wrong(
        df, str(figures_dir / "must_have_vs_easy_wrong.png")
    )

    # Fact-level heatmap (pool from all verifiers)
    fact_rows = []
    for scores in [gemini_scores, audio_llm_scores]:
        if scores is None:
            continue
        for cs in scores["per_clip"]:
            if "error" in cs:
                continue
            for fd in cs.get("fact_details", []):
                if fd["correct"] is not None:
                    fact_rows.append({
                        "tradition": cs["tradition"],
                        "category": fd["category"],
                        "correct": float(fd["correct"]),
                    })
    if fact_rows:
        df_facts = pd.DataFrame(fact_rows)
        plot_category_correctness(
            df_facts, str(figures_dir / "category_correctness_heatmap.png")
        )

    print(f"\nAll figures saved → {figures_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hallucination analysis & plots")
    parser.add_argument("--gemini_scores", default=None)
    parser.add_argument("--audio_llm_scores", default=None)
    parser.add_argument("--figures_dir", default="outputs/figures/")
    args = parser.parse_args()

    run_analysis(args.gemini_scores, args.audio_llm_scores, args.figures_dir)
