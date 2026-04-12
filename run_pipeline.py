"""
End-to-end orchestrator for the Multicultural Music Hallucination Benchmark.

Stages:
  1. Generate — produce audio clips from prompts using MusicGen
  2. Verify  — run Gemini and/or open-source audio-LLM verification
  3. Score   — compute hallucination / correctness rates
  4. Analyze — per-genre / per-model breakdowns + figures

Usage:
    # Full pipeline
    python run_pipeline.py --config configs/benchmark_config.yaml

    # Individual stages
    python run_pipeline.py --config configs/benchmark_config.yaml --stage generate
    python run_pipeline.py --config configs/benchmark_config.yaml --stage verify
    python run_pipeline.py --config configs/benchmark_config.yaml --stage score
    python run_pipeline.py --config configs/benchmark_config.yaml --stage analyze

    # Skip generation (use pre-existing audio, e.g., from Suno/Udio)
    python run_pipeline.py --config configs/benchmark_config.yaml --stage verify \
        --manifest outputs/generated_audio/generation_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stage_generate(cfg: dict):
    """Stage 1: Generate audio clips from benchmark prompts."""
    from generate.musicgen_generate import run_generation

    gen_cfg = cfg["generation"]
    return run_generation(
        prompts_path=cfg["paths"]["prompts"],
        output_dir=cfg["paths"]["generated_audio"],
        model_id=gen_cfg["model"],
        duration_sec=gen_cfg["duration_sec"],
        guidance_scale=gen_cfg["guidance_scale"],
        max_new_tokens=gen_cfg["max_new_tokens"],
        batch_size=gen_cfg["batch_size"],
        device=gen_cfg["device"],
    )


def stage_verify(cfg: dict, manifest_path: str | None = None):
    """Stage 2: Run verification with Gemini and audio-LLM."""
    prompts_path = cfg["paths"]["prompts"]
    verify_dir = Path(cfg["paths"]["verification_results"])

    if manifest_path is None:
        manifest_path = str(
            Path(cfg["paths"]["generated_audio"]) / "generation_manifest.json"
        )

    results = {}

    # ── Gemini verification ──
    gemini_cfg = cfg.get("gemini", {})
    if gemini_cfg:
        from verify.gemini_verifier import run_gemini_verification
        gemini_out = verify_dir / "gemini"
        print("\n" + "=" * 60)
        print("STAGE 2a: Gemini Verification")
        print("=" * 60)
        results["gemini"] = run_gemini_verification(
            manifest_path=manifest_path,
            prompts_path=prompts_path,
            output_dir=str(gemini_out),
            model=gemini_cfg.get("model", "gemini-2.0-flash"),
            rpm_limit=gemini_cfg.get("requests_per_minute", 15),
            max_retries=gemini_cfg.get("max_retries", 3),
        )

    # ── Audio-LLM verification ──
    allm_cfg = cfg.get("audio_llm", {})
    if allm_cfg:
        from verify.audio_llm_verifier import run_audio_llm_verification
        backend = allm_cfg.get("backend", "qwen2-audio")
        allm_out = verify_dir / backend
        print("\n" + "=" * 60)
        print(f"STAGE 2b: {backend} Verification")
        print("=" * 60)
        results["audio_llm"] = run_audio_llm_verification(
            manifest_path=manifest_path,
            prompts_path=prompts_path,
            output_dir=str(allm_out),
            backend=backend,
            model_id=allm_cfg.get("model_id"),
            device=allm_cfg.get("device", "cuda"),
            torch_dtype=allm_cfg.get("torch_dtype", "bfloat16"),
        )

    return results


def stage_score(cfg: dict):
    """Stage 3: Score verification results."""
    from evaluate.hallucination_scorer import run_scoring

    verify_dir = Path(cfg["paths"]["verification_results"])
    scores_dir = Path(cfg["paths"]["final_scores"])
    scores_dir.mkdir(parents=True, exist_ok=True)

    outputs = {}

    # Gemini scores
    gemini_results = verify_dir / "gemini" / "gemini_all_results.json"
    if gemini_results.exists():
        print("\n" + "=" * 60)
        print("STAGE 3a: Scoring Gemini Results")
        print("=" * 60)
        outputs["gemini"] = run_scoring(
            str(gemini_results),
            str(scores_dir / "gemini_scores.json"),
        )

    # Audio-LLM scores
    backend = cfg.get("audio_llm", {}).get("backend", "qwen2-audio")
    allm_results = verify_dir / backend / f"{backend}_all_results.json"
    if allm_results.exists():
        print("\n" + "=" * 60)
        print(f"STAGE 3b: Scoring {backend} Results")
        print("=" * 60)
        outputs["audio_llm"] = run_scoring(
            str(allm_results),
            str(scores_dir / f"{backend}_scores.json"),
        )

    return outputs


def stage_analyze(cfg: dict):
    """Stage 4: Run analysis and generate figures."""
    from evaluate.analysis import run_analysis

    scores_dir = Path(cfg["paths"]["final_scores"])
    backend = cfg.get("audio_llm", {}).get("backend", "qwen2-audio")

    gemini_path = scores_dir / "gemini_scores.json"
    allm_path = scores_dir / f"{backend}_scores.json"

    print("\n" + "=" * 60)
    print("STAGE 4: Analysis & Visualization")
    print("=" * 60)

    return run_analysis(
        gemini_scores_path=str(gemini_path) if gemini_path.exists() else None,
        audio_llm_scores_path=str(allm_path) if allm_path.exists() else None,
        figures_dir=cfg["paths"]["figures"],
    )


STAGES = {
    "generate": stage_generate,
    "verify": stage_verify,
    "score": stage_score,
    "analyze": stage_analyze,
}


def main():
    parser = argparse.ArgumentParser(
        description="Multicultural Music Hallucination Benchmark Pipeline"
    )
    parser.add_argument(
        "--config", default="configs/benchmark_config.yaml",
        help="Path to benchmark config YAML"
    )
    parser.add_argument(
        "--stage", default=None,
        choices=list(STAGES.keys()),
        help="Run a single stage (default: run all stages)"
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Override: path to generation_manifest.json (for verify stage)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.stage:
        print(f"\n{'#'*60}")
        print(f"  Running stage: {args.stage}")
        print(f"{'#'*60}")
        if args.stage == "verify":
            STAGES[args.stage](cfg, args.manifest)
        else:
            STAGES[args.stage](cfg)
    else:
        # Run all stages sequentially
        print(f"\n{'#'*60}")
        print(f"  FULL PIPELINE RUN")
        print(f"{'#'*60}")

        print("\n[1/4] Generating audio...")
        stage_generate(cfg)

        print("\n[2/4] Verifying facts...")
        stage_verify(cfg)

        print("\n[3/4] Scoring...")
        stage_score(cfg)

        print("\n[4/4] Analyzing...")
        stage_analyze(cfg)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Audio clips   → {cfg['paths']['generated_audio']}")
        print(f"  Verifications → {cfg['paths']['verification_results']}")
        print(f"  Scores        → {cfg['paths']['final_scores']}")
        print(f"  Figures       → {cfg['paths']['figures']}")


if __name__ == "__main__":
    main()
