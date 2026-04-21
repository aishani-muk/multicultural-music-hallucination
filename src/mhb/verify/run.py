"""Orchestrate a verifier across a set of ClipItems (understanding) or
BenchmarkPrompts+audio (generation), producing FactVerification records.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Union

from tqdm import tqdm

from ..schema import (
    AtomicFact, BenchmarkPrompt, ClipItem, ClipVerification,
    FactVerification, dump_json, dumps,
)
from ..utils.io import append_jsonl, atomic_write_json

log = logging.getLogger("mhb.verify.run")


def verify_fact(
    verifier,
    audio_path: Union[str, Path],
    fact: AtomicFact,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> FactVerification:
    resp = verifier.verify(audio_path, fact.verification_question,
                           start_sec=start_sec, end_sec=end_sec)
    if resp.answer is None:
        correct = None
    else:
        correct = (bool(resp.answer) == bool(fact.expected_answer))
    return FactVerification(
        fact_id=fact.fact_id,
        category=fact.category,
        claim=fact.claim,
        verification_question=fact.verification_question,
        expected_answer=fact.expected_answer,
        difficulty=fact.difficulty,
        model_answer=resp.answer,
        confidence=resp.confidence,
        evidence=resp.evidence,
        correct=correct,
        verifier=verifier.name,
        latency_sec=resp.latency_sec,
    )


def verify_clip(
    verifier,
    clip: ClipItem,
    facts: Optional[List[AtomicFact]] = None,
) -> ClipVerification:
    facts = facts if facts is not None else clip.derived_facts
    results = []
    for f in facts:
        try:
            results.append(verify_fact(
                verifier, clip.audio_path, f,
                start_sec=clip.start_sec, end_sec=clip.end_sec,
            ))
        except Exception as e:
            log.warning("Verifier %s errored on fact %s: %s",
                        verifier.name, f.fact_id, e)
            results.append(FactVerification(
                fact_id=f.fact_id, category=f.category, claim=f.claim,
                verification_question=f.verification_question,
                expected_answer=f.expected_answer, difficulty=f.difficulty,
                model_answer=None, confidence=0.0,
                evidence=f"Verifier error: {e}", correct=None,
                verifier=verifier.name, latency_sec=None,
            ))
    return ClipVerification(
        clip_id=clip.clip_id, tradition=clip.tradition,
        audio_path=str(clip.audio_path),
        verifier=verifier.name, fact_results=results, source="clip",
    )


def run_verification_over_clips(
    verifier,
    clips: List[ClipItem],
    out_dir: Union[str, Path],
    max_facts_per_clip: Optional[int] = None,
    progress: bool = True,
) -> List[ClipVerification]:
    """Run a verifier across a list of ClipItems and persist results."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    incremental = out_dir / f"{_safe(verifier.name)}_incremental.jsonl"
    incremental.unlink(missing_ok=True)

    all_results: List[ClipVerification] = []
    it = tqdm(clips, desc=f"verify/{verifier.name}") if progress else clips
    for clip in it:
        facts = clip.derived_facts
        if max_facts_per_clip is not None:
            facts = facts[:max_facts_per_clip]
        cv = verify_clip(verifier, clip, facts=facts)
        all_results.append(cv)
        # Incremental persist (JSONL) for crash-safety.
        append_jsonl(incremental, _to_dict(cv))

    combined_path = out_dir / f"{_safe(verifier.name)}_all_results.json"
    atomic_write_json(
        combined_path, [_to_dict(cv) for cv in all_results]
    )
    log.info("Saved %d clip verifications → %s",
             len(all_results), combined_path)
    return all_results


def run_verification_over_prompts(
    verifier,
    prompts: List[BenchmarkPrompt],
    audio_manifest: List[dict],
    out_dir: Union[str, Path],
) -> List[ClipVerification]:
    """Generation-track orchestration: one audio file per prompt_id."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {e["prompt_id"]: e for e in audio_manifest}

    all_results: List[ClipVerification] = []
    for p in tqdm(prompts, desc=f"verify/{verifier.name}"):
        entry = by_id.get(p.prompt_id)
        if entry is None:
            log.warning("No audio for prompt %s — skipping", p.prompt_id)
            continue
        results = [verify_fact(verifier, entry["audio_path"], f)
                   for f in p.atomic_facts]
        cv = ClipVerification(
            clip_id=p.prompt_id, tradition=p.tradition,
            audio_path=str(entry["audio_path"]),
            verifier=verifier.name, fact_results=results, source="prompt",
        )
        all_results.append(cv)
    path = out_dir / f"{_safe(verifier.name)}_all_results.json"
    atomic_write_json(path, [_to_dict(cv) for cv in all_results])
    log.info("Saved %d prompt verifications → %s", len(all_results), path)
    return all_results


def _safe(name: str) -> str:
    return name.replace("/", "__").replace(":", "_")


def _to_dict(cv: ClipVerification) -> dict:
    return {
        "clip_id": cv.clip_id,
        "tradition": cv.tradition,
        "audio_path": cv.audio_path,
        "verifier": cv.verifier,
        "source": cv.source,
        "fact_results": [asdict(r) for r in cv.fact_results],
    }
