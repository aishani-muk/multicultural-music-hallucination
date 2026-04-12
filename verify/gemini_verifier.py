"""
Verify atomic facts about generated audio clips using Google Gemini API.

Gemini natively supports audio inputs. For each atomic fact we send:
  - The audio clip
  - A binary verification question
  - System prompt enforcing evidence-disciplined answering

Returns structured JSON with per-fact verdicts.

Requires: GOOGLE_API_KEY environment variable.

Usage:
    python -m verify.gemini_verifier \
        --manifest outputs/generated_audio/generation_manifest.json \
        --prompts prompts/seed_prompts.json \
        --output outputs/verification_results/gemini/
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.schema import load_prompts, BenchmarkPrompt, AtomicFact

# ── Gemini client setup ──────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert music analyst. You will be given an audio clip and a yes/no \
question about its musical content. Answer ONLY based on what you can hear in \
the audio. Do not guess or use prior knowledge about specific artists, albums, \
or recordings.

Respond in this exact JSON format:
{
  "answer": true or false,
  "confidence": 0.0 to 1.0,
  "evidence": "Brief description of the audible evidence supporting your answer"
}

If you genuinely cannot determine the answer from the audio, respond:
{
  "answer": null,
  "confidence": 0.0,
  "evidence": "Insufficient evidence: <reason>"
}
"""


def _init_gemini(api_key: str | None = None, model: str = "gemini-2.0-flash"):
    """Initialize the Gemini client."""
    from google import genai
    api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY env var or pass api_key argument"
        )
    client = genai.Client(api_key=api_key)
    return client, model


def _parse_json_response(text: str) -> dict:
    """Extract JSON from a Gemini response that may contain markdown fences."""
    text = text.strip()
    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"answer": None, "confidence": 0.0, "evidence": f"Parse error: {text[:200]}"}


def verify_fact_gemini(
    client,
    model: str,
    audio_path: str,
    fact: AtomicFact,
    max_retries: int = 3,
) -> dict:
    """
    Verify a single atomic fact against an audio clip using Gemini.

    Returns:
        Dict with keys: fact_id, answer, confidence, evidence, expected_answer, correct
    """
    from google.genai import types

    audio_bytes = Path(audio_path).read_bytes()

    user_content = [
        types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
        types.Part.from_text(
            text=f"Question: {fact.verification_question}\n\n"
            f"Answer with the JSON format specified in your instructions."
        ),
    ]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Content(role="user", parts=user_content)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            parsed = _parse_json_response(response.text)

            model_answer = parsed.get("answer")
            if model_answer is not None:
                correct = (bool(model_answer) == fact.expected_answer)
            else:
                correct = None  # Abstention

            return {
                "fact_id": fact.fact_id,
                "category": fact.category,
                "claim": fact.claim,
                "verification_question": fact.verification_question,
                "expected_answer": fact.expected_answer,
                "difficulty": fact.difficulty,
                "model_answer": model_answer,
                "confidence": parsed.get("confidence", 0.0),
                "evidence": parsed.get("evidence", ""),
                "correct": correct,
                "verifier": f"gemini/{model}",
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  Retry {attempt+1}/{max_retries} after error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                return {
                    "fact_id": fact.fact_id,
                    "category": fact.category,
                    "claim": fact.claim,
                    "verification_question": fact.verification_question,
                    "expected_answer": fact.expected_answer,
                    "difficulty": fact.difficulty,
                    "model_answer": None,
                    "confidence": 0.0,
                    "evidence": f"API error after {max_retries} retries: {e}",
                    "correct": None,
                    "verifier": f"gemini/{model}",
                }


def verify_prompt_gemini(
    client,
    model: str,
    audio_path: str,
    prompt: BenchmarkPrompt,
    rpm_limit: int = 15,
    max_retries: int = 3,
) -> dict:
    """Verify all atomic facts for a single prompt/clip."""
    results = []
    delay = 60.0 / rpm_limit if rpm_limit > 0 else 0

    for fact in prompt.atomic_facts:
        result = verify_fact_gemini(client, model, audio_path, fact, max_retries)
        results.append(result)
        if delay > 0:
            time.sleep(delay)

    return {
        "prompt_id": prompt.prompt_id,
        "tradition": prompt.tradition,
        "audio_path": audio_path,
        "verifier": f"gemini/{model}",
        "fact_results": results,
    }


def run_gemini_verification(
    manifest_path: str,
    prompts_path: str,
    output_dir: str,
    model: str = "gemini-2.0-flash",
    api_key: str | None = None,
    rpm_limit: int = 15,
    max_retries: int = 3,
):
    """Run Gemini verification on all generated clips."""
    client, model = _init_gemini(api_key, model)
    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)}

    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for entry in tqdm(manifest, desc="Gemini verification"):
        pid = entry["prompt_id"]
        if pid not in prompts:
            print(f"  Skipping {pid}: no matching prompt")
            continue

        result = verify_prompt_gemini(
            client, model, entry["audio_path"],
            prompts[pid], rpm_limit, max_retries,
        )
        all_results.append(result)

        # Save incrementally
        per_file = output_dir / f"{pid}_gemini.json"
        with open(per_file, "w") as f:
            json.dump(result, f, indent=2)

    # Save combined results
    combined = output_dir / "gemini_all_results.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGemini verification complete → {combined}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini audio fact verification")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompts", default="prompts/seed_prompts.json")
    parser.add_argument("--output", default="outputs/verification_results/gemini/")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--rpm_limit", type=int, default=15)
    parser.add_argument("--max_retries", type=int, default=3)
    args = parser.parse_args()

    run_gemini_verification(
        args.manifest, args.prompts, args.output,
        args.model, rpm_limit=args.rpm_limit, max_retries=args.max_retries,
    )
