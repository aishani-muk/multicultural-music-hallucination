"""
Verify atomic facts using an open-source audio-LLM.

Supported backends:
  - qwen2-audio   : Qwen/Qwen2-Audio-7B-Instruct (HuggingFace)
  - audio-flamingo : Audio Flamingo 2 / Music Flamingo (if available)

This module mirrors the Gemini verifier interface so both can be used
interchangeably in the pipeline.

Usage:
    python -m verify.audio_llm_verifier \
        --manifest outputs/generated_audio/generation_manifest.json \
        --prompts prompts/seed_prompts.json \
        --output outputs/verification_results/qwen2_audio/ \
        --backend qwen2-audio
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.schema import load_prompts, BenchmarkPrompt, AtomicFact


SYSTEM_PROMPT = """\
You are an expert music analyst. You will hear an audio clip and be asked a \
yes/no question about it. Answer ONLY based on what you hear. Do not guess.

Respond in this exact JSON format:
{
  "answer": true or false,
  "confidence": 0.0 to 1.0,
  "evidence": "Brief description of audible evidence"
}

If you cannot determine the answer, respond:
{
  "answer": null,
  "confidence": 0.0,
  "evidence": "Insufficient evidence: <reason>"
}
"""


# ── Qwen2-Audio backend ──────────────────────────────────────

class Qwen2AudioVerifier:
    """Verification using Qwen2-Audio-7B-Instruct."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        dtype = getattr(torch, torch_dtype, torch.bfloat16)
        print(f"Loading Qwen2-Audio: {model_id}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device,
        )
        self.model.eval()
        self.device = device

    def verify(self, audio_path: str, question: str) -> dict:
        """Ask a yes/no question about an audio clip."""
        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": f"Question: {question}\n\nRespond with the JSON format specified."},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        audios = [audio]

        inputs = self.processor(
            text=text, audios=audios, return_tensors="pt", padding=True,
            sampling_rate=16000,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        # Trim input tokens from generated output
        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        return _parse_json_response(response)


# ── Audio Flamingo 2 backend (placeholder) ────────────────────

class AudioFlamingoVerifier:
    """
    Verification using Audio Flamingo 2 / Music Flamingo.

    NOTE: As of this writing, Music Flamingo weights may not yet be publicly
    available on HuggingFace. This class provides the interface so that when
    weights become available, you can drop them in.

    If you have a local clone of the Music Flamingo repo, set model_id to the
    local path.
    """

    def __init__(
        self,
        model_id: str = "audio-flamingo/audio-flamingo-2",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ):
        # Attempt to load; fall back to a clear error message
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            dtype = getattr(torch, torch_dtype, torch.bfloat16)
            print(f"Loading Audio Flamingo: {model_id}")
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype, device_map=device,
                trust_remote_code=True,
            )
            self.model.eval()
            self.device = device
            self._available = True
        except Exception as e:
            print(
                f"WARNING: Could not load Audio Flamingo ({e}). "
                f"Falling back to Qwen2-Audio. To use Audio Flamingo, "
                f"ensure the model is downloaded or cloned locally."
            )
            self._fallback = Qwen2AudioVerifier(device=device, torch_dtype=torch_dtype)
            self._available = False

    def verify(self, audio_path: str, question: str) -> dict:
        if not self._available:
            return self._fallback.verify(audio_path, question)

        import librosa
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Question: {question}\n\nRespond with JSON."
        )

        inputs = self.processor(
            text=prompt, audios=[audio], return_tensors="pt",
            sampling_rate=16000,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=256)

        input_len = inputs["input_ids"].shape[1]
        output_ids = generated_ids[:, input_len:]
        response = self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0]
        return _parse_json_response(response)


# ── Helpers ───────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """Extract JSON from a model response that may contain markdown fences."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Best-effort: look for true/false keywords
        lower = text.lower()
        if "true" in lower and "false" not in lower:
            return {"answer": True, "confidence": 0.5, "evidence": text[:200]}
        elif "false" in lower and "true" not in lower:
            return {"answer": False, "confidence": 0.5, "evidence": text[:200]}
        return {"answer": None, "confidence": 0.0, "evidence": f"Parse error: {text[:200]}"}


def get_verifier(backend: str = "qwen2-audio", **kwargs):
    """Factory: return the appropriate verifier backend."""
    if backend == "qwen2-audio":
        return Qwen2AudioVerifier(**kwargs)
    elif backend in ("audio-flamingo", "audio-flamingo-2"):
        return AudioFlamingoVerifier(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'qwen2-audio' or 'audio-flamingo'.")


def verify_fact_audio_llm(
    verifier,
    audio_path: str,
    fact: AtomicFact,
) -> dict:
    """Verify a single atomic fact using the audio-LLM."""
    parsed = verifier.verify(audio_path, fact.verification_question)

    model_answer = parsed.get("answer")
    if model_answer is not None:
        correct = (bool(model_answer) == fact.expected_answer)
    else:
        correct = None

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
        "verifier": f"audio_llm/{type(verifier).__name__}",
    }


def verify_prompt_audio_llm(
    verifier,
    audio_path: str,
    prompt: BenchmarkPrompt,
) -> dict:
    """Verify all atomic facts for a single prompt/clip."""
    results = []
    for fact in prompt.atomic_facts:
        result = verify_fact_audio_llm(verifier, audio_path, fact)
        results.append(result)
    return {
        "prompt_id": prompt.prompt_id,
        "tradition": prompt.tradition,
        "audio_path": audio_path,
        "verifier": f"audio_llm/{type(verifier).__name__}",
        "fact_results": results,
    }


def run_audio_llm_verification(
    manifest_path: str,
    prompts_path: str,
    output_dir: str,
    backend: str = "qwen2-audio",
    model_id: str | None = None,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
):
    """Run open-source audio-LLM verification on all generated clips."""
    kwargs = {"device": device, "torch_dtype": torch_dtype}
    if model_id:
        kwargs["model_id"] = model_id
    verifier = get_verifier(backend, **kwargs)

    prompts = {p.prompt_id: p for p in load_prompts(prompts_path)}
    with open(manifest_path) as f:
        manifest = json.load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for entry in tqdm(manifest, desc=f"{backend} verification"):
        pid = entry["prompt_id"]
        if pid not in prompts:
            print(f"  Skipping {pid}: no matching prompt")
            continue

        result = verify_prompt_audio_llm(
            verifier, entry["audio_path"], prompts[pid]
        )
        all_results.append(result)

        per_file = output_dir / f"{pid}_{backend}.json"
        with open(per_file, "w") as f:
            json.dump(result, f, indent=2)

    combined = output_dir / f"{backend}_all_results.json"
    with open(combined, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{backend} verification complete → {combined}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-LLM fact verification")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prompts", default="prompts/seed_prompts.json")
    parser.add_argument("--output", default="outputs/verification_results/audio_llm/")
    parser.add_argument("--backend", default="qwen2-audio",
                        choices=["qwen2-audio", "audio-flamingo", "audio-flamingo-2"])
    parser.add_argument("--model_id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16")
    args = parser.parse_args()

    run_audio_llm_verification(
        args.manifest, args.prompts, args.output,
        args.backend, args.model_id, args.device, args.torch_dtype,
    )
