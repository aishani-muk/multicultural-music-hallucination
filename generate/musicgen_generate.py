"""
Generate audio clips from text prompts using Meta's MusicGen (HuggingFace).

Supports:
  - facebook/musicgen-small   (300M)
  - facebook/musicgen-medium  (1.5B)
  - facebook/musicgen-large   (3.3B)

Usage:
    python -m generate.musicgen_generate \
        --prompts prompts/seed_prompts.json \
        --output outputs/generated_audio/ \
        --model facebook/musicgen-large \
        --duration 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts.schema import load_prompts, BenchmarkPrompt


def load_musicgen(model_id: str, device: str = "cuda"):
    """Load MusicGen model and processor from HuggingFace."""
    print(f"Loading MusicGen model: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()
    return model, processor


def generate_audio(
    model,
    processor,
    prompts: list[str],
    duration_sec: float = 10.0,
    guidance_scale: float = 3.0,
    max_new_tokens: int = 1024,
    device: str = "cuda",
) -> list[tuple[torch.Tensor, int]]:
    """
    Generate audio for a batch of text prompts.

    Returns:
        List of (audio_tensor, sampling_rate) tuples.
    """
    inputs = processor(
        text=prompts,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # MusicGen generates ~50 tokens/sec at 32kHz
    tokens_per_sec = 50
    calc_tokens = int(duration_sec * tokens_per_sec)
    n_tokens = min(max_new_tokens, calc_tokens)

    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            guidance_scale=guidance_scale,
            do_sample=True,
        )

    sr = model.config.audio_encoder.sampling_rate
    results = []
    for i in range(audio_values.shape[0]):
        wav = audio_values[i, 0].cpu().float().numpy()
        results.append((wav, sr))
    return results


def run_generation(
    prompts_path: str,
    output_dir: str,
    model_id: str = "facebook/musicgen-large",
    duration_sec: float = 10.0,
    guidance_scale: float = 3.0,
    max_new_tokens: int = 1024,
    batch_size: int = 4,
    device: str = "cuda",
):
    """Generate audio for all benchmark prompts and save to disk."""
    benchmark_prompts = load_prompts(prompts_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_musicgen(model_id, device)

    manifest = []

    # Process in batches
    for i in tqdm(range(0, len(benchmark_prompts), batch_size),
                  desc="Generating audio"):
        batch = benchmark_prompts[i : i + batch_size]
        text_prompts = [p.generation_prompt for p in batch]

        results = generate_audio(
            model, processor, text_prompts,
            duration_sec=duration_sec,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens,
            device=device,
        )

        for prompt, (wav, sr) in zip(batch, results):
            out_path = output_dir / f"{prompt.prompt_id}.wav"
            sf.write(str(out_path), wav, sr)
            manifest.append({
                "prompt_id": prompt.prompt_id,
                "tradition": prompt.tradition,
                "generation_prompt": prompt.generation_prompt,
                "audio_path": str(out_path),
                "model": model_id,
                "duration_sec": duration_sec,
                "sampling_rate": sr,
            })

    # Save manifest
    manifest_path = output_dir / "generation_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nGenerated {len(manifest)} clips → {output_dir}")
    print(f"Manifest saved → {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate audio with MusicGen")
    parser.add_argument("--prompts", default="prompts/seed_prompts.json")
    parser.add_argument("--output", default="outputs/generated_audio/")
    parser.add_argument("--model", default="facebook/musicgen-large")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_generation(
        args.prompts, args.output, args.model,
        args.duration, args.guidance_scale, args.max_new_tokens,
        args.batch_size, args.device,
    )
