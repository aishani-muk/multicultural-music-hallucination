"""Stable Audio Open generation via HuggingFace Diffusers.

Uses stabilityai/stable-audio-open-1.0 — generates up to 47s stereo
audio at 44.1kHz from text prompts. No special deps beyond diffusers + torch.

    pip install diffusers torch soundfile
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from tqdm import tqdm

from ..schema import BenchmarkPrompt, load_prompts
from ..utils.io import atomic_write_json
from ..utils.seeds import seed_all

log = logging.getLogger("mhb.generate.stable_audio")


def load_stable_audio(model_id: str = "stabilityai/stable-audio-open-1.0",
                      device: str = "cuda",
                      torch_dtype: str = "float16"):
    import torch
    from diffusers import StableAudioPipeline

    dtype = getattr(torch, torch_dtype, torch.float16)
    log.info("Loading Stable Audio: %s (device=%s dtype=%s)", model_id, device, torch_dtype)
    pipe = StableAudioPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    return pipe


def run_generation(
    prompts_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_id: str = "stabilityai/stable-audio-open-1.0",
    duration_sec: float = 10.0,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.0,
    device: str = "cuda",
    torch_dtype: str = "float16",
    seed: int = 848,
    resume: bool = True,
) -> list:
    """Generate audio for all prompts using Stable Audio Open."""
    seed_all(seed)
    import torch
    import soundfile as sf

    prompts: List[BenchmarkPrompt] = load_prompts(prompts_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "generation_manifest.json"
    if resume and manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        done = {e["prompt_id"] for e in existing}
        log.info("Resuming: %d clips already present", len(done))
    else:
        existing, done = [], set()

    todo = [p for p in prompts if p.prompt_id not in done]
    if not todo:
        log.info("Nothing to generate — manifest already complete.")
        return existing

    pipe = load_stable_audio(model_id, device, torch_dtype)
    sr = pipe.vae.sampling_rate  # 44100

    manifest = list(existing)
    for i, p in enumerate(tqdm(todo, desc="StableAudio")):
        clip_seed = seed + i
        generator = torch.Generator(device=device).manual_seed(clip_seed)

        audio = pipe(
            prompt=p.generation_prompt,
            negative_prompt="Low quality, noise, distortion.",
            num_inference_steps=num_inference_steps,
            audio_end_in_s=duration_sec,
            num_waveforms_per_prompt=1,
            guidance_scale=guidance_scale,
            generator=generator,
        ).audios

        # audio shape: [1, channels, samples] — take first, transpose for soundfile
        wav = audio[0].T.float().cpu().numpy()
        out_path = output_dir / f"{p.prompt_id}.wav"
        sf.write(str(out_path), wav, sr)

        manifest.append({
            "prompt_id": p.prompt_id,
            "tradition": p.tradition,
            "generation_prompt": p.generation_prompt,
            "audio_path": str(out_path.resolve()),
            "model": model_id,
            "duration_sec": duration_sec,
            "sampling_rate": sr,
            "seed": clip_seed,
        })
        atomic_write_json(manifest_path, manifest)

    log.info("Stable Audio generation complete → %s (%d clips)", output_dir, len(manifest))
    return manifest
