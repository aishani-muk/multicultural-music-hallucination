"""MusicGen generation via HuggingFace Transformers.

Replaces the legacy `generate/musicgen_generate.py`:
  - Seed control (reproducibility)
  - Atomic manifest writes
  - Robust batching; resumes if manifest exists
  - No `audiocraft` dependency
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Union

from tqdm import tqdm

from ..schema import BenchmarkPrompt, load_prompts
from ..utils.io import atomic_write_json
from ..utils.seeds import seed_all

log = logging.getLogger("mhb.generate.musicgen")


def load_musicgen(model_id: str = "facebook/musicgen-large",
                  device: str = "cuda",
                  torch_dtype: str = "float16"):
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    dtype = getattr(torch, torch_dtype, torch.float16)
    log.info("Loading MusicGen: %s (device=%s dtype=%s)", model_id, device, torch_dtype)
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype,
    ).to(device)
    model.eval()
    return model, processor


def generate_batch(
    model, processor, prompts: List[str],
    duration_sec: float, guidance_scale: float, max_new_tokens: int,
    device: str, seed: int,
):
    import torch
    inputs = processor(text=prompts, padding=True, return_tensors="pt").to(device)
    tokens_per_sec = 50
    n_tokens = min(max_new_tokens, int(duration_sec * tokens_per_sec))
    # Seed globally — newer transformers versions don't accept `generator` kwarg
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    with torch.no_grad():
        audio = model.generate(
            **inputs, max_new_tokens=n_tokens,
            guidance_scale=guidance_scale, do_sample=True,
        )
    sr = model.config.audio_encoder.sampling_rate
    return [(audio[i, 0].detach().cpu().float().numpy(), sr)
            for i in range(audio.shape[0])]


def run_generation(
    prompts_path: Union[str, Path],
    output_dir: Union[str, Path],
    model_id: str = "facebook/musicgen-large",
    duration_sec: float = 10.0,
    guidance_scale: float = 3.0,
    max_new_tokens: int = 1024,
    batch_size: int = 4,
    device: str = "cuda",
    torch_dtype: str = "float16",
    seed: int = 848,
    resume: bool = True,
) -> list:
    seed_all(seed)
    import soundfile as sf

    prompts: List[BenchmarkPrompt] = load_prompts(prompts_path)
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "generation_manifest.json"
    if resume and manifest_path.exists():
        import json
        existing = json.loads(manifest_path.read_text())
        done = {e["prompt_id"] for e in existing}
        log.info("Resuming: %d clips already present", len(done))
    else:
        existing, done = [], set()

    todo = [p for p in prompts if p.prompt_id not in done]
    if not todo:
        log.info("Nothing to generate — manifest already complete.")
        return existing

    model, processor = load_musicgen(model_id, device, torch_dtype)

    manifest = list(existing)
    for i in tqdm(range(0, len(todo), batch_size), desc="MusicGen"):
        batch = todo[i : i + batch_size]
        texts = [p.generation_prompt for p in batch]
        audios = generate_batch(
            model, processor, texts,
            duration_sec=duration_sec,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens,
            device=device, seed=seed + i,
        )
        for p, (wav, sr) in zip(batch, audios):
            out = output_dir / f"{p.prompt_id}.wav"
            sf.write(str(out), wav, sr)
            manifest.append({
                "prompt_id": p.prompt_id,
                "tradition": p.tradition,
                "generation_prompt": p.generation_prompt,
                "audio_path": str(out),
                "model": model_id,
                "duration_sec": duration_sec,
                "sampling_rate": sr,
                "seed": seed + i,
            })
            atomic_write_json(manifest_path, manifest)

    log.info("Generation complete → %s (%d clips)", output_dir, len(manifest))
    return manifest
