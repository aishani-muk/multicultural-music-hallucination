"""ACE-Step 1.5 generation via the ace-step package.

ACE-Step is a diffusion-based music generation model that supports
text-to-music with optional lyrics, BPM, key, and time signature control.

Requires the ACE-Step package:
    pip install git+https://github.com/ace-step/ACE-Step-1.5.git

Models auto-download from HuggingFace on first run (~10GB).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Union

from tqdm import tqdm

from ..schema import BenchmarkPrompt, load_prompts
from ..utils.io import atomic_write_json
from ..utils.seeds import seed_all

log = logging.getLogger("mhb.generate.acestep")


def _init_acestep(device: str = "cuda"):
    """Initialize ACE-Step handlers. Returns (dit_handler, llm_handler)."""
    try:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
    except ImportError as e:
        raise ImportError(
            "ACE-Step not installed. Run:\n"
            "  pip install git+https://github.com/ace-step/ACE-Step-1.5.git"
        ) from e

    dit_handler = AceStepHandler()
    llm_handler = LLMHandler()

    # Find the project root for ACE-Step config
    import acestep
    acestep_root = Path(acestep.__file__).resolve().parent.parent

    log.info("Initializing ACE-Step DiT handler (device=%s)", device)
    init_status, enable_generate = dit_handler.initialize_service(
        project_root=str(acestep_root),
        config_path=None,
        device=device,
        use_flash_attention=True,
        compile_model=False,
        offload_to_cpu=False,
        offload_dit_to_cpu=False,
        quantization="none",
        prefer_source="huggingface",
    )
    log.info("DiT init: %s (generate_enabled=%s)", init_status, enable_generate)

    log.info("Initializing ACE-Step LLM handler")
    lm_status, lm_success = llm_handler.initialize(
        checkpoint_dir=None,
        lm_model_path=None,
        backend="transformers",
        device=device,
        offload_to_cpu=False,
        dtype="bfloat16",
    )
    log.info("LLM init: %s (success=%s)", lm_status, lm_success)

    return dit_handler, llm_handler


def generate_one(
    dit_handler,
    llm_handler,
    prompt_text: str,
    duration_sec: float = 30.0,
    seed: int = 848,
    inference_steps: int = 8,
    guidance_scale: float = 7.0,
    save_dir: Optional[str] = None,
) -> tuple:
    """Generate a single audio clip. Returns (audio_path, sample_rate)."""
    try:
        from acestep.pipeline_params import GenerationParams, GenerationConfig
        from acestep.generate_music import generate_music
    except ImportError:
        raise ImportError(
            "ACE-Step not installed. Run:\n"
            "  pip install git+https://github.com/ace-step/ACE-Step-1.5.git"
        )

    params = GenerationParams(
        caption=prompt_text,
        lyrics="",
        bpm=0,
        keyscale="",
        timesignature="4/4",
        vocal_language="",
        duration=duration_sec,
        thinking=False,
        use_cot_metas=False,
        use_cot_caption=False,
        use_cot_language=False,
        use_constrained_decoding=False,
        inference_steps=inference_steps,
        seed=seed,
        task_type="text2music",
        lm_temperature=0.85,
        lm_cfg_scale=2.0,
        guidance_scale=guidance_scale,
    )

    config = GenerationConfig(
        batch_size=1,
        seeds=[seed],
        use_random_seed=False,
        audio_format="wav",
    )

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="acestep_")

    result = generate_music(
        dit_handler, llm_handler, params, config, save_dir=save_dir,
    )

    if not result.success:
        raise RuntimeError(f"ACE-Step generation failed: {result.error}")

    # Find the output audio file
    audio_info = result.audios[0] if result.audios else {}
    audio_path = audio_info.get("path") or audio_info.get("audio_path")

    # If path not in result, scan save_dir for the generated file
    if not audio_path:
        wavs = list(Path(save_dir).glob("*.wav")) + list(Path(save_dir).glob("*.flac"))
        if wavs:
            audio_path = str(wavs[0])
        else:
            raise RuntimeError(f"No audio file found in {save_dir}")

    sample_rate = audio_info.get("sample_rate", 48000)
    return str(audio_path), sample_rate


def run_generation(
    prompts_path: Union[str, Path],
    output_dir: Union[str, Path],
    duration_sec: float = 30.0,
    inference_steps: int = 8,
    guidance_scale: float = 7.0,
    device: str = "cuda",
    seed: int = 848,
    resume: bool = True,
) -> list:
    """Generate audio for all prompts using ACE-Step 1.5."""
    seed_all(seed)
    import shutil
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

    dit_handler, llm_handler = _init_acestep(device)

    manifest = list(existing)
    for i, p in enumerate(tqdm(todo, desc="ACE-Step")):
        clip_seed = seed + i
        tmp_dir = tempfile.mkdtemp(prefix=f"acestep_{p.prompt_id}_")
        try:
            audio_path, sr = generate_one(
                dit_handler, llm_handler,
                prompt_text=p.generation_prompt,
                duration_sec=duration_sec,
                seed=clip_seed,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                save_dir=tmp_dir,
            )
            # Copy to output dir with canonical name
            out_path = output_dir / f"{p.prompt_id}.wav"
            if Path(audio_path).suffix == ".wav":
                shutil.copy2(audio_path, out_path)
            else:
                # Convert to WAV if needed
                y, orig_sr = sf.read(audio_path)
                sf.write(str(out_path), y, orig_sr)
                sr = orig_sr

            manifest.append({
                "prompt_id": p.prompt_id,
                "tradition": p.tradition,
                "generation_prompt": p.generation_prompt,
                "audio_path": str(out_path.resolve()),
                "model": "ACE-Step/Ace-Step1.5",
                "duration_sec": duration_sec,
                "sampling_rate": sr,
                "seed": clip_seed,
            })
            atomic_write_json(manifest_path, manifest)
        except Exception as e:
            log.error("Failed to generate %s: %s", p.prompt_id, e)
            continue
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info("ACE-Step generation complete → %s (%d clips)", output_dir, len(manifest))
    return manifest
