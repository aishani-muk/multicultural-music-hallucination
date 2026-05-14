"""NVIDIA Audio Flamingo 3 verifier (HuggingFace).

Supports two HF checkpoint formats:
  - nvidia/audio-flamingo-3-hf  (native HF, needs transformers >= 4.52)
  - nvidia/audio-flamingo-3     (trust_remote_code, works on older transformers)

This loader will raise loudly if it cannot load the model — we explicitly
do NOT silently fall back to another verifier, because doing so makes the
cross-verifier comparison meaningless.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

from .base import (
    SYSTEM_PROMPT, USER_TEMPLATE, VerifierResponse,
    parse_json_response, with_retries,
)

log = logging.getLogger("mhb.verify.music_flamingo")


DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3-hf"
ALTERNATE_MODEL_IDS = [
    "nvidia/audio-flamingo-3-hf",
    "nvidia/audio-flamingo-3",
]


def _load_model_and_processor(model_id: str, dtype, device: str, hf_token: Optional[str]):
    """Try native HF class first, fall back to Auto* with trust_remote_code."""
    # Attempt 1: native AudioFlamingo3 class (transformers >= 4.52)
    try:
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
        log.info("Using native AudioFlamingo3ForConditionalGeneration")
        processor = AutoProcessor.from_pretrained(model_id, token=hf_token)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, token=hf_token,
        )
        return model, processor
    except (ImportError, Exception) as e:
        log.info("Native class not available (%s), trying trust_remote_code", e)

    # Attempt 2: Auto* with trust_remote_code (works on older transformers)
    from transformers import AutoModelForCausalLM, AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device,
        trust_remote_code=True, token=hf_token,
    )
    return model, processor


class MusicFlamingoVerifier:
    """Open-source audio-LLM verifier using NVIDIA Audio Flamingo 3."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 256,
        sample_rate: int = 16000,
        hf_token: Optional[str] = None,
    ):
        try:
            import torch  # noqa: F401
            from transformers import AutoProcessor  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "transformers + torch required. `pip install transformers torch`"
            ) from e

        import torch
        dtype = getattr(torch, torch_dtype, torch.bfloat16)
        hf_token = hf_token or os.environ.get("HF_TOKEN")

        log.info("Loading Audio Flamingo: %s  (device=%s, dtype=%s)",
                 model_id, device, torch_dtype)
        try:
            self.model, self.processor = _load_model_and_processor(
                model_id, dtype, device, hf_token,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Audio Flamingo ({model_id}).\n"
                f"  Underlying error: {type(e).__name__}: {e}\n"
                f"  Fix: pip install --upgrade transformers\n"
                f"  Or:  pip install git+https://github.com/huggingface/transformers.git\n"
                f"  Alternates: {ALTERNATE_MODEL_IDS}"
            ) from e

        self.model.eval()
        self.device = device
        self.model_id = model_id
        self.name = f"music_flamingo/{model_id}"
        self.max_new_tokens = max_new_tokens
        self.sample_rate = sample_rate

    def _load_audio(self, audio_path: Path,
                    start_sec: Optional[float],
                    end_sec: Optional[float]):
        import librosa
        offset = float(start_sec or 0.0)
        duration = (float(end_sec) - offset) if end_sec is not None else None
        y, sr = librosa.load(str(audio_path), sr=self.sample_rate,
                             mono=True, offset=offset, duration=duration)
        return y, sr

    def verify(
        self,
        audio_path: Union[str, Path],
        question: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> VerifierResponse:
        import torch

        audio, sr = self._load_audio(Path(audio_path), start_sec, end_sec)

        conversation = [
            {"role": "system",
             "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user",
             "content": [
                 {"type": "audio", "audio_url": str(audio_path)},
                 {"type": "text",
                  "text": USER_TEMPLATE.format(question=question)},
             ]},
        ]

        text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False,
        )

        def _call():
            t0 = time.perf_counter()
            inputs = self.processor(
                text=text, audios=[audio], return_tensors="pt",
                sampling_rate=sr, padding=True,
            )
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v)
                      for k, v in inputs.items()}

            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1] \
                if "input_ids" in inputs else 0
            new_ids = out_ids[:, input_len:]
            decoded = self.processor.batch_decode(
                new_ids, skip_special_tokens=True,
            )[0]
            elapsed = time.perf_counter() - t0
            parsed = parse_json_response(decoded)
            parsed.latency_sec = elapsed
            return parsed

        return with_retries(_call, max_retries=2, logger=log)
