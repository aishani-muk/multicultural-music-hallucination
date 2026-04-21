"""NVIDIA Music Flamingo verifier (HuggingFace).

Music Flamingo (https://huggingface.co/nvidia/audio-flamingo-3 and the
music-specialized variant) uses a chat-template interface that takes an
audio array + a text prompt. The exact HF repo id may shift as NVIDIA
rotates releases; we expose `model_id` as a parameter.

This loader will raise loudly if it cannot load the model — we explicitly
do NOT silently fall back to another verifier, because doing so makes the
Gemini-vs-Music-Flamingo comparison meaningless.
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


DEFAULT_MODEL_ID = "nvidia/audio-flamingo-3"   # music-capable; swap as NVIDIA publishes
ALTERNATE_MODEL_IDS = [
    "nvidia/music-flamingo-hf",
    "nvidia/music-flamingo",
    "nvidia/audio-flamingo-3",
]


class MusicFlamingoVerifier:
    """Open-source audio-LLM verifier using NVIDIA Music/Audio Flamingo."""

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
            from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "transformers + torch required. `pip install transformers torch`"
            ) from e

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        dtype = getattr(torch, torch_dtype, torch.bfloat16)
        hf_token = hf_token or os.environ.get("HF_TOKEN")

        log.info("Loading Music Flamingo: %s  (device=%s, dtype=%s)",
                 model_id, device, torch_dtype)
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, token=hf_token,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=dtype,
                device_map=device, trust_remote_code=True, token=hf_token,
            )
        except Exception as e:
            hint = (
                "Failed to load Music Flamingo. Check:\n"
                "  1) HF_TOKEN is set and you accepted the model license.\n"
                "  2) `transformers` is recent enough.\n"
                "  3) Try an alternate model_id: "
                f"{ALTERNATE_MODEL_IDS}\n"
                "If you want to smoke-test without this model, use the \n"
                "`--no-music-flamingo` CLI flag."
            )
            raise RuntimeError(hint) from e

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
