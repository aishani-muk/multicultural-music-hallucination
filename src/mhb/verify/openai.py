"""OpenAI (GPT-4o-audio-preview) verifier.

Uses the OpenAI Python SDK with audio inputs via base64 encoding.
Defaults to `gpt-4o-audio-preview` which supports audio input.
Provide the key via $OPENAI_API_KEY.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

from ..utils.audio import extract_window
from .base import (
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    VerifierResponse,
    parse_json_response,
    with_retries,
)

log = logging.getLogger("mhb.verify.openai")


class OpenAIVerifier:
    """Verifier backed by the OpenAI Chat Completions API with audio input."""

    def __init__(
        self,
        model: str = "gpt-4o-audio-preview",
        api_key: Optional[str] = None,
        rpm_limit: int = 30,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        try:
            import openai  # noqa: F401
        except ImportError as e:
            raise ImportError("openai not installed. `pip install openai`.") from e

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY env var or pass api_key to OpenAIVerifier."
            )
        import openai

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.name = f"openai/{model}"
        self.rpm_limit = rpm_limit
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call_ts: float = 0.0

    def _rate_limit(self):
        if self.rpm_limit <= 0:
            return
        min_interval = 60.0 / float(self.rpm_limit)
        elapsed = time.time() - self._last_call_ts
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_ts = time.time()

    def _audio_bytes_for(
        self, audio_path: Path, start_sec: Optional[float], end_sec: Optional[float]
    ) -> bytes:
        """Return raw bytes of the full audio OR a trimmed window."""
        audio_path = Path(audio_path)
        if start_sec is None and end_sec is None:
            return audio_path.read_bytes()

        if self.cache_dir is not None:
            cached = (
                self.cache_dir
                / f"{audio_path.stem}_{int(start_sec or 0):06d}_{int(end_sec or 0):06d}.wav"
            )
        else:
            import tempfile

            cached = Path(tempfile.gettempdir()) / (
                f"mhb_{audio_path.stem}_{int(start_sec or 0)}_{int(end_sec or 0)}.wav"
            )
        if not cached.exists():
            extract_window(
                audio_path,
                float(start_sec or 0.0),
                float(end_sec or 0.0),
                cached,
                target_sr=16000,
            )
        return cached.read_bytes()

    def verify(
        self,
        audio_path: Union[str, Path],
        question: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> VerifierResponse:
        audio_bytes = self._audio_bytes_for(Path(audio_path), start_sec, end_sec)
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav",
                        },
                    },
                    {
                        "type": "text",
                        "text": USER_TEMPLATE.format(question=question),
                    },
                ],
            },
        ]

        def _call():
            self._rate_limit()
            t0 = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.model,
                modalities=["text"],
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            elapsed = time.perf_counter() - t0
            text = response.choices[0].message.content or ""
            parsed = parse_json_response(text)
            parsed.latency_sec = elapsed
            return parsed

        return with_retries(_call, max_retries=self.max_retries, logger=log)
