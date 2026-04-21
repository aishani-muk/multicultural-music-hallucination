"""Gemini (Google GenAI) verifier.

Uses the `google-genai` Python SDK with audio inputs. Defaults to
`gemini-2.5-flash`. Provide the key via $GOOGLE_API_KEY.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional, Union

from ..utils.audio import extract_window
from .base import (
    SYSTEM_PROMPT, USER_TEMPLATE, VerifierResponse,
    parse_json_response, with_retries,
)

log = logging.getLogger("mhb.verify.gemini")


class GeminiVerifier:
    """Verifier backed by the Gemini API."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        rpm_limit: int = 15,
        max_retries: int = 3,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        cache_dir: Optional[Union[str, Path]] = None,
    ):
        try:
            from google import genai  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "google-genai not installed. `pip install google-genai`."
            ) from e

        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Set GOOGLE_API_KEY env var or pass api_key to GeminiVerifier."
            )
        from google import genai
        self._genai = genai
        self._types = __import__("google.genai.types", fromlist=["types"])
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.name = f"gemini/{model}"
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

    def _audio_bytes_for(self, audio_path: Path,
                         start_sec: Optional[float],
                         end_sec: Optional[float]) -> bytes:
        """Return raw bytes of the full audio OR a trimmed window if start/end given."""
        audio_path = Path(audio_path)
        if start_sec is None and end_sec is None:
            return audio_path.read_bytes()

        # Trim a window to keep upload small + faithful to the clip.
        if self.cache_dir is not None:
            cached = (self.cache_dir
                      / f"{audio_path.stem}_{int(start_sec or 0):06d}_{int(end_sec or 0):06d}.wav")
        else:
            import tempfile
            cached = Path(tempfile.gettempdir()) / (
                f"mhb_{audio_path.stem}_{int(start_sec or 0)}_{int(end_sec or 0)}.wav"
            )
        if not cached.exists():
            extract_window(audio_path, float(start_sec or 0.0),
                           float(end_sec or 0.0), cached, target_sr=16000)
        return cached.read_bytes()

    def verify(
        self,
        audio_path: Union[str, Path],
        question: str,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
    ) -> VerifierResponse:
        audio_bytes = self._audio_bytes_for(Path(audio_path), start_sec, end_sec)

        types = self._types
        user_content = [
            types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
            types.Part.from_text(text=USER_TEMPLATE.format(question=question)),
        ]

        def _call():
            self._rate_limit()
            t0 = time.perf_counter()
            response = self.client.models.generate_content(
                model=self.model,
                contents=[types.Content(role="user", parts=user_content)],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                ),
            )
            elapsed = time.perf_counter() - t0
            text = getattr(response, "text", "") or ""
            parsed = parse_json_response(text)
            parsed.latency_sec = elapsed
            return parsed

        return with_retries(_call, max_retries=self.max_retries, logger=log)
