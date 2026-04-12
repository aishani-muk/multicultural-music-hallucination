"""Audio I/O and format conversion utilities."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
import soundfile as sf


def save_wav(audio: np.ndarray, path: str | Path, sr: int = 32000) -> Path:
    """Save a numpy array as a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
    return path


def load_wav(path: str | Path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Load a WAV file, optionally resampling."""
    import librosa
    y, orig_sr = librosa.load(str(path), sr=sr, mono=True)
    return y, sr if sr else orig_sr


def wav_to_base64(path: str | Path) -> str:
    """Read a WAV file and return its base64-encoded contents."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def numpy_to_wav_bytes(audio: np.ndarray, sr: int = 32000) -> bytes:
    """Convert a numpy array to WAV bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    buf.seek(0)
    return buf.read()
