"""Audio I/O + QC helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def load_wav(
    path: Union[str, Path],
    sr: Optional[int] = None,
    offset_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Load audio via librosa; resample if sr is given. Returns (y, sr)."""
    import librosa
    y, out_sr = librosa.load(
        str(path), sr=sr, mono=mono, offset=offset_sec, duration=duration_sec,
    )
    return y, out_sr


def save_wav(y: np.ndarray, path: Union[str, Path], sr: int) -> Path:
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)
    return path


def rms_dbfs(y: np.ndarray) -> float:
    """Return RMS in dBFS (0 = full scale). Silence -> -inf capped at -120."""
    if y.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(np.square(y.astype(np.float32))))
    if rms <= 0:
        return -120.0
    return float(20.0 * np.log10(rms))


def peak_dbfs(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    peak = float(np.max(np.abs(y.astype(np.float32))))
    if peak <= 0:
        return -120.0
    return float(20.0 * np.log10(peak))


def clip_quality_audit(
    y: np.ndarray,
    sr: int,
    target_duration_sec: float,
    min_rms_dbfs: float = -45.0,
    max_peak_dbfs: float = -0.1,
    duration_tolerance: float = 0.10,
) -> dict:
    """Run acoustic QC; return a dict of pass/fail flags + measured values."""
    dur = y.shape[-1] / sr if sr > 0 else 0.0
    rms = rms_dbfs(y)
    peak = peak_dbfs(y)
    dur_ok = abs(dur - target_duration_sec) <= target_duration_sec * duration_tolerance
    rms_ok = rms >= min_rms_dbfs
    peak_ok = peak <= max_peak_dbfs
    return {
        "duration_sec": round(dur, 3),
        "rms_dbfs": round(rms, 2),
        "peak_dbfs": round(peak, 2),
        "duration_ok": bool(dur_ok),
        "rms_ok": bool(rms_ok),
        "peak_ok": bool(peak_ok),
        "passed": bool(dur_ok and rms_ok and peak_ok),
    }


def extract_window(
    path: Union[str, Path],
    start_sec: float,
    end_sec: float,
    out_path: Union[str, Path],
    target_sr: int = 16000,
) -> Path:
    """Extract [start, end] window, resample to target_sr, save as WAV."""
    y, sr = load_wav(path, sr=target_sr, offset_sec=start_sec,
                     duration_sec=max(0.0, end_sec - start_sec))
    return save_wav(y, out_path, sr)
