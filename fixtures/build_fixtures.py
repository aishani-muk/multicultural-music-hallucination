"""Create a small WAV fixture (2s 440Hz tone) if not already present.

Idempotent. Invoked by `make fixtures` and by the smoke test setup.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


def main():
    here = Path(__file__).resolve().parent
    out = here / "audio" / "tone_440hz_2s.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"exists: {out}")
        return
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, 2.0, 2 * sr, endpoint=False)
    y = 0.2 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    sf.write(str(out), y, sr)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
