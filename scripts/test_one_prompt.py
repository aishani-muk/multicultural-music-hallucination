"""One-shot sanity test: ask every verifier ONE yes/no question on ONE audio clip.

Usage:
    python scripts/test_one_prompt.py --audio path/to/clip.wav \
        --question "Is a tabla audible?" --expected true

Runs whichever verifiers are installed + have credentials. Prints a table.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mhb.verify.base import VerifierResponse   # noqa: E402


def _try_gemini():
    if not os.environ.get("GOOGLE_API_KEY"):
        return None, "GOOGLE_API_KEY not set"
    try:
        from mhb.verify.gemini import GeminiVerifier
        return GeminiVerifier(model=os.environ.get("MHB_GEMINI_MODEL", "gemini-2.5-flash")), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _try_music_flamingo():
    try:
        from mhb.verify.music_flamingo import MusicFlamingoVerifier
        return MusicFlamingoVerifier(
            model_id=os.environ.get("MHB_MF_MODEL", "nvidia/audio-flamingo-3"),
            device=os.environ.get("MHB_DEVICE", "cuda"),
        ), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def _row(name: str, r: VerifierResponse, expected: bool | None) -> str:
    ans = "null" if r.answer is None else str(r.answer).lower()
    mark = "-" if r.answer is None or expected is None else ("✓" if bool(r.answer) == expected else "✗")
    lat = f"{r.latency_sec:.2f}s" if r.latency_sec else "?"
    return f"  {name:<40s}  answer={ans:<5s} conf={r.confidence:.2f} {mark} ({lat})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=str(ROOT / "fixtures/audio/tone_440hz_2s.wav"))
    ap.add_argument("--question", default="Is a sustained tonal sound audible in the clip?")
    ap.add_argument("--expected", choices=["true", "false", "null"], default="true")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Limit to named verifiers: gemini, music_flamingo")
    args = ap.parse_args()

    # Ensure fixture exists for default path.
    if not Path(args.audio).exists():
        if args.audio.endswith("tone_440hz_2s.wav"):
            import subprocess
            subprocess.run([sys.executable, str(ROOT / "fixtures/build_fixtures.py")], check=True)
        else:
            raise FileNotFoundError(args.audio)

    expected = {"true": True, "false": False, "null": None}[args.expected]
    candidates = {"gemini": _try_gemini, "music_flamingo": _try_music_flamingo}
    if args.only:
        candidates = {k: candidates[k] for k in args.only if k in candidates}

    print(f"\nAudio:    {args.audio}")
    print(f"Question: {args.question}")
    print(f"Expected: {expected}\n")

    for name, loader in candidates.items():
        v, err = loader()
        if v is None:
            print(f"  {name:<40s}  [skipped] {err}")
            continue
        try:
            r = v.verify(args.audio, args.question)
            print(_row(v.name, r, expected))
        except Exception as e:
            print(f"  {v.name:<40s}  [error] {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
