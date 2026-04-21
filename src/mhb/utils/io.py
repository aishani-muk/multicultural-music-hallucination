"""Atomic JSON/JSONL IO + run-directory helpers."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


# ─────────────────────────────────────────────────────────────────────────
#  Atomic writes
# ─────────────────────────────────────────────────────────────────────────


def atomic_write_text(path: Path, text: str) -> Path:
    """Write text atomically (tmp file + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path, obj: Any, indent: int = 2) -> Path:
    return atomic_write_text(Path(path), json.dumps(obj, indent=indent, ensure_ascii=False))


def append_jsonl(path: Path, record: dict) -> Path:
    """Append a JSON record as one line. Non-atomic but crash-friendly for incremental writes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> Iterable[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ─────────────────────────────────────────────────────────────────────────
#  Run directories (timestamped, config-stamped)
# ─────────────────────────────────────────────────────────────────────────


def make_run_id(tag: str = "run") -> str:
    """Return a sortable run id: YYYYMMDD-HHMMSS-<tag>."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in tag)
    return f"{stamp}-{safe}"


def ensure_run_dir(base: Path, run_id: str) -> Path:
    d = Path(base) / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "logs").mkdir(exist_ok=True)
    return d


def write_run_manifest(run_dir: Path, config: dict, extras: Optional[dict] = None) -> Path:
    """Drop a reproducibility manifest into the run directory."""
    manifest = {
        "run_id": run_dir.name,
        "utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "python": sys.version,
        "argv": sys.argv,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
            "SLURM_NODELIST": os.environ.get("SLURM_NODELIST"),
        },
        "git_sha": _git_sha(),
        "packages": _pkg_versions(),
        "config": config,
    }
    if extras:
        manifest["extras"] = extras
    return atomic_write_json(run_dir / "run_manifest.json", manifest)


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, cwd=Path(__file__).resolve().parent,
        )
        return out.decode().strip()
    except Exception:
        return None


def _pkg_versions() -> dict:
    names = ["torch", "transformers", "accelerate", "librosa", "soundfile",
             "google.genai", "numpy", "pandas", "scipy", "pyyaml"]
    out = {}
    for n in names:
        try:
            mod = __import__(n)
            out[n] = getattr(mod, "__version__", "?")
        except Exception:
            out[n] = None
    return out


# ─────────────────────────────────────────────────────────────────────────
#  File hashing
# ─────────────────────────────────────────────────────────────────────────


def sha1_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def timer():
    """Context-friendly wall-clock timer.

    Usage:
        with timer() as t: ...
        t.elapsed  -> seconds (float)
    """
    class _T:
        def __enter__(self):
            self._t0 = time.perf_counter(); return self
        def __exit__(self, *a):
            self.elapsed = time.perf_counter() - self._t0
    return _T()
