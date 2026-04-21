"""End-to-end smoke test: CLI `mhb smoke` runs without any models / network."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cli_smoke_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(ROOT / "src"))
    # Ensure fixture audio exists (creates a 2s 440Hz tone).
    subprocess.run([sys.executable, str(ROOT / "fixtures" / "build_fixtures.py")],
                   check=True)
    r = subprocess.run(
        [sys.executable, "-m", "mhb", "smoke"],
        cwd=ROOT,
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"smoke failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    assert "SMOKE OK" in r.stdout


def test_cli_help_lists_subcommands():
    r = subprocess.run(
        [sys.executable, "-m", "mhb", "--help"],
        cwd=ROOT,
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert r.returncode == 0
    for sub in ("data", "generate", "verify", "score", "analyze", "smoke"):
        assert sub in r.stdout
