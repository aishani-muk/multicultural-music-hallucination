# Contributing

## Before you push
1. `pytest -q` must pass.
2. `python -m mhb smoke` must produce `SMOKE OK`.
3. Don't commit audio, model weights, API keys, or anything under `outputs/`.

## Adding a verifier
1. New file in `src/mhb/verify/<name>.py` with a class exposing
   `name: str` and `verify(audio_path, question, start_sec=None, end_sec=None) -> VerifierResponse`.
2. Register it in `src/mhb/cli.py::_build_verifier`.
3. Optional: add a SLURM `slurm/2X_verify_<name>.sbatch`.

## Adding a tradition or corpus
1. Extend `src/mhb/data/vocab/*.yaml` (or add new vocab files).
2. Add `src/mhb/data/auto_templates/<tradition>.yaml`.
3. Add a loader `src/mhb/data/<corpus>.py` that returns `ClipItem`s.
4. Register the loader in `src/mhb/data/curate.py::SOURCE_ALIASES`.
5. Add a test that curates a tiny fixture and checks tier/coverage.

## Commits
- Small, focused commits. One concern per commit.
- No `outputs/` or `.env` or `*.wav` outside `fixtures/audio/`.
- If you add a heavy dependency, put it behind an `extras` key in `pyproject.toml` and guard its import.
