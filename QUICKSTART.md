# QUICKSTART

For your first session after cloning.

## 1. Install + smoke (5 min, no GPU)

```bash
bash scripts/bootstrap.sh
```

You should see `SMOKE OK. Run dir: outputs/runs/<timestamp>-smoke`.

## 2. Add credentials

Edit `.env`:
```
GOOGLE_API_KEY=your-key-from-aistudio.google.com/apikey
HF_TOKEN=your-hf-token
```

Accept the HF license for `nvidia/audio-flamingo-3` (or whichever Music Flamingo variant the config points to).

## 3. Sanity-check a real model

```bash
source setup_env.sh
python scripts/test_one_prompt.py --only gemini \
    --question "Is a sustained tonal sound audible?" --expected true
```

If Gemini answers `true` with high confidence, the plumbing works.

## 4. Where things go

- Curated data → `outputs/data/curated/<tradition>/clips.jsonl`
- Generated audio → `outputs/generated_audio/`
- Verification → `outputs/verification_results/<verifier>/`
- Scores → `outputs/scores/`
- Figures + analysis → `outputs/figures/`
- Each `mhb smoke` run → `outputs/runs/<timestamp>-smoke/` with logs + manifest

## 5. NEXUS (SLURM)

```bash
sbatch slurm/99_smoke.sbatch           # no GPU, 2 min
sbatch slurm/00_unpack_saraga.sbatch   # no GPU, ~20 min
sbatch slurm/01_curate_hindustani.sbatch
sbatch slurm/10_generate_musicgen.sbatch       # A100
sbatch slurm/20_verify_gemini.sbatch           # CPU, rate-limited
sbatch slurm/21_verify_music_flamingo.sbatch   # A100
sbatch slurm/40_score_analyze.sbatch
```

## 6. Tests

```bash
pytest -q
```
