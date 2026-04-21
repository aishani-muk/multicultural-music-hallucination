# mhb — Multicultural Music-Hallucination Benchmark

Dual-track evaluation of **Gemini** and **Music Flamingo** on music hallucination:

1. **Understanding track**: real multicultural audio (Saraga / CompMusic / MTG-Jamendo reggaeton) with culture-specific ground truth (raga, tala, makam, usul, ṭabʿ, mizān, dembow, form, instruments).
2. **Generation track**: MusicGen outputs scored against auto-derived atomic facts from the same vocabulary.

Culturally grounded atomic facts are auto-synthesized from corpus metadata + controlled vocabularies + per-tradition templates — no hand-authored claims.

## Install

```bash
bash scripts/bootstrap.sh          # creates .env, pip install, fixtures, smoke
# or on NEXUS:
conda env create -f environment.yml && conda activate mhb
```

Set `GOOGLE_API_KEY` (Gemini) and `HF_TOKEN` (Music Flamingo) in `.env`.

## Smoke test (no models, no network)

```bash
python -m mhb smoke
```

Writes a timestamped run directory under `outputs/runs/` with scorer output, figures, and a reproducibility manifest.

## One-prompt sanity check (real models)

```bash
source setup_env.sh
python scripts/test_one_prompt.py \
    --audio fixtures/audio/tone_440hz_2s.wav \
    --question "Is a sustained tonal sound audible?" \
    --expected true
```

Runs every verifier that has credentials + dependencies installed.

## Full pipeline

```bash
python -m mhb data unpack  --zip "$SARAGA_HINDUSTANI_ZIP" --out outputs/data/raw/saraga_hindustani
python -m mhb data curate  --source saraga_hindustani --raw outputs/data/raw/saraga_hindustani --out outputs/data/curated
python -m mhb data audit   --curated outputs/data/curated

python -m mhb generate --config configs/generation.yaml
python -m mhb verify   --config configs/understanding.yaml --verifier gemini music_flamingo \
                       --target clips --clips outputs/data/curated/hindustani/clips.jsonl
python -m mhb score    --results outputs/verification_results/gemini/*_all_results.json \
                       --out outputs/scores/gemini_scores.json
python -m mhb analyze  --scores "gemini=outputs/scores/gemini_scores.json" \
                                 "music_flamingo=outputs/scores/music_flamingo_scores.json" \
                       --out outputs/figures
```

SLURM templates for each step live in `slurm/`.

## Layout

```
pipeline/
├── src/mhb/               # package
│   ├── cli.py             # `python -m mhb ...`
│   ├── schema.py          # ClipItem, GroundTruth, AtomicFact, ...
│   ├── data/              # vocab, templates, loaders, segmentation, splits
│   ├── verify/            # base, gemini, music_flamingo
│   ├── generate/          # MusicGen
│   ├── evaluate/          # scorer (bootstrap CIs), analysis, muchomusic
│   ├── mitigations/       # abstain, cove, rarr stub
│   ├── probe/             # CLAP (optional)
│   └── prompts/           # auto-derived seed prompts
├── configs/               # default / understanding / generation / full
├── slurm/                 # NEXUS sbatch scripts
├── scripts/               # bootstrap, regenerate prompts, one-prompt test
├── fixtures/              # CPU-only smoke data
└── tests/                 # pytest
```

## Docs

- `QUICKSTART.md` — 5-minute onboarding
- `DATA.md` — corpora, ground-truth schema, tiering
- `CONTRIBUTING.md` — adding a tradition / prompt / verifier
