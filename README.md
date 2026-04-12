# Multicultural Music Hallucination Benchmark — Pipeline

An end-to-end pipeline for evaluating hallucination in generative music models,
using **Gemini** and **Qwen2-Audio / Audio Flamingo 2** as verifiers and
**FActScore-style atomic fact decomposition** adapted to audio generation.

## Architecture

```
prompts/seed_prompts.json     ← Curated prompts + atomic facts per tradition
        │
        ▼
┌──────────────────────┐
│  1. GENERATE         │  MusicGen (HuggingFace) → .wav clips
│  musicgen_generate   │  (also supports external clips from Suno/Udio)
└─────────┬────────────┘
          │  generation_manifest.json
          ▼
┌──────────────────────┐
│  2. VERIFY           │  For each clip × each atomic fact:
│  gemini_verifier     │   → binary question → JSON verdict
│  audio_llm_verifier  │   Gemini API + Qwen2-Audio / Audio Flamingo 2
└─────────┬────────────┘
          │  *_all_results.json
          ▼
┌──────────────────────┐
│  3. SCORE            │  Correctness / Hallucination / Abstention rates
│  hallucination_scorer│  Per-clip, per-tradition, per-category
└─────────┬────────────┘
          │  *_scores.json
          ▼
┌──────────────────────┐
│  4. ANALYZE          │  Cross-verifier agreement, breakdowns,
│  analysis            │  publication-ready figures (PNG/CSV)
└──────────────────────┘
```

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set Gemini API key
export GOOGLE_API_KEY="your-key-here"

# 3. Run full pipeline (generate + verify + score + analyze)
python run_pipeline.py --config configs/benchmark_config.yaml

# Or run individual stages:
python run_pipeline.py --config configs/benchmark_config.yaml --stage generate
python run_pipeline.py --config configs/benchmark_config.yaml --stage verify
python run_pipeline.py --config configs/benchmark_config.yaml --stage score
python run_pipeline.py --config configs/benchmark_config.yaml --stage analyze
```

## Using External Audio (Suno / Udio)

If you generate clips externally (Suno, Udio, etc.), create a `generation_manifest.json`:

```json
[
  {
    "prompt_id": "carnatic_001",
    "tradition": "carnatic",
    "generation_prompt": "...",
    "audio_path": "path/to/clip.wav",
    "model": "suno-v4",
    "duration_sec": 10,
    "sampling_rate": 44100
  }
]
```

Then run verification directly:

```bash
python run_pipeline.py --config configs/benchmark_config.yaml \
    --stage verify --manifest path/to/generation_manifest.json
```

## Module-Level Usage

### Generate audio
```bash
python -m generate.musicgen_generate \
    --prompts prompts/seed_prompts.json \
    --output outputs/generated_audio/ \
    --model facebook/musicgen-large \
    --duration 10
```

### Verify with Gemini
```bash
python -m verify.gemini_verifier \
    --manifest outputs/generated_audio/generation_manifest.json \
    --prompts prompts/seed_prompts.json \
    --output outputs/verification_results/gemini/
```

### Verify with Qwen2-Audio
```bash
python -m verify.audio_llm_verifier \
    --manifest outputs/generated_audio/generation_manifest.json \
    --prompts prompts/seed_prompts.json \
    --output outputs/verification_results/qwen2_audio/ \
    --backend qwen2-audio
```

### Verify with Audio Flamingo 2
```bash
python -m verify.audio_llm_verifier \
    --manifest outputs/generated_audio/generation_manifest.json \
    --prompts prompts/seed_prompts.json \
    --output outputs/verification_results/audio_flamingo/ \
    --backend audio-flamingo \
    --model_id /path/to/audio-flamingo-2-weights
```

### Score results
```bash
python -m evaluate.hallucination_scorer \
    --results outputs/verification_results/gemini/gemini_all_results.json \
    --output outputs/scores/gemini_scores.json
```

### Analyze & plot
```bash
python -m evaluate.analysis \
    --gemini_scores outputs/scores/gemini_scores.json \
    --audio_llm_scores outputs/scores/qwen2_audio_scores.json \
    --figures_dir outputs/figures/
```

## Prompt Schema

Each benchmark prompt in `prompts/seed_prompts.json` contains:

| Field | Description |
|---|---|
| `prompt_id` | Unique ID (e.g., `carnatic_001`) |
| `tradition` | `carnatic` / `hindustani` / `turkish_makam` / `arab_andalusian` / `western_control` |
| `generation_prompt` | Text prompt for MusicGen / Suno / Udio |
| `atomic_facts[]` | List of verifiable claims |
| `atomic_facts[].difficulty` | `must_have` or `easy_to_get_wrong` |
| `atomic_facts[].category` | `instrument` / `scale` / `rhythm` / `structure` / `timbre` / `tuning` / `cultural` |
| `atomic_facts[].verification_question` | Binary yes/no question for the verifier |
| `atomic_facts[].expected_answer` | Ground-truth boolean |

### Adding New Prompts

Edit `prompts/seed_prompts.json` or use the schema programmatically:

```python
from prompts.schema import BenchmarkPrompt, AtomicFact, save_prompts

prompt = BenchmarkPrompt(
    prompt_id="carnatic_003",
    tradition="carnatic",
    generation_prompt="A Carnatic flute solo in Raga Hamsadhwani",
    description="...",
    atomic_facts=[
        AtomicFact(
            fact_id="c003_f01", category="instrument",
            claim="A bamboo flute is the lead",
            verification_question="Is a bamboo flute the primary instrument?",
            expected_answer=True, difficulty="must_have",
        ),
    ],
)
save_prompts([prompt], "prompts/custom_prompts.json")
```

## Metrics

| Metric | Definition |
|---|---|
| **Correctness Rate** | Fraction of verified facts where model answer matches expected |
| **Hallucination Rate** | Fraction of verified facts where model answer is wrong |
| **Abstention Rate** | Fraction of facts where verifier could not determine answer |
| **Must-Have Correctness** | Correctness on `must_have` facts only |
| **Easy-to-Get-Wrong Error** | Error rate on `easy_to_get_wrong` facts only |

## Cluster Usage (NEXUS)

```bash
# Generate on GPU
sbatch --gres=gpu:a100:1 --mem=48G --time=4:00:00 \
    --wrap="python run_pipeline.py --config configs/benchmark_config.yaml --stage generate"

# Verify with Qwen2-Audio on GPU
sbatch --gres=gpu:a100:1 --mem=48G --time=6:00:00 \
    --wrap="python run_pipeline.py --config configs/benchmark_config.yaml --stage verify"

# Score + analyze (CPU-only)
python run_pipeline.py --config configs/benchmark_config.yaml --stage score
python run_pipeline.py --config configs/benchmark_config.yaml --stage analyze
```

## Directory Structure

```
pipeline/
├── README.md
├── requirements.txt
├── configs/
│   └── benchmark_config.yaml
├── prompts/
│   ├── schema.py                 # Prompt + AtomicFact dataclasses
│   └── seed_prompts.json         # 8 seed prompts across 5 traditions
├── generate/
│   └── musicgen_generate.py      # MusicGen audio generation
├── verify/
│   ├── gemini_verifier.py        # Gemini API verification
│   └── audio_llm_verifier.py     # Qwen2-Audio / Audio Flamingo 2
├── evaluate/
│   ├── hallucination_scorer.py   # Scoring engine
│   └── analysis.py               # Breakdowns + figures
├── utils/
│   └── audio_utils.py            # Audio I/O helpers
├── run_pipeline.py               # End-to-end orchestrator
└── outputs/                      # Created at runtime
    ├── generated_audio/
    ├── verification_results/
    ├── scores/
    └── figures/
```
