# Data — sources, schema, tiering

## Sources

| Tradition | Source | License | Audio? | Destination on NEXUS |
|---|---|---|---|---|
| Hindustani | Saraga 1.5 Hindustani (Zenodo 4301737) | CC-BY-NC-SA-4.0 | Yes | `/fs/classhomes/amukherj/research/saraga1.5_hindustani.zip` |
| Carnatic | Saraga 1.5 Carnatic (Zenodo 4301737) | CC-BY-NC-SA-4.0 | Yes | `/fs/classhomes/amukherj/research/saraga1.5_carnatic.zip` |
| Arab-Andalusian | CompMusic Moroccan Andalusian (Zenodo 1291776) | CC-BY-NC-SA-4.0 | Yes (mp3) | `/fs/classhomes/amukherj/research/ArabAndalusianDataset.tar.gz` |
| Turkish makam | CompMusic OTMM (Zenodo 58408) | CC-BY-NC-SA-4.0 | **No audio** — metadata + pitch tracks only (copyright-restricted) | `/fs/classhomes/amukherj/research/otmm_makam_recognition_dataset.zip` |
| Reggaeton | MTG-Jamendo `reggaeton` tag | CC-BY-NC-SA-4.0 | Yes | `/fs/classhomes/amukherj/research/mtg_jamendo_reggaeton/` |
| Iranian radif | (placeholder) | unknown | — | Skeleton only — loader returns `[]` |
| Western control | — | — | — | Generation-track only |

Download everything in one shot:
```bash
sbatch slurm/-2_download_datasets.sbatch all
# or a single corpus:
bash scripts/download_datasets.sh carnatic
```

The script resumes interrupted transfers (`wget -c`) and is idempotent — re-running is safe.

**Turkish Makam audio caveat.** The OTMM dataset distributes *no audio* due to copyright; it provides MBIDs + makam labels + pitch tracks only. For evaluation we currently (a) use the pitch-track-driven synthetic facts only, or (b) source a handful of openly-licensed Turkish makam clips (e.g. public-domain historical recordings on Archive.org) and tag them against OTMM's makam list. This limitation is documented in `audit_report.json` produced by `mhb data curate --source compmusic_turkish_makam`.

We do **not redistribute audio**. Manifests store absolute paths; reruns require the raw files on disk.

## Ground truth per clip

Every curated `ClipItem` carries `GroundTruth` with the fields that are *distinctive* to each tradition:

- `modal_system`: raga / makam / ṭabʿ / dastgāh (via controlled vocab)
- `metric_cycle`: tala / usul / mizān / dembow with beat count
- `form`: kriti / khayal / taksim / nuba / reggaeton_song / ...
- `sections`: time-aligned spans (when corpus provides them)
- `instruments`: canonical names from `vocab/instruments.yaml`
- `tonic_hz`, `tempo_bpm`, `language`, `artist_id`, `provenance`

See `src/mhb/schema.py`.

## Tiering (A / B / C)

Computed by `data.segmentation.assign_tier`:

- **A** modal + cycle + form + ≥2 instruments + ≥1 section annotation
- **B** modal + cycle + ≥2 instruments
- **C** (modal OR cycle) + ≥1 instrument

Clips below C are dropped.

## Segmentation

Hybrid: section-aligned when sections exist, else fixed 30s windows with 60s stride, capped at `max_windows_per_track`.

## Splits

Artist-disjoint eval/retrieval/held-back split (`data.splits.artist_disjoint_split`). Eval caps clips per artist; retrieval corpus never shares an artist with eval.

## Atomic facts (auto-synthesis)

Facts are emitted by `data.facts.synthesize_facts` from three sources:

1. Per-tradition YAML templates (`data/auto_templates/*.yaml`)
2. Positive "is X present?" per annotated instrument
3. Negative "is X present?" for instruments whose `out_of_culture_for` list contains this tradition

Every fact records `source` and `grounding` so you can audit which metadata field produced it.

## Adding a tradition / raga / instrument

1. Append to the relevant YAML in `src/mhb/data/vocab/`.
2. Add templates in `src/mhb/data/auto_templates/<tradition>.yaml` if it's a new tradition.
3. Run `python scripts/regenerate_seed_prompts.py` if you want it in the generation track.
4. Re-run `mhb data curate` to re-derive facts on curated clips.

## Vocabulary misses

Loaders log unknown metadata values into `audit_report.json` under `vocab_misses`. Review that file after every curation run.
