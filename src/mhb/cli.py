"""mhb CLI: `python -m mhb <subcommand> ...`

Subcommands:

    mhb data unpack   --zip PATH --out DIR
    mhb data curate   --source <SOURCE> --raw DIR --out DIR
    mhb data audit    --curated DIR
    mhb data splits   --curated DIR --out splits.json

    mhb generate      --config PATH --out DIR
    mhb verify        --config PATH --target {prompts,clips} --clips PATH|--manifest PATH
    mhb score         --results PATH --out PATH
    mhb analyze       --scores "label=path" ... --out DIR
    mhb muchomusic    --config PATH --out DIR  [--max-items N]
    mhb smoke         --fixtures             (CPU-only end-to-end fixture run)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .utils.io import atomic_write_json, ensure_run_dir, make_run_id, write_run_manifest
from .utils.logging import get_logger
from .utils.seeds import seed_all


def _load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ─────────────────────────────────────────────────────────────────────────
#  data sub-commands
# ─────────────────────────────────────────────────────────────────────────


def _cmd_data_unpack(args):
    import zipfile
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log = get_logger("mhb.cli")
    log.info("Unpacking %s → %s", args.zip, out)
    with zipfile.ZipFile(args.zip) as zf:
        zf.extractall(out)
    log.info("Done. Entries: %d", sum(1 for _ in out.rglob("*")))


def _cmd_data_curate(args):
    from .data.curate import curate_source
    res = curate_source(
        source=args.source,
        raw_root=Path(args.raw),
        out_dir=Path(args.out),
        window_sec=args.window_sec,
        max_windows_per_track=args.max_windows,
        max_negative_instruments=args.max_negatives,
        sample_rate=args.sample_rate,
        seed=args.seed,
    )
    print(json.dumps(res, indent=2, default=str))


def _cmd_data_audit(args):
    from .data.curate import audit_all
    out = audit_all(Path(args.curated))
    path = Path(args.curated) / "AUDIT.json"
    atomic_write_json(path, out)
    print(json.dumps(out.get("total", {}).get("tiers", {}), indent=2))
    print(f"\nFull audit → {path}")


def _cmd_data_splits(args):
    from .data.curate import make_splits
    make_splits(Path(args.curated), Path(args.out),
                eval_fraction=args.eval_fraction,
                retrieval_fraction=args.retrieval_fraction,
                max_clips_per_artist=args.max_clips_per_artist,
                seed=args.seed)


# ─────────────────────────────────────────────────────────────────────────
#  generate / verify / score / analyze
# ─────────────────────────────────────────────────────────────────────────


def _cmd_generate(args):
    from .generate.musicgen import run_generation
    cfg = _load_config(args.config)
    gen = cfg.get("generation", {})
    out = Path(args.out or cfg.get("paths", {}).get("generated_audio", "outputs/generated_audio"))
    run_generation(
        prompts_path=args.prompts or cfg["paths"]["prompts"],
        output_dir=out,
        model_id=gen.get("model", "facebook/musicgen-large"),
        duration_sec=gen.get("duration_sec", 10.0),
        guidance_scale=gen.get("guidance_scale", 3.0),
        max_new_tokens=gen.get("max_new_tokens", 1024),
        batch_size=gen.get("batch_size", 4),
        device=gen.get("device", "cuda"),
        torch_dtype=gen.get("torch_dtype", "float16"),
        seed=cfg.get("seed", 848),
    )


def _build_verifier(kind: str, cfg: dict):
    kind = kind.lower()
    if kind == "gemini":
        from .verify.gemini import GeminiVerifier
        g = cfg.get("gemini", {})
        return GeminiVerifier(
            model=g.get("model", "gemini-2.5-flash"),
            api_key=None,
            rpm_limit=g.get("requests_per_minute", 15),
            max_retries=g.get("max_retries", 3),
            temperature=g.get("temperature", 0.0),
        )
    if kind == "music_flamingo":
        from .verify.music_flamingo import MusicFlamingoVerifier
        mf = cfg.get("music_flamingo", {})
        return MusicFlamingoVerifier(
            model_id=mf.get("model_id", "nvidia/audio-flamingo-3"),
            device=mf.get("device", "cuda"),
            torch_dtype=mf.get("torch_dtype", "bfloat16"),
            max_new_tokens=mf.get("max_new_tokens", 256),
        )
    raise ValueError(f"Unknown verifier kind: {kind}. Use gemini|music_flamingo.")


def _cmd_verify(args):
    cfg = _load_config(args.config)
    out_dir = Path(args.out or cfg.get("paths", {}).get("verification_results", "outputs/verification_results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    kinds = args.verifier or cfg.get("verifiers", ["gemini"])
    if isinstance(kinds, str):
        kinds = [kinds]

    target = args.target
    for kind in kinds:
        try:
            verifier = _build_verifier(kind, cfg)
        except Exception as e:
            print(f"[skip] verifier '{kind}' failed to load: {e}", file=sys.stderr)
            continue

        if target == "clips":
            from .schema import load_clip_items
            from .verify.run import run_verification_over_clips
            clips = load_clip_items(args.clips)
            run_verification_over_clips(
                verifier, clips, out_dir / kind,
                max_facts_per_clip=args.max_facts_per_clip,
            )
        else:
            from .schema import load_prompts
            from .verify.run import run_verification_over_prompts
            prompts = load_prompts(args.prompts or cfg["paths"]["prompts"])
            manifest = json.loads(Path(args.manifest).read_text())
            run_verification_over_prompts(verifier, prompts, manifest, out_dir / kind)


def _cmd_score(args):
    from .evaluate.scorer import score_verification_file
    score_verification_file(
        args.results, args.out,
        bootstrap_n=args.bootstrap_n, alpha=args.alpha, seed=args.seed,
    )


def _cmd_analyze(args):
    from .evaluate.analysis import run_analysis
    score_paths = {}
    for spec in args.scores:
        if "=" not in spec:
            raise SystemExit(f"--scores entries must be label=path; got {spec!r}")
        label, path = spec.split("=", 1)
        score_paths[label] = Path(path)
    run_analysis(score_paths, Path(args.out))


def _cmd_muchomusic(args):
    cfg = _load_config(args.config)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    from .evaluate.muchomusic_eval import run_muchomusic
    kinds = args.verifier or cfg.get("verifiers", ["gemini"])
    if isinstance(kinds, str):
        kinds = [kinds]
    for kind in kinds:
        try:
            verifier = _build_verifier(kind, cfg)
        except Exception as e:
            print(f"[skip] verifier '{kind}' failed: {e}", file=sys.stderr)
            continue
        run_muchomusic(verifier, out_dir,
                       split=args.split, max_items=args.max_items)


# ─────────────────────────────────────────────────────────────────────────
#  smoke
# ─────────────────────────────────────────────────────────────────────────


def _cmd_smoke(args):
    """Run the scorer+analysis path end-to-end against fixtures (CPU, no network).

    Exercises: schema load -> segmentation math -> fact synthesis ->
    scorer -> analysis -> figures. No model calls.
    """
    from .schema import load_clip_items
    from .evaluate.scorer import score_verification_file
    from .evaluate.analysis import run_analysis

    root = Path(__file__).resolve().parents[2]
    fixtures = root / "fixtures"
    run_id = make_run_id("smoke")
    out_dir = ensure_run_dir(root / "outputs" / "runs", run_id)

    log = get_logger("mhb.smoke", log_file=out_dir / "logs" / "smoke.log")
    seed_all(848)
    log.info("Smoke run_id=%s", run_id)
    log.info("Fixtures dir: %s", fixtures)

    # 1. Scorer on fixture results JSON
    results_a = fixtures / "verification_results_gemini.json"
    results_b = fixtures / "verification_results_music_flamingo.json"
    scores_dir = out_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    sa = score_verification_file(results_a, scores_dir / "gemini_scores.json",
                                 bootstrap_n=100, alpha=0.05, seed=848)
    sb = score_verification_file(results_b, scores_dir / "music_flamingo_scores.json",
                                 bootstrap_n=100, alpha=0.05, seed=848)

    # 2. Analysis over both
    run_analysis(
        {"gemini": scores_dir / "gemini_scores.json",
         "music_flamingo": scores_dir / "music_flamingo_scores.json"},
        out_dir / "figures",
    )

    # 3. ClipItem load sanity
    clips = load_clip_items(fixtures / "clips_mini.jsonl")
    log.info("Loaded %d fixture clips", len(clips))
    assert clips and clips[0].derived_facts, "fixture clips should carry facts"

    # 4. Manifest
    write_run_manifest(out_dir, {"smoke": True, "fixtures_dir": str(fixtures)})

    print(f"\nSMOKE OK. Run dir: {out_dir}")


# ─────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mhb",
        description="Multicultural Music-Hallucination Benchmark CLI",
    )
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="cmd", required=True)

    # data
    d = sub.add_parser("data", help="Dataset curation")
    dsub = d.add_subparsers(dest="data_cmd", required=True)

    du = dsub.add_parser("unpack", help="Unpack a corpus zip")
    du.add_argument("--zip", required=True)
    du.add_argument("--out", required=True)
    du.set_defaults(func=_cmd_data_unpack)

    dc = dsub.add_parser("curate", help="Curate a source into ClipItems")
    dc.add_argument("--source", required=True,
                    choices=["saraga_hindustani", "saraga_carnatic",
                             "compmusic_turkish_makam", "compmusic_arab_andalusian",
                             "mtg_jamendo_reggaeton", "radif"])
    dc.add_argument("--raw", required=True, help="Path to raw corpus dir or manifest")
    dc.add_argument("--out", required=True, help="Curated output root")
    dc.add_argument("--window-sec", type=float, default=25.0)
    dc.add_argument("--max-windows", type=int, default=4)
    dc.add_argument("--max-negatives", type=int, default=3)
    dc.add_argument("--sample-rate", type=int, default=16000)
    dc.add_argument("--seed", type=int, default=848)
    dc.set_defaults(func=_cmd_data_curate)

    da = dsub.add_parser("audit", help="Summarize curated dataset coverage")
    da.add_argument("--curated", required=True)
    da.set_defaults(func=_cmd_data_audit)

    ds = dsub.add_parser("splits", help="Artist-disjoint eval/retrieval split")
    ds.add_argument("--curated", required=True)
    ds.add_argument("--out", required=True)
    ds.add_argument("--eval-fraction", type=float, default=0.6)
    ds.add_argument("--retrieval-fraction", type=float, default=0.3)
    ds.add_argument("--max-clips-per-artist", type=int, default=3)
    ds.add_argument("--seed", type=int, default=848)
    ds.set_defaults(func=_cmd_data_splits)

    # generate
    g = sub.add_parser("generate", help="Generate audio with MusicGen")
    g.add_argument("--config", required=True)
    g.add_argument("--prompts", default=None)
    g.add_argument("--out", default=None)
    g.set_defaults(func=_cmd_generate)

    # verify
    v = sub.add_parser("verify", help="Run a verifier across clips/prompts")
    v.add_argument("--config", required=True)
    v.add_argument("--verifier", nargs="+", default=None,
                   help="gemini / music_flamingo (one or more)")
    v.add_argument("--target", choices=["clips", "prompts"], default="clips")
    v.add_argument("--clips", default=None,
                   help="For --target clips: path to clips.jsonl")
    v.add_argument("--prompts", default=None)
    v.add_argument("--manifest", default=None,
                   help="For --target prompts: path to generation_manifest.json")
    v.add_argument("--out", default=None)
    v.add_argument("--max-facts-per-clip", type=int, default=None)
    v.set_defaults(func=_cmd_verify)

    # score
    sc = sub.add_parser("score", help="Score a verification results JSON")
    sc.add_argument("--results", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--bootstrap-n", type=int, default=1000)
    sc.add_argument("--alpha", type=float, default=0.05)
    sc.add_argument("--seed", type=int, default=848)
    sc.set_defaults(func=_cmd_score)

    # analyze
    an = sub.add_parser("analyze", help="Cross-verifier analysis + figures")
    an.add_argument("--scores", nargs="+", required=True,
                    help='Entries of the form label=path/to/scores.json')
    an.add_argument("--out", required=True)
    an.set_defaults(func=_cmd_analyze)

    # muchomusic
    mm = sub.add_parser("muchomusic", help="Run MuChoMusic MCQ calibration")
    mm.add_argument("--config", required=True)
    mm.add_argument("--verifier", nargs="+", default=None)
    mm.add_argument("--out", required=True)
    mm.add_argument("--split", default="test")
    mm.add_argument("--max-items", type=int, default=None)
    mm.set_defaults(func=_cmd_muchomusic)

    # smoke
    sm = sub.add_parser("smoke", help="End-to-end fixture smoke test (no models)")
    sm.add_argument("--fixtures", default=None)
    sm.set_defaults(func=_cmd_smoke)

    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    return args.func(args)


if __name__ == "__main__":
    main()
