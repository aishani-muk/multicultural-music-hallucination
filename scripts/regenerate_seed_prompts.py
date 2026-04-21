"""Regenerate src/mhb/prompts/seed_prompts.json from the vocab + templates.

Produces one BenchmarkPrompt per (tradition, modal, cycle, form, instruments)
tuple hand-listed below. Facts are auto-derived from templates — no manual
fact authoring, so every claim traces back to vocab/templates.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mhb.data.facts import synthesize_facts   # noqa: E402
from mhb.data.vocab import load_vocab          # noqa: E402
from mhb.schema import (                        # noqa: E402
    BenchmarkPrompt, GroundTruth, InstrumentTag, MetricCycle, ModalLabel,
    save_prompts,
)


# (prompt_id, tradition, raga/makam/etc, cycle, form, instruments, natural-language prompt)
SEEDS = [
    ("carnatic_001", "carnatic", "shankarabharanam", "adi", "kriti",
     ["voice", "mridangam", "tanpura"],
     "A Carnatic kriti in raga Shankarabharanam with mridangam accompaniment in Adi tala"),
    ("carnatic_002", "carnatic", "kalyani", None, "alapana",
     ["violin_carnatic", "tanpura"],
     "A Carnatic violin alapana in raga Kalyani with tambura drone"),
    ("hindustani_001", "hindustani", "yaman", "teental", "khayal",
     ["sitar", "tabla", "tanpura"],
     "A Hindustani sitar performance in raga Yaman with tabla in teental"),
    ("hindustani_002", "hindustani", "bhairavi", "rupak", "thumri",
     ["voice", "harmonium", "tabla", "tanpura"],
     "A Hindustani vocal thumri in raga Bhairavi with harmonium and tabla in rupak tala"),
    ("turkish_001", "turkish_makam", "hicaz", None, "taksim",
     ["ney"],
     "An Ottoman Turkish taksim in makam Hicaz on ney"),
    ("turkish_002", "turkish_makam", "nihavend", "aksak", "saz_semai",
     ["kanun", "oud", "kudum"],
     "A Turkish ensemble in makam Nihavend with kanun, oud, and kudum in aksak (9/8)"),
    ("arab_001", "arab_andalusian", "rasd_al_dhil", "btayhi", "nuba",
     ["oud", "rabab", "darbuka"],
     "A Moroccan Andalusian nuba in Rasd al-Dhil with oud and rabab"),
    ("reggaeton_001", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass"],
     "A reggaeton track at ~95 BPM with a dembow beat and Spanish vocals"),
    ("western_ctrl_001", "western_control", None, None, "sonata",
     ["piano"],
     "A solo piano sonata in C major in the style of Mozart"),
]


def _modal(tradition: str, name: str | None) -> ModalLabel | None:
    if not name:
        return None
    vocab_by_trad = {
        "carnatic": "ragas_carnatic",
        "hindustani": "ragas_hindustani",
        "turkish_makam": "makam",
        "arab_andalusian": "tab3",
        "radif": "dastgah",
    }
    if tradition not in vocab_by_trad:
        return ModalLabel(name=name, system="western_key")
    vocab = load_vocab(vocab_by_trad[tradition])
    entry = vocab.get(name, {})
    system = {"carnatic": "raga_carnatic", "hindustani": "raga_hindustani",
              "turkish_makam": "makam", "arab_andalusian": "tab3"}[tradition]
    return ModalLabel(name=name, system=system, aliases=entry.get("aliases", []))


def _cycle(tradition: str, name: str | None) -> MetricCycle | None:
    if not name:
        return None
    vocab_by_trad = {
        "carnatic": "talas_carnatic",
        "hindustani": "talas_hindustani",
        "turkish_makam": "usul",
        "arab_andalusian": "mizan",
        "reggaeton": "reggaeton",
    }
    if tradition not in vocab_by_trad:
        return None
    vocab = load_vocab(vocab_by_trad[tradition])
    entry = vocab.get(name, {})
    system = {"carnatic": "tala_carnatic", "hindustani": "tala_hindustani",
              "turkish_makam": "usul", "arab_andalusian": "mizan",
              "reggaeton": "dembow"}[tradition]
    return MetricCycle(
        name=name, system=system,
        beats=entry.get("beats"),
        accent_pattern=entry.get("accent_pattern") or entry.get("division") or entry.get("angas"),
        aliases=entry.get("aliases", []),
    )


def _build_prompt(pid, tradition, modal, cycle, form, instruments, text):
    gt = GroundTruth(
        modal_system=_modal(tradition, modal),
        metric_cycle=_cycle(tradition, cycle),
        form=form,
        instruments=[InstrumentTag(canonical=i, source="metadata") for i in instruments],
        tempo_bpm=95.0 if tradition == "reggaeton" else None,
        language="es" if tradition == "reggaeton" else None,
    )
    facts = synthesize_facts(tradition, gt, max_negative_instruments=3)
    return BenchmarkPrompt(
        prompt_id=pid, tradition=tradition,
        generation_prompt=text, description=f"Auto-derived prompt for {tradition}/{modal}/{cycle}",
        atomic_facts=facts,
        reference_metadata={"modal": modal, "cycle": cycle, "form": form,
                            "instruments": instruments},
    )


def main():
    prompts = [_build_prompt(*s) for s in SEEDS]
    out = ROOT / "src" / "mhb" / "prompts" / "seed_prompts.json"
    save_prompts(prompts, out)
    total_facts = sum(len(p.atomic_facts) for p in prompts)
    print(f"Wrote {len(prompts)} prompts, {total_facts} auto-derived facts → {out}")


if __name__ == "__main__":
    main()
