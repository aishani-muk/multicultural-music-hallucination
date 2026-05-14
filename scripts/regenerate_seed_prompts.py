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
    # ═══════════════════════════════════════════════════════════════════
    #  CARNATIC (20 prompts)
    # ═══════════════════════════════════════════════════════════════════
    ("carnatic_001", "carnatic", "shankarabharanam", "adi", "kriti",
     ["voice", "mridangam", "tanpura"],
     "A Carnatic kriti in raga Shankarabharanam with mridangam accompaniment in Adi tala"),
    ("carnatic_002", "carnatic", "kalyani", None, "alapana",
     ["violin_carnatic", "tanpura"],
     "A Carnatic violin alapana in raga Kalyani with tambura drone"),
    ("carnatic_003", "carnatic", "kharaharapriya", "adi", "kriti",
     ["voice", "mridangam", "violin_carnatic", "tanpura"],
     "A Carnatic vocal kriti in raga Kharaharapriya with violin and mridangam in Adi tala"),
    ("carnatic_004", "carnatic", "todi", "rupaka", "varnam",
     ["voice", "mridangam", "tanpura"],
     "A Carnatic varnam in raga Todi with mridangam in Rupaka tala"),
    ("carnatic_005", "carnatic", "natabhairavi", "mishra_chapu", "kriti",
     ["voice", "mridangam", "tanpura"],
     "A Carnatic kriti in raga Natabhairavi with mridangam in Mishra Chapu tala"),
    ("carnatic_006", "carnatic", "hamsadhwani", "adi", "kriti",
     ["flute_carnatic", "mridangam", "tanpura"],
     "A Carnatic flute kriti in raga Hamsadhwani with mridangam in Adi tala"),
    ("carnatic_007", "carnatic", "mohanam", None, "alapana",
     ["veena", "tanpura"],
     "A Carnatic veena alapana in raga Mohanam with tambura drone"),
    ("carnatic_008", "carnatic", "bhairavi_carnatic", "adi", "kriti",
     ["voice", "violin_carnatic", "mridangam", "tanpura"],
     "A Carnatic vocal kriti in raga Bhairavi with violin and mridangam in Adi tala"),
    ("carnatic_009", "carnatic", "kambhoji", "khanda_chapu", "kriti",
     ["voice", "mridangam", "tanpura"],
     "A Carnatic kriti in raga Kambhoji with mridangam in Khanda Chapu tala"),
    ("carnatic_010", "carnatic", "abhogi", "adi", "kriti",
     ["voice", "mridangam", "ghatam", "tanpura"],
     "A Carnatic kriti in raga Abhogi with mridangam and ghatam in Adi tala"),
    ("carnatic_011", "carnatic", "shankarabharanam", "triputa", "kriti",
     ["veena", "mridangam", "tanpura"],
     "A Carnatic veena kriti in raga Shankarabharanam with mridangam in Triputa tala"),
    ("carnatic_012", "carnatic", "kalyani", "adi", "varnam",
     ["voice", "violin_carnatic", "mridangam", "tanpura"],
     "A Carnatic vocal varnam in raga Kalyani with violin and mridangam in Adi tala"),
    ("carnatic_013", "carnatic", "todi", None, "alapana",
     ["voice", "tanpura"],
     "A Carnatic vocal alapana in raga Todi with tambura drone"),
    ("carnatic_014", "carnatic", "hamsadhwani", "rupaka", "kriti",
     ["voice", "mridangam", "tanpura"],
     "A Carnatic vocal kriti in raga Hamsadhwani with mridangam in Rupaka tala"),
    ("carnatic_015", "carnatic", "kharaharapriya", None, "alapana",
     ["violin_carnatic", "tanpura"],
     "A Carnatic violin alapana in raga Kharaharapriya with tambura drone"),
    ("carnatic_016", "carnatic", "mohanam", "adi", "kriti",
     ["flute_carnatic", "mridangam", "kanjira", "tanpura"],
     "A Carnatic flute kriti in raga Mohanam with mridangam and kanjira in Adi tala"),
    ("carnatic_017", "carnatic", "natabhairavi", "adi", "kriti",
     ["voice", "violin_carnatic", "mridangam", "ghatam", "tanpura"],
     "A Carnatic vocal kriti in raga Natabhairavi with full ensemble in Adi tala"),
    ("carnatic_018", "carnatic", "shankarabharanam", "ata", "kriti",
     ["voice", "mridangam", "tanpura"],
     "A Carnatic vocal kriti in raga Shankarabharanam with mridangam in Ata tala"),
    ("carnatic_019", "carnatic", "kalyani", "mishra_chapu", "kriti",
     ["veena", "mridangam", "tanpura"],
     "A Carnatic veena kriti in raga Kalyani with mridangam in Mishra Chapu tala"),
    ("carnatic_020", "carnatic", "kambhoji", None, "alapana",
     ["voice", "tanpura"],
     "A Carnatic vocal alapana in raga Kambhoji with tambura drone"),

    # ═══════════════════════════════════════════════════════════════════
    #  HINDUSTANI (20 prompts)
    # ═══════════════════════════════════════════════════════════════════
    ("hindustani_001", "hindustani", "yaman", "teental", "khayal",
     ["sitar", "tabla", "tanpura"],
     "A Hindustani sitar performance in raga Yaman with tabla in teental"),
    ("hindustani_002", "hindustani", "bhairavi", "rupak", "thumri",
     ["voice", "harmonium", "tabla", "tanpura"],
     "A Hindustani vocal thumri in raga Bhairavi with harmonium and tabla in rupak tala"),
    ("hindustani_003", "hindustani", "darbari", "jhaptaal", "khayal",
     ["voice", "tabla", "tanpura"],
     "A Hindustani vocal khayal in raga Darbari Kanada with tabla in jhaptaal"),
    ("hindustani_004", "hindustani", "malkauns", "teental", "khayal",
     ["voice", "tabla", "tanpura"],
     "A Hindustani vocal khayal in raga Malkauns with tabla in teental"),
    ("hindustani_005", "hindustani", "bageshri", "teental", "khayal",
     ["voice", "harmonium", "tabla", "tanpura"],
     "A Hindustani vocal khayal in raga Bageshri with harmonium and tabla in teental"),
    ("hindustani_006", "hindustani", "yaman", None, "alap",
     ["sitar", "tanpura"],
     "A Hindustani sitar alap in raga Yaman with tanpura drone, no percussion"),
    ("hindustani_007", "hindustani", "desh", "keherwa", "thumri",
     ["voice", "harmonium", "tabla", "tanpura"],
     "A Hindustani vocal thumri in raga Desh with harmonium and tabla in keherwa"),
    ("hindustani_008", "hindustani", "todi", "jhaptaal", "khayal",
     ["voice", "tabla", "tanpura"],
     "A Hindustani vocal khayal in raga Todi with tabla in jhaptaal"),
    ("hindustani_009", "hindustani", "marwa", "teental", "khayal",
     ["voice", "tabla", "tanpura"],
     "A Hindustani vocal khayal in raga Marwa with tabla in teental"),
    ("hindustani_010", "hindustani", "puriya_dhanashri", "ektaal", "khayal",
     ["voice", "tabla", "tanpura"],
     "A Hindustani vocal khayal in raga Puriya Dhanashri with tabla in ektaal"),
    ("hindustani_011", "hindustani", "bhairav", "teental", "khayal",
     ["voice", "tabla", "tanpura"],
     "A Hindustani morning raga Bhairav khayal with tabla in teental"),
    ("hindustani_012", "hindustani", "yaman", "teental", "khayal",
     ["sarod", "tabla", "tanpura"],
     "A Hindustani sarod performance in raga Yaman with tabla in teental"),
    ("hindustani_013", "hindustani", "darbari", None, "alap",
     ["sarangi", "tanpura"],
     "A Hindustani sarangi alap in raga Darbari Kanada with tanpura drone"),
    ("hindustani_014", "hindustani", "bhairavi", "dadra", "thumri",
     ["voice", "tabla", "tanpura"],
     "A Hindustani vocal thumri in raga Bhairavi with tabla in dadra tala"),
    ("hindustani_015", "hindustani", "malkauns", None, "alap",
     ["bansuri", "tanpura"],
     "A Hindustani bansuri alap in raga Malkauns with tanpura drone"),
    ("hindustani_016", "hindustani", "desh", "teental", "khayal",
     ["voice", "tabla", "tanpura"],
     "A Hindustani vocal khayal in raga Desh with tabla in teental"),
    ("hindustani_017", "hindustani", "bageshri", None, "alap",
     ["sitar", "tanpura"],
     "A Hindustani sitar alap in raga Bageshri with tanpura drone"),
    ("hindustani_018", "hindustani", "bhairav", "chautal", "dhrupad",
     ["voice", "tabla", "tanpura"],
     "A Hindustani dhrupad in raga Bhairav with tabla in chautal"),
    ("hindustani_019", "hindustani", "todi", "teental", "khayal",
     ["sitar", "tabla", "tanpura"],
     "A Hindustani sitar khayal in raga Todi with tabla in teental"),
    ("hindustani_020", "hindustani", "marwa", "dhamar", "khayal",
     ["voice", "tabla", "tanpura"],
     "A Hindustani vocal khayal in raga Marwa with tabla in dhamar tala"),

    # ═══════════════════════════════════════════════════════════════════
    #  TURKISH MAKAM (20 prompts)
    # ═══════════════════════════════════════════════════════════════════
    ("turkish_001", "turkish_makam", "hicaz", None, "taksim",
     ["ney"],
     "An Ottoman Turkish taksim in makam Hicaz on ney"),
    ("turkish_002", "turkish_makam", "nihavend", "aksak", "saz_semai",
     ["kanun", "oud", "kudum"],
     "A Turkish ensemble in makam Nihavend with kanun, oud, and kudum in aksak (9/8)"),
    ("turkish_003", "turkish_makam", "rast", "sofyan", "sarki",
     ["voice", "oud", "bendir"],
     "A Turkish vocal sarki in makam Rast with oud and bendir in sofyan"),
    ("turkish_004", "turkish_makam", "ussak", None, "taksim",
     ["oud"],
     "An Ottoman Turkish oud taksim in makam Ussak"),
    ("turkish_005", "turkish_makam", "kurdi", "duyek", "sarki",
     ["voice", "kanun", "kudum"],
     "A Turkish vocal sarki in makam Kurdi with kanun in duyek"),
    ("turkish_006", "turkish_makam", "huseyni", "aksak", "sarki",
     ["voice", "ney", "oud", "kudum"],
     "A Turkish vocal sarki in makam Huseyni with ney and oud in aksak"),
    ("turkish_007", "turkish_makam", "saba", None, "taksim",
     ["tanbur"],
     "A Turkish tanbur taksim in makam Saba"),
    ("turkish_008", "turkish_makam", "segah", "semai", "sarki",
     ["voice", "oud", "kudum"],
     "A Turkish vocal sarki in makam Segah with oud in semai"),
    ("turkish_009", "turkish_makam", "hicaz", "duyek", "sarki",
     ["voice", "kanun", "ney", "kudum"],
     "A Turkish vocal sarki in makam Hicaz with kanun and ney in duyek"),
    ("turkish_010", "turkish_makam", "suzinak", "aksak", "sarki",
     ["voice", "oud", "kudum"],
     "A Turkish vocal sarki in makam Suzinak with oud in aksak"),
    ("turkish_011", "turkish_makam", "rast", None, "taksim",
     ["kanun"],
     "A Turkish kanun taksim in makam Rast"),
    ("turkish_012", "turkish_makam", "nihavend", "sofyan", "sarki",
     ["voice", "ney", "kudum"],
     "A Turkish vocal sarki in makam Nihavend with ney in sofyan"),
    ("turkish_013", "turkish_makam", "acemasiran", "turkaksagi", "sarki",
     ["voice", "tanbur", "kudum"],
     "A Turkish vocal sarki in makam Acemasiran with tanbur in turk aksagi"),
    ("turkish_014", "turkish_makam", "ussak", "duyek", "sarki",
     ["voice", "oud", "bendir"],
     "A Turkish vocal sarki in makam Ussak with oud and bendir in duyek"),
    ("turkish_015", "turkish_makam", "huseyni", None, "taksim",
     ["kemence"],
     "A Turkish kemence taksim in makam Huseyni"),
    ("turkish_016", "turkish_makam", "saba", "aksak", "sarki",
     ["voice", "kanun", "oud", "darbuka"],
     "A Turkish vocal sarki in makam Saba with kanun and oud in aksak"),
    ("turkish_017", "turkish_makam", "hicaz", "semai", "saz_semai",
     ["ney", "tanbur", "kudum"],
     "A Turkish saz semai in makam Hicaz with ney and tanbur in semai"),
    ("turkish_018", "turkish_makam", "kurdi", None, "taksim",
     ["ney"],
     "A Turkish ney taksim in makam Kurdi"),
    ("turkish_019", "turkish_makam", "segah", "devr_i_hindi", "sarki",
     ["voice", "oud", "kanun", "kudum"],
     "A Turkish vocal sarki in makam Segah with oud and kanun in devr-i Hindi"),
    ("turkish_020", "turkish_makam", "rast", "evfer", "sarki",
     ["voice", "tanbur", "ney", "bendir"],
     "A Turkish vocal sarki in makam Rast with tanbur and ney in evfer"),

    # ═══════════════════════════════════════════════════════════════════
    #  ARAB-ANDALUSIAN (15 prompts)
    # ═══════════════════════════════════════════════════════════════════
    ("arab_001", "arab_andalusian", "rasd_al_dhil", "btayhi", "nuba",
     ["oud", "rabab", "darbuka"],
     "A Moroccan Andalusian nuba in Rasd al-Dhil with oud and rabab"),
    ("arab_002", "arab_andalusian", "isbihan", "basit", "nuba",
     ["oud", "rabab", "bendir"],
     "A Moroccan Andalusian nuba in tab Isbihan with oud and rabab in basit"),
    ("arab_003", "arab_andalusian", "hijaz_mashriqi", "qaim_wa_nisf", "nuba",
     ["voice", "oud", "darbuka"],
     "A Moroccan Andalusian vocal nuba in Hijaz Mashriqi in qaim wa nisf"),
    ("arab_004", "arab_andalusian", "hijaz_al_kabir", "darj", "nuba",
     ["oud", "rabab", "tar_percussion"],
     "A Moroccan Andalusian nuba in Hijaz al-Kabir with oud and rabab in darj"),
    ("arab_005", "arab_andalusian", "rasd", "btayhi", "nuba",
     ["voice", "oud", "rabab", "bendir"],
     "A Moroccan Andalusian vocal nuba in tab Rasd with oud and rabab in btayhi"),
    ("arab_006", "arab_andalusian", "iraq_al_ajam", "quddam", "nuba",
     ["oud", "rabab", "darbuka"],
     "A Moroccan Andalusian nuba in Iraq al-Ajam with oud and rabab in quddam"),
    ("arab_007", "arab_andalusian", "gharibat_al_husayn", "basit", "nuba",
     ["voice", "oud", "bendir"],
     "A Moroccan Andalusian vocal nuba in Gharibat al-Husayn in basit"),
    ("arab_008", "arab_andalusian", "rasd_al_dhil", "darj", "nuba",
     ["oud", "ney", "darbuka"],
     "A Moroccan Andalusian nuba in Rasd al-Dhil with oud and ney in darj"),
    ("arab_009", "arab_andalusian", "isbihan", "qaim_wa_nisf", "nuba",
     ["voice", "rabab", "tar_percussion"],
     "A Moroccan Andalusian vocal nuba in Isbihan in qaim wa nisf"),
    ("arab_010", "arab_andalusian", "hijaz_mashriqi", "btayhi", "nuba",
     ["oud", "rabab", "bendir"],
     "A Moroccan Andalusian nuba in Hijaz Mashriqi with oud and rabab in btayhi"),
    ("arab_011", "arab_andalusian", "rasd", "darj", "nuba",
     ["oud", "darbuka"],
     "A Moroccan Andalusian nuba in tab Rasd with oud in darj"),
    ("arab_012", "arab_andalusian", "hijaz_al_kabir", "quddam", "nuba",
     ["voice", "oud", "rabab", "darbuka"],
     "A Moroccan Andalusian vocal nuba in Hijaz al-Kabir in quddam"),
    ("arab_013", "arab_andalusian", "iraq_al_ajam", "btayhi", "nuba",
     ["oud", "rabab", "bendir"],
     "A Moroccan Andalusian nuba in Iraq al-Ajam with oud and rabab in btayhi"),
    ("arab_014", "arab_andalusian", "gharibat_al_husayn", "darj", "nuba",
     ["voice", "oud", "darbuka"],
     "A Moroccan Andalusian vocal nuba in Gharibat al-Husayn in darj"),
    ("arab_015", "arab_andalusian", "rasd_al_dhil", "basit", "nuba",
     ["voice", "oud", "rabab", "tar_percussion"],
     "A Moroccan Andalusian vocal nuba in Rasd al-Dhil with oud and rabab in basit"),

    # ═══════════════════════════════════════════════════════════════════
    #  REGGAETON (15 prompts)
    # ═══════════════════════════════════════════════════════════════════
    ("reggaeton_001", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass"],
     "A reggaeton track at ~95 BPM with a dembow beat and Spanish vocals"),
    ("reggaeton_002", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass", "voice"],
     "An energetic reggaeton track with dembow beat, heavy synth bass and male Spanish vocals"),
    ("reggaeton_003", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "synth_bass", "synthesizer"],
     "A moody reggaeton track with dembow rhythm and atmospheric synth pads at ~95 BPM"),
    ("reggaeton_004", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass", "voice"],
     "A reggaeton track with female Spanish vocals over a dembow beat at ~100 BPM"),
    ("reggaeton_005", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass"],
     "A slow reggaeton perreo track at ~90 BPM with heavy dembow kick and synth bass"),
    ("reggaeton_006", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass", "synthesizer", "voice"],
     "An uptempo reggaeton banger at ~105 BPM with synth leads and Spanish vocals"),
    ("reggaeton_007", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "synth_bass"],
     "A minimal reggaeton instrumental with dembow kick and deep sub-bass at ~95 BPM"),
    ("reggaeton_008", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass", "voice"],
     "A romantic reggaeton track with melodic Spanish vocals and dembow beat at ~92 BPM"),
    ("reggaeton_009", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass", "synthesizer"],
     "A reggaeton track with trap-influenced hi-hats and dembow rhythm at ~98 BPM"),
    ("reggaeton_010", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass", "voice"],
     "A classic reggaeton track with call-and-response Spanish vocals and dembow beat"),
    ("reggaeton_011", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "synth_bass", "voice"],
     "A dancehall-influenced reggaeton track with Spanish patois vocals and dembow bass"),
    ("reggaeton_012", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass", "synthesizer"],
     "A futuristic reggaeton track with vocoder effects and synth arpeggios over dembow"),
    ("reggaeton_013", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass", "voice"],
     "A party reggaeton anthem with chanted Spanish vocals and punchy dembow at ~95 BPM"),
    ("reggaeton_014", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "synth_bass", "synthesizer", "voice"],
     "A Latin pop-reggaeton crossover with melodic vocals and soft dembow at ~94 BPM"),
    ("reggaeton_015", "reggaeton", None, "dembow", "reggaeton_song",
     ["dembow_kick", "clap", "synth_bass"],
     "An aggressive reggaeton instrumental with distorted dembow kicks and heavy bass"),

    # ═══════════════════════════════════════════════════════════════════
    #  WESTERN CONTROL (10 prompts)
    # ═══════════════════════════════════════════════════════════════════
    ("western_ctrl_001", "western_control", None, None, "sonata",
     ["piano"],
     "A solo piano sonata in C major in the style of Mozart"),
    ("western_ctrl_002", "western_control", None, None, "concerto",
     ["piano"],
     "A piano concerto movement in D minor in a Romantic style"),
    ("western_ctrl_003", "western_control", None, None, "symphony",
     ["piano"],
     "An orchestral symphony excerpt in G major with prominent piano"),
    ("western_ctrl_004", "western_control", None, None, "waltz",
     ["piano"],
     "A Viennese waltz in 3/4 time played on solo piano"),
    ("western_ctrl_005", "western_control", None, None, "nocturne",
     ["piano"],
     "A Chopin-style nocturne in E-flat major on solo piano"),
    ("western_ctrl_006", "western_control", None, None, "fugue",
     ["piano"],
     "A Bach-style keyboard fugue in C minor on piano"),
    ("western_ctrl_007", "western_control", None, None, "etude",
     ["piano"],
     "A fast virtuosic piano etude in A minor"),
    ("western_ctrl_008", "western_control", None, None, "prelude",
     ["piano"],
     "A Debussy-style impressionist piano prelude"),
    ("western_ctrl_009", "western_control", None, None, "march",
     ["piano"],
     "A military march in B-flat major on piano"),
    ("western_ctrl_010", "western_control", None, None, "lullaby",
     ["piano"],
     "A gentle piano lullaby in F major, slow tempo"),
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
