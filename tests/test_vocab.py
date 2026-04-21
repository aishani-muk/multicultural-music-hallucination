from mhb.data.vocab import load_vocab, normalize, list_vocabs


def test_list_vocabs_nonempty():
    names = list_vocabs()
    assert "ragas_hindustani" in names
    assert "talas_hindustani" in names
    assert "instruments" in names


def test_normalize_case_insensitive_and_aliases():
    v = load_vocab("ragas_hindustani")
    assert normalize("Yaman", v) == "yaman"
    assert normalize("YAMAN", v) == "yaman"
    # alias
    assert normalize("kalyan", v) == "yaman"


def test_unknown_returns_none():
    v = load_vocab("ragas_hindustani")
    assert normalize("nonsenseraga", v) is None


def test_talas_beats_consistent():
    v = load_vocab("talas_hindustani")
    assert v["teental"]["beats"] == 16
    assert v["rupak"]["beats"] == 7


def test_makam_has_hicaz():
    v = load_vocab("makam")
    assert normalize("hijaz", v) == "hicaz"


def test_instruments_out_of_culture():
    v = load_vocab("instruments")
    # drum_kit should be out_of_culture_for the four art-music traditions.
    ooc = v["drum_kit"].get("out_of_culture_for", [])
    for trad in ("carnatic", "hindustani", "turkish_makam", "arab_andalusian"):
        assert trad in ooc
