"""The ASLLRP token table is not valid CSV, and a silent skip would be invisible.

Two specific things have to hold, and both are ways a plausible-looking number goes
wrong without anyone noticing:

* **Every row parses.** The file contains bare inner double quotes inside quoted gloss
  labels, so a standard reader fails on ~2 rows in 6,000. Skipping them would drop the
  most unusual glosses - exactly the ones a recognition model finds hardest - and the
  token count would look fine.
* **Every token gets a signer.** Signer-disjoint splits are the whole reason this file
  is usable, and one signer here appears only in a `1-Name-topic` form. A name set
  derived from the other form silently leaves all 4,389 of their tokens unassigned,
  pooling them into one anonymous LOSO group.
"""

from __future__ import annotations

import pytest

from seam.data import asllrp


def test_split_row_tolerates_a_bare_inner_quote() -> None:
    # The real line that breaks a standard CSV reader.
    line = (
        '"345140","5"ok, hey"","5"ok, hey"","5"ok hey"","1997","2003","1867","2218",'
        '"crvd-5","crvd-5","crvd-5","crvd-5","sign_521019.mp4","5597513.mp4",'
        '"Rachel_2011-12-08_sc45","7","2011-12-08_0045-cam1-for-ss3.mp4","Gestures",'
        '"5"ok, hey"","F"'
    )
    row = asllrp._split_row(line)
    assert len(row) == 20, "must recover the header's 20 fields"
    assert row[1] == '5"ok, hey"'
    assert row[13] == "5597513.mp4"
    assert row[19] == "F"


def test_split_row_is_a_plain_csv_reader_on_ordinary_lines() -> None:
    line = '"a","b,c","d"'
    assert asllrp._split_row(line) == ["a", "b,c", "d"]


def test_signer_from_both_collection_forms() -> None:
    assert asllrp._signer_from_underscore("Cory_2013-6-27_sc115") == "Cory"
    assert asllrp._signer_from_dashed("8-Ben-Control-of-sound-places") == "Ben"
    assert asllrp._signer_from_dashed("4-Ben-Deaf-Speech") == "Ben"
    # Documented limitation: in isolation the dashed rule returns whatever follows the
    # index, because a topic word and a name are indistinguishable in one string. The
    # guard is on the derived set, asserted in `test_derived_signer_set_is_only_people`.
    assert asllrp._signer_from_dashed("1-Introduction-x") == "Introduction"


def test_derived_signer_set_is_only_people() -> None:
    """The guard on the dashed-form heuristic.

    `_signer_from_dashed` reads the token after the collection index, so a topic word
    would be indistinguishable from a name per row. The check that actually protects the
    LOSO splits is on the derived set: if a topic word leaked in it would appear as an
    extra signer, splitting one person into two groups.
    """
    pytest.importorskip("pandas")
    _tokens, report = asllrp.load()
    assert report["signer_set_plausible"]
    assert report["signers"] == ["Ben", "Cory", "Jonathan", "Rachel"]


def test_every_token_gets_a_signer() -> None:
    pytest.importorskip("pandas")
    tokens, report = asllrp.load()
    assert tokens, "the token table should not be empty"
    assert report["unmapped_source_collections"] == [], (
        "an unmapped collection would pool tokens into an anonymous LOSO group"
    )
    assert report["unmapped_collection_rows"] == 0
    # The dashed-only signer is the case that broke the first implementation.
    assert "Ben" in report["signers"]
    assert report["n_signers"] == 4
    assert all(t.signer for t in tokens)


def test_load_reports_malformed_rows_rather_than_dropping_them() -> None:
    pytest.importorskip("pandas")
    _tokens, report = asllrp.load()
    assert "malformed_rows" in report
    assert "parse_note" in report
    # Every one of the 17,522 annotated tokens must survive the parse.
    assert report["n_tokens"] == 17_522
    assert report["malformed_rows"] == 0


def test_utterance_id_is_the_join_key_for_syntactic_context() -> None:
    """Recognised glosses need their linguistic context, which joins on utterance id."""
    pytest.importorskip("pandas")
    tokens, _ = asllrp.load()
    with_utt = [t for t in tokens if t.utterance_id]
    assert len(with_utt) == len(tokens), "every token should carry a numeric utterance id"
    assert all(t.utterance_id.isdigit() for t in with_utt)


def test_vocab_is_frequency_ordered() -> None:
    pytest.importorskip("pandas")
    tokens, _ = asllrp.load()
    v = asllrp.GlossVocab.build(tokens)
    assert len(v) == 1956
    assert v.itos[0] == "IX", (
        "most frequent gloss first, so a collapsed CTC model is visibly biased"
    )
    assert v.encode(v.itos[0]) == 0
    assert v.decode(0) == "IX"
    assert v.encode("NOT_A_GLOSS") == -1


def test_alignment_with_emosign_clips_is_reported_as_broken() -> None:
    """The token frame indices do NOT index the EmoSign excerpts.

    Measured: 1,725 of 1,738 overlapping tokens fall entirely outside the extracted
    video, because the ASLLRP indices are absolute positions in a long session
    recording (median utterance span 5,123 frames, ~2.8 min) while the clips are
    4.6-second excerpts (median 109 frames). The ratio is not a constant rescale, so it
    cannot be recovered arithmetically.

    This test exists so that a future recogniser cannot quietly train on misaligned
    labels and report a CER that looks like a model result.
    """
    import glob
    import json

    pytest.importorskip("pandas")
    from seam.paths import default_data_root

    lm = default_data_root() / "emosign" / "landmarks"
    counts: dict[str, int] = {}
    for p in glob.glob(str(lm / "*.json")):
        with open(p) as fh:
            meta = json.loads(fh.read())
        counts[str(meta["utterance_id"])] = int(meta["frame_count"])
    if not counts:
        pytest.skip("no EmoSign landmarks on disk")

    tokens, _ = asllrp.load()
    rep = asllrp.check_alignment(tokens, counts)
    d = rep.as_dict()
    assert not rep.aligned, (
        "if this now passes, the indices DO align and M5a can use these labels - "
        "update the AlignmentReport docstring and the plan"
    )
    assert d["outside_fraction"] > 0.9, f"expected near-total misalignment, got {d}"
    # Not recoverable by rescaling: the ratio varies by two orders of magnitude.
    assert rep.ratio_cv > 0.5, f"ratio looks constant ({rep.ratio_cv}); rescaling might work"
    assert "Sign video filename" in asllrp.AlignmentReport.__doc__


def test_alignment_passes_when_indices_do_index_the_clips() -> None:
    """The check must be able to say yes, or it is not a check."""
    fake = [
        asllrp.SignToken(
            video_id="1",
            gloss="A",
            start_frame=0,
            end_frame=5,
            utterance_start=0,
            utterance_end=20,
            utterance_video="42.mp4",
            signer="Cory",
            source_collection="Cory_x",
            sign_type="Signs",
        )
    ]
    rep = asllrp.check_alignment(fake, {"42": 20})
    assert rep.aligned
    assert rep.n_tokens_outside == 0
