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


def _token(**kw: object) -> asllrp.SignToken:
    base: dict[str, object] = {
        "video_id": "1",
        "gloss": "A",
        "start_frame": 100,
        "end_frame": 110,
        "utterance_start": 100,
        "utterance_end": 200,
        "utterance_video": "7.mp4",
        "signer": "Cory",
        "source_collection": "Cory_x",
        "sign_type": "Signs",
    }
    base.update(kw)
    return asllrp.SignToken(**base)  # type: ignore[arg-type]


def _local_clips() -> tuple[dict[str, int], dict[str, float]]:
    import glob
    import json

    from seam.paths import default_data_root

    lm = default_data_root() / "emosign" / "landmarks"
    counts: dict[str, int] = {}
    rates: dict[str, float] = {}
    for p in glob.glob(str(lm / "*.json")):
        with open(p) as fh:
            meta = json.loads(fh.read())
        if meta.get("native_fps"):
            counts[str(meta["utterance_id"])] = int(meta["frame_count"])
            rates[str(meta["utterance_id"])] = float(meta["native_fps"])
    return counts, rates


def test_token_indices_are_on_a_30fps_timeline_whatever_the_clip_rate() -> None:
    """The third conclusion this test has held, and the first checked the right way.

    It first asserted the labels were unusable (absolute indices against clip length),
    then that ``crop index = session frame - utterance_start + 1`` held for ~90% of
    tokens. In range is not aligned. The utterance span over the extracted frame count
    is 1.00 for the 30 fps clips and 1.25 for the 24 fps ones, so the indices are on a
    30 fps timeline and the frame-for-frame mapping runs 25% fast on 138 of 200 clips.
    The "10% overshoot" that used to be asserted here as a real and expected case was
    the last fifth of those clips falling off the end.
    """
    pytest.importorskip("pandas")
    counts, rates = _local_clips()
    if not counts:
        pytest.skip("no EmoSign landmarks on disk")
    tokens, _ = asllrp.load()
    rep = asllrp.check_alignment(tokens, counts, rates)

    ratio = rep.span_ratio_median_by_fps
    assert set(ratio) == {"24", "30"}, f"expected two clip frame rates, got {ratio}"
    assert ratio["30"] == pytest.approx(1.00, abs=0.02)
    assert ratio["24"] == pytest.approx(30 / 24, abs=0.02)

    assert rep.aligned
    assert rep.n_tokens_aligned / rep.n_tokens >= 0.99, rep.as_dict()
    # The size of the correction: a tenth of all tokens were out of range before, and
    # being in range was never evidence of landing on the right frames.
    assert rep.n_tokens_aligned_one_to_one < 0.92 * rep.n_tokens_aligned


def test_crop_frame_index_rescales_by_the_clip_frame_rate() -> None:
    tok = _token()
    # A 30 fps clip is frame-for-frame.
    assert asllrp.crop_frame_index(tok, 100, clip_fps=30.0) == 1
    assert asllrp.crop_frame_index(tok, 110, clip_fps=30.0) == 11
    # A 24 fps clip of the same footage has four frames for every five.
    assert asllrp.crop_frame_index(tok, 100, clip_fps=24.0) == 1
    assert asllrp.crop_frame_index(tok, 110, clip_fps=24.0) == 9
    assert asllrp.crop_frame_index(tok, 200, clip_fps=24.0) == 81
    assert asllrp.crop_frame_range(tok, clip_fps=24.0) == (1, 9)
    # Before the utterance: not clamped - clamping mislabels.
    assert asllrp.crop_frame_index(tok, 98, clip_fps=30.0) is None


def test_the_clip_frame_rate_has_no_default() -> None:
    """A default of 30 fps is the bug: it is silently right for 62 clips of 200."""
    tok = _token()
    with pytest.raises(TypeError):
        asllrp.crop_frame_index(tok, 100)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        asllrp.crop_frame_range(tok)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="positive"):
        asllrp.crop_frame_index(tok, 100, clip_fps=0.0)


def test_alignment_passes_when_indices_do_index_the_clips() -> None:
    """The check must be able to say yes, or it is not a check."""
    fake = [_token(start_frame=0, end_frame=5, utterance_start=0, utterance_end=20)]
    fake[0].utterance_video = "42.mp4"
    uid = fake[0].utterance_id
    rep = asllrp.check_alignment(fake, {uid: 21}, {uid: 30.0})
    assert rep.aligned
    assert rep.n_tokens_aligned == 1
    assert rep.n_tokens_overshoot == 0


def test_alignment_can_say_no() -> None:
    """...and to say no. A token past the end of a 24 fps clip is out of range."""
    fake = [_token(start_frame=190, end_frame=200)]
    uid = fake[0].utterance_id
    # 101 session frames at 24 fps is 81 clip frames; give it a clip of 60.
    rep = asllrp.check_alignment(fake, {uid: 60}, {uid: 24.0})
    assert not rep.aligned
    assert rep.n_tokens_overshoot == 1
