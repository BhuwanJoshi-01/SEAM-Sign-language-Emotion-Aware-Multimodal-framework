"""Tests for the SignStream XML parser, written against the REAL export format.

**Rewritten after the data arrived.** The first version tested against the example in
ASLLRP Report 18 §8.3 and passed 22 tests while being wrong about the corpus: the real
export uses `SIGNSTREAM_DAI` > `COLLECTION` > `SEGMENT-TIER` > `UTTERANCE`, has no
`STATEMENT-FIELD`/`FIELD-ID`, and puts `PARTICIPANT` on the tier rather than the utterance.
A documented example is a fine thing to build against and a poor substitute for the file.

The decisive fact these tests exist to protect: **the marker lives in LABEL *and* VALUE.**
`eye brows` appears 4,957 times in the corpus with 11 different values; mapping on the
label alone cannot distinguish a brow raise from a brow furrow, which would flatten the
most important non-manual in the language while looking like it worked.

Tests that need the real corpus skip when it is absent, so the suite still runs on a
machine that has not downloaded the annotations.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seam.data.signstream import (
    MARKER_RULES,
    ParseReport,
    Utterance,
    gloss_vocabulary,
    map_event,
    parse_directory,
    parse_file,
)

REAL = Path("/mnt/DevProd/seam_data/asllrp_signstream_xml/raw")

#: A minimal document in the format the DAI actually exports. Field-for-field from the
#: first file in the download, including the parts that are easy to get wrong:
#: PARTICIPANT on the tier, and UTTERANCE-NUMBER as a child of UTTERANCE.
REAL_XML = """<?xml version='1.0' encoding='utf-8'?>
<SIGNSTREAM_DAI AUTHOR='BOSTON UNIVERSITY' DOWNLOAD_DATE='Sun Jul 05 20:03:06 EDT 2026'>
  <COLLECTIONS>
  <COLLECTION ID='1379840' NAME='Cory_2013-6-27_sc107'>
    <MEDIA-FILES>
      <MEDIA-FILE ID='1379956' FILE-NAME='2013-06-27_CB_0107-cam1-for-ss3.mp4'/>
    </MEDIA-FILES>
    <TEMPORAL-PARTITIONS>
      <TEMPORAL-PARTITION ID='1' NAME='Temporal Partition 1'>
        <SEGMENT-TIERS>
          <SEGMENT-TIER ID='1' NAME='Segment 1'>
            <PARTICIPANT>'Cory'</PARTICIPANT>
            <DOMINANT-HAND>'R'</DOMINANT-HAND>
            <UTTERANCES>
              <UTTERANCE ID='627' START_FRAME='36' END_FRAME='221'>
                <TRANSLATION>'The mouse escapes and the cat runs after it.'</TRANSLATION>
                <UTTERANCE-NUMBER>'U1'</UTTERANCE-NUMBER>
                <MANUALS>
                  <SIGN ID='269261'>
                    <LABEL>'MOUSE/FICTION'</LABEL>
                    <SIGN_TYPE>'Lexical Signs'</SIGN_TYPE>
                    <DOMINANT_HAND START_FRAME='46' END_FRAME='55'/>
                  </SIGN>
                  <SIGN ID='269262'>
                    <LABEL>'ESCAPE'</LABEL>
                    <SIGN_TYPE>'Lexical Signs'</SIGN_TYPE>
                    <NON_DOMINANT_HAND START_FRAME='60' END_FRAME='70'/>
                  </SIGN>
                </MANUALS>
                <NON_MANUALS>
                  <NON_MANUAL ID='267934' START_FRAME='89' END_FRAME='106'>
                    <LABEL>'eye brows'</LABEL>
                    <VALUE>'raised'</VALUE>
                  </NON_MANUAL>
                  <NON_MANUAL ID='267935' START_FRAME='98' END_FRAME='152'>
                    <LABEL>'head mvmt: shake'</LABEL>
                    <VALUE>'rapid'</VALUE>
                    <OFFSET START_FRAME='152' END_FRAME='167'/>
                  </NON_MANUAL>
                  <NON_MANUAL ID='267999' START_FRAME='160' END_FRAME='170'>
                    <LABEL>'shoulders: shrug'</LABEL>
                    <VALUE>'once'</VALUE>
                  </NON_MANUAL>
                </NON_MANUALS>
              </UTTERANCE>
            </UTTERANCES>
          </SEGMENT-TIER>
        </SEGMENT-TIERS>
      </TEMPORAL-PARTITION>
    </TEMPORAL-PARTITIONS>
  </COLLECTION>
  </COLLECTIONS>
</SIGNSTREAM_DAI>
"""


def _corpus() -> Path:
    if not REAL.is_dir() or not any(REAL.glob("*.xml")):
        pytest.skip(f"ASLLRP SignStream XML not present at {REAL}")
    return REAL


# --- the real format -------------------------------------------------------


def test_parses_the_real_export_shape(tmp_path: Path) -> None:
    p = tmp_path / "a.xml"
    p.write_text(REAL_XML)
    got, rep = parse_directory(tmp_path)
    assert len(got) == 1
    u = got[0]
    assert u.utterance_id == "627"
    assert u.participant == "Cory", "PARTICIPANT lives on SEGMENT-TIER, not UTTERANCE"
    assert u.collection == "Cory_2013-6-27_sc107"
    assert u.collection_id == "1379840"
    assert (u.start_frame, u.end_frame) == (36, 221)
    assert u.utterance_number == "U1"
    assert "The mouse escapes" in u.translation
    assert [g for g, _, _ in u.glosses] == ["MOUSE/FICTION", "ESCAPE"]
    assert u.glosses[0][1:] == (46, 55), "gloss frames come from DOMINANT_HAND"
    assert u.glosses[1][1:] == (60, 70), "or NON_DOMINANT_HAND"
    assert len(u.non_manuals) == 3
    rep_stats = rep.as_dict()
    assert rep_stats["utterances"] == 1 and rep_stats["signs"] == 2


def test_utterance_number_is_not_mistaken_for_an_utterance(tmp_path: Path) -> None:
    """`UTTERANCE-NUMBER` contains "UTTERANCE"; a substring match emits it as its own
    utterance and doubles the count. Caught before the data arrived, confirmed by it."""
    p = tmp_path / "b.xml"
    p.write_text(REAL_XML)
    got, rep = parse_directory(tmp_path)
    assert len(got) == 1
    assert rep.utterances == 1


def test_onset_and_offset_are_captured(tmp_path: Path) -> None:
    p = tmp_path / "c.xml"
    p.write_text(REAL_XML)
    got, _ = parse_directory(tmp_path)
    shake = got[0].non_manuals[1]
    assert shake.offset == (152, 167)
    assert got[0].non_manuals[0].offset is None


def test_single_quotes_are_stripped_everywhere(tmp_path: Path) -> None:
    """Every text value in this format is wrapped in single quotes."""
    p = tmp_path / "d.xml"
    p.write_text(REAL_XML)
    u = parse_directory(tmp_path)[0][0]
    assert u.participant == "Cory"
    assert u.utterance_number == "U1"
    assert u.non_manuals[0].label == "eye brows"
    assert u.glosses[0][0] == "MOUSE/FICTION"


# --- the decisive property: label AND value -------------------------------


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    [
        ("eye brows", "raised", "brow_raise"),
        ("eye brows", "slightly raised", "brow_raise"),
        ("eye brows", "further raised", "brow_raise"),
        ("eye brows", "lowered", "brow_furrow"),
        ("eye brows", "raised-furrowed", "brow_furrow"),
        ("eye brows", "further raised-furrowed", "brow_furrow"),
        ("eye brows", "left raised/right lowered", "brow_raise"),
        ("head mvmt: shake", "rapid", "head_shake"),
        ("head mvmt: shake", "slight rapid head shake", "head_shake"),
        ("head mvmt: nod", "single", "head_nod"),
        ("head pos: tilt fr/bk", "slightly back", "head_tilt"),
        ("head pos: tilt side", "left", "head_tilt_side"),
        ("eye aperture", "blink", "eye_aperture"),
        ("mouth", "open & tongue visible", "mouth_morpheme"),
        ("negative", "negation", "negation"),
        ("wh question", "whq", "question_wh"),
        ("yes-no question", "yes-no", "question_yn"),
        ("topic/focus", "topic", "topic"),
        ("body lean", "slightly forward", "body_lean"),
        ("shoulders", "shrug", None),
        ("eye brows", "", None),
    ],
)
def test_label_and_value_mapping(label: str, value: str, expected: str | None) -> None:
    got = map_event(label, value)
    if expected is None:
        assert expected not in got
    else:
        assert expected in got


def test_raised_furrowed_maps_to_both_and_not_only_to_raise() -> None:
    """A linguist marked it both; collapsing it to one discards their judgement."""
    got = map_event("eye brows", "raised-furrowed")
    assert "brow_furrow" in got
    assert "brow_raise" not in got, (
        "raised-furrowed is not a plain raise; the raise rule must not swallow it"
    )


def test_an_empty_value_maps_to_nothing() -> None:
    """A label with no value cannot be resolved to a marker, and must not default."""
    assert map_event("eye brows", "") == []


def test_brow_values_partition_rather_than_overlap() -> None:
    """Every brow value in the real corpus must resolve to exactly one direction."""
    raise_ = {"raised", "slightly raised", "further raised"}
    furrow = {"lowered", "slightly lowered", "further lowered", "raised-furrowed"}
    for v in raise_:
        assert map_event("eye brows", v) == ["brow_raise"], v
    for v in furrow:
        assert map_event("eye brows", v) == ["brow_furrow"], v


# --- corpus-level checks, skipped without the download --------------------


def test_the_real_corpus_parses_fully(tmp_path: Path) -> None:
    """The headline number, asserted rather than remembered."""
    root = _corpus()
    _utterances, rep = parse_directory(root)
    d = rep.as_dict()
    assert d["files"] >= 40, "the download covers all four signers plus RIT"
    assert d["utterances"] > 2000
    assert d["non_manual_events"] > 30000
    assert d["events_with_frame_alignment"] == d["non_manual_events"], (
        "every non-manual must carry frames; they are what joins to the landmarks"
    )
    assert d["mapped_fraction"] == 1.0, f"unmapped pairs remain: {d['unmapped_label_value_pairs']}"
    assert not d["unparseable_files"]
    assert "Cory" in d["utterances_by_participant"]
    assert "Rachel" in d["utterances_by_participant"]


def test_every_emosign_utterance_is_present(tmp_path: Path) -> None:
    """The join that unblocks M3: 200/200, by direct utterance ID."""
    import json

    from seam.paths import default_data_root

    lm = default_data_root() / "emosign" / "landmarks"
    if not lm.is_dir():
        pytest.skip("EmoSign landmarks not extracted")
    ours = {json.loads(p.read_text())["utterance_id"] for p in lm.glob("*.json")}
    utterances, _ = parse_directory(_corpus())
    ids = {u.utterance_id for u in utterances}
    assert len(ours) == 200
    missing = ours - ids
    assert not missing, f"{len(missing)} of our utterances are absent from the XML"


def test_gloss_vocabulary_is_reported(tmp_path: Path) -> None:
    v = gloss_vocabulary(parse_directory(_corpus())[0])
    assert len(v) > 1000
    assert sum(v.values()) > 10000


def test_an_unmapped_pair_is_reported_not_dropped(tmp_path: Path) -> None:
    p = tmp_path / "e.xml"
    p.write_text(REAL_XML)
    parsed, rep = parse_directory(tmp_path)
    assert len(parsed[0].non_manuals) == 3, "the unmappable event must still be returned"
    assert sum(rep.unmapped.values()) == 1
    assert "shrug" in " ".join(rep.unmapped)


def test_a_truncated_file_is_counted_not_fatal(tmp_path: Path) -> None:
    (tmp_path / "good.xml").write_text(REAL_XML)
    (tmp_path / "bad.xml").write_text("<SIGNSTREAM_DAI><COLLECTIONS><COLLECTION ID='1'>")
    got, rep = parse_directory(tmp_path)
    assert len(got) == 1, "the readable file must still be parsed"
    assert rep.files == 2
    assert rep.unparseable and "bad.xml" in rep.unparseable[0]


def test_report_serialises_with_a_mapped_fraction(tmp_path: Path) -> None:
    p = tmp_path / "mix.xml"
    p.write_text(REAL_XML)
    _, rep = parse_directory(tmp_path)
    d = rep.as_dict()
    assert d["non_manual_events"] == 3
    assert d["events_mapped_to_a_marker"] == 2
    assert d["mapped_fraction"] == round(2 / 3, 4)
    assert "caveat" in d


def test_parse_file_accumulates_into_one_report(tmp_path: Path) -> None:
    for i in range(3):
        (tmp_path / f"f{i}.xml").write_text(REAL_XML)
    rep = ParseReport()
    total = sum(len(parse_file(f, rep)) for f in sorted(tmp_path.glob("*.xml")))
    assert total == 3
    assert rep.files == 3 and rep.utterances == 3


def test_markers_present_view(tmp_path: Path) -> None:
    p = tmp_path / "f.xml"
    p.write_text(REAL_XML)
    u = parse_directory(tmp_path)[0][0]
    assert u.markers_present == {"brow_raise", "head_shake"}
    assert isinstance(u, Utterance)


def test_every_rule_has_a_non_empty_label_set() -> None:
    for marker, (labels, _values) in MARKER_RULES.items():
        assert labels, f"{marker} has no label to match"


def test_rule_keys_are_unique_and_marker_shaped() -> None:
    """MARKER_RULES maps marker -> (labels, values), so counting keys counts *values*."""
    assert len(MARKER_RULES) == len(set(MARKER_RULES))
    for marker in MARKER_RULES:
        assert isinstance(marker, str) and marker
        assert " " not in marker, f"marker names should be identifiers: {marker!r}"
    # A rule must not be reachable only through a value that no label can produce.
    for marker, (labels, values) in MARKER_RULES.items():
        assert labels or not values, f"{marker} constrains values with no label to match"


# --- events onto the clip's own frames ------------------------------------------------


def _utt(*events: tuple[int, int, str]) -> object:
    from seam.data.signstream import NonManual, Utterance

    return Utterance(
        utterance_id="1",
        participant="Cory",
        collection="c",
        collection_id="0",
        start_frame=1000,
        end_frame=1100,
        non_manuals=[
            NonManual(label="eye brows", value="raised", start_frame=a, end_frame=b, markers=[m])
            for a, b, m in events
        ],
    )


def test_frame_mask_is_frame_for_frame_on_a_30fps_clip() -> None:
    from seam.data.signstream import marker_frame_mask

    mask = marker_frame_mask(_utt((1010, 1019, "brow_raise")), "brow_raise", 101, 30.0)  # type: ignore[arg-type]
    assert mask.sum() == 10
    assert mask[10] and mask[19] and not mask[9] and not mask[20]


def test_frame_mask_rescales_on_a_24fps_clip() -> None:
    """The case that was wrong: 138 of the 200 EmoSign clips are 24 fps.

    Session frames 1050-1060 are 50-60 frames into the utterance, which is clip frames
    40-48 at 24 fps. A frame-for-frame mapping puts the label on 50-60, where on a
    two-second clip there is by then nothing left of the event.
    """
    from seam.data.signstream import marker_frame_mask

    mask = marker_frame_mask(_utt((1050, 1060, "brow_raise")), "brow_raise", 81, 24.0)  # type: ignore[arg-type]
    assert mask[40] and mask[48]
    assert not mask[39] and not mask[49]
    assert not mask[50:61].any()


def test_frame_mask_selects_by_marker_and_clips_to_the_clip() -> None:
    from seam.data.signstream import marker_frame_mask

    u = _utt((1000, 1004, "brow_raise"), (1095, 1300, "head_shake"), (2000, 2010, "head_shake"))
    brow = marker_frame_mask(u, "brow_raise", 101, 30.0)  # type: ignore[arg-type]
    shake = marker_frame_mask(u, "head_shake", 101, 30.0)  # type: ignore[arg-type]
    assert brow[:5].all() and brow.sum() == 5
    # Runs past the end: marks the frames that exist. Wholly outside: marks none.
    assert shake[95:].all() and shake.sum() == 6


def test_a_one_frame_event_is_not_rounded_away() -> None:
    from seam.data.signstream import marker_frame_mask

    mask = marker_frame_mask(_utt((1033, 1033, "blink")), "blink", 81, 24.0)  # type: ignore[arg-type]
    assert mask.sum() >= 1


def test_the_frame_mapping_is_validated_against_an_independent_signal() -> None:
    """In range is necessary; this is the check that can fail.

    Annotated blink frames against the eyeBlink blendshape share nothing but the video.
    Each clip-rate group must peak at the scale its frame rate implies, and on the
    24 fps clips the frame-rate mapping must beat frame-for-frame by a wide margin -
    0.710 against 0.554 when this was written.
    """
    import json

    path = Path(__file__).resolve().parents[1] / "artifacts" / "m3" / "frame_alignment.json"
    if not path.is_file():
        pytest.skip("frame_alignment.json not built; run scripts/check_signstream_alignment.py")
    groups = json.loads(path.read_text())["by_clip_fps"]
    assert set(groups) == {"24", "30"}

    slow = groups["24"]
    assert slow["frame_rate_mapping"]["auc"] - slow["frame_for_frame"]["auc"] >= 0.10
    assert slow["frame_for_frame"]["auc"] < 0.60, "frame-for-frame should be near chance here"
    assert max(slow["by_scale"], key=slow["by_scale"].get) in ("0.80", "0.82")

    fast = groups["30"]
    assert max(fast["by_scale"], key=fast["by_scale"].get) == "1.00"
    assert fast["frame_rate_mapping"]["auc"] >= 0.75
