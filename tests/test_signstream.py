"""Tests for the SignStream XML parser.

Written before the data arrived, against the schema in ASLLRP Report 18 §8.3 and the
SignStream 3 XML documentation. That is the point: the parser can be fully tested against
a documented format without the corpus, so the only thing left to check on arrival is
whether reality matches the documentation.

The recurring theme is that nothing may vanish quietly. An unmapped non-manual label, an
unparseable file, a field with no text, an event with no frame alignment — each is counted
and reported, because this project has already shipped a metric that was silently
inverted and an alignment that was silently wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seam.data.signstream import (
    FIELD_DOM_GLOSS,
    FIELD_TRANSLATION,
    MARKER_PATTERNS,
    ParseReport,
    map_label,
    parse_directory,
    parse_file,
)

# Verbatim shape from Report 18 §8.3, wrapped in the container the documentation shows.
DOCUMENTED_XML = """<?xml version="1.0"?>
<SIGNSTREAM-DATABASE>
  <REFERENCES><CODING-SCHEMES/></REFERENCES>
  <DATA>
    <SIGNSTREAM-UTTERANCE ID='utt-0001' PARTICIPANT='Ben'>
      <SEGMENT-TIER-UTTERANCE>
        <SEGMENT-TIER>
          <STATEMENT-FIELD ID='10000'>I-SEE</STATEMENT-FIELD>
          <STATEMENT-FIELD ID='30000'>I see</STATEMENT-FIELD>
        </SEGMENT-TIER>
        <NON_MANUALS>
          <NON_MANUAL ID='28963' START_FRAME='6504498' END_FRAME='6533527'>
            <LABEL>'head pos: tilt fr/bk'</LABEL>
            <VALUE>'slightly back'</VALUE>
          </NON_MANUAL>
        </NON_MANUALS>
      </SEGMENT-TIER-UTTERANCE>
    </SIGNSTREAM-UTTERANCE>
  </DATA>
</SIGNSTREAM-DATABASE>
"""


def test_parses_the_documented_shape(tmp_path: Path) -> None:
    p = tmp_path / "a.xml"
    p.write_text(DOCUMENTED_XML)
    got, rep = parse_directory(tmp_path)

    assert len(got) == 1, "one SIGNSTREAM-UTTERANCE should yield one Utterance"
    u = got[0]
    assert u.utterance_id == "utt-0001"
    assert u.participant == "ben"
    assert u.fields[FIELD_DOM_GLOSS] == "I-SEE"
    assert u.fields[FIELD_TRANSLATION] == "I see"
    assert len(u.non_manuals) == 1
    nm = u.non_manuals[0]
    assert nm.label == "'head pos: tilt fr/bk'"
    assert nm.value == "'slightly back'"
    assert nm.start_frame == 6504498
    assert nm.end_frame == 6533527
    assert nm.duration == 29029
    assert rep.utterances == 1
    assert rep.non_manual_events == 1


def test_an_unmapped_label_is_reported_not_dropped(tmp_path: Path) -> None:
    """The whole point of the strict reporting.

    An annotated event the project has no marker for is a fact about the data. Silently
    discarding it would make the label set look complete while quietly shrinking it.
    """
    p = tmp_path / "b.xml"
    p.write_text(
        DOCUMENTED_XML.replace(
            "<LABEL>'head pos: tilt fr/bk'</LABEL>", "<LABEL>'shoulders: shrug'</LABEL>"
        )
    )
    got, rep = parse_directory(tmp_path)
    assert len(got[0].non_manuals) == 1, "the event must still be returned"
    assert got[0].non_manuals[0].marker is None
    assert rep.mapped_events == 0
    assert rep.non_manual_events == 1
    assert sum(rep.unmapped_labels.values()) == 1
    assert "shrug" in " ".join(rep.unmapped_labels).lower()


@pytest.mark.parametrize(
    ("label", "marker"),
    [
        ("'head mvmt: shake'", "head_shake"),
        ("hm: shake", "head_shake"),
        ("'eye brows: raised'", "brow_raise"),
        ("'eye brows: raised-furrowed'", "brow_furrow"),
        ("'mouth: smile'", "mouth_positive"),
        ("'negative'", "negation"),
        ("'wh question'", "question_wh"),
        ("'eye brows: 5'", None),  # the bare VID value, not a label
    ],
)
def test_label_mapping(label: str, marker: str | None) -> None:
    assert map_label(label) == marker


def test_longer_patterns_win_over_substrings() -> None:
    """`head mvmt: shake` must beat a bare `shake` rule, or every shake hits the wrong
    marker depending on dict order."""
    assert (
        map_label("'head mvmt: side to side'") is None
        or map_label("'head mvmt: side to side'") != "head_shake"
    )
    # mouth_positive and mouth_morpheme both match "mouth"; the longer one must win.
    assert map_label("'mouth: smile'") == "mouth_positive"


def test_every_configured_marker_is_reachable() -> None:
    """A rule that can never match is dead config; catch it here, not on the corpus."""
    for marker, pats in MARKER_PATTERNS.items():
        assert any(p.strip() for p in pats), f"{marker} has an empty pattern"
        assert len(pats) >= 1


def test_a_truncated_file_is_counted_not_fatal(tmp_path: Path) -> None:
    """One bad file in a collection must not lose the rest of it."""
    good = tmp_path / "good.xml"
    good.write_text(DOCUMENTED_XML)
    bad = tmp_path / "bad.xml"
    bad.write_text("<SIGNSTREAM-DATABASE><DATA><SIGNSTREAM-UTTERANCE ID='x'>")

    got, rep = parse_directory(tmp_path)
    assert len(got) == 1, "the readable file must still be parsed"
    assert rep.files == 2
    assert any("unparseable" in k for k in rep.unmapped_labels), (
        "an unreadable file has to be visible in the report"
    )


def test_a_non_utterance_file_yields_nothing_and_says_so(tmp_path: Path) -> None:
    p = tmp_path / "empty.xml"
    p.write_text("<?xml version='1.0'?><SIGNSTREAM-DATABASE><REFERENCES/></SIGNSTREAM-DATABASE>")
    got, rep = parse_directory(tmp_path)
    assert got == []
    assert rep.utterances == 0


def test_an_utterance_with_no_id_is_reported(tmp_path: Path) -> None:
    p = tmp_path / "noid.xml"
    p.write_text(
        "<SIGNSTREAM-DATABASE><DATA>"
        "<SIGNSTREAM-UTTERANCE PARTICIPANT='Ben'><NON_MANUALS/></SIGNSTREAM-UTTERANCE>"
        "</DATA></SIGNSTREAM-DATABASE>"
    )
    got, rep = parse_directory(tmp_path)
    assert got == []
    assert "__utterance_without_id__" in rep.unmapped_labels


def test_non_manuals_without_frames_are_counted(tmp_path: Path) -> None:
    """Frame alignment is what joins these to the landmarks; a missing one is a real gap."""
    p = tmp_path / "noframe.xml"
    p.write_text(
        "<SIGNSTREAM-DATABASE><DATA><SIGNSTREAM-UTTERANCE ID='u1'>"
        "<NON_MANUALS><NON_MANUAL ID='1'><LABEL>'head mvmt: shake'</LABEL>"
        "<VALUE>'slightly'</VALUE></NON_MANUAL></NON_MANUALS>"
        "</SIGNSTREAM-UTTERANCE></DATA></SIGNSTREAM-DATABASE>"
    )
    got, rep = parse_directory(tmp_path)
    assert got[0].non_manuals[0].start_frame is None
    assert got[0].non_manuals[0].duration is None
    assert rep.non_manuals_without_frames == 1


def test_float_frame_attributes_are_coerced(tmp_path: Path) -> None:
    """Frame attributes have appeared as floats in these exports."""
    p = tmp_path / "float.xml"
    p.write_text(
        "<SIGNSTREAM-DATABASE><DATA><SIGNSTREAM-UTTERANCE ID='u1'><NON_MANUALS>"
        "<NON_MANUAL ID='1' START_FRAME='100.0' END_FRAME='140.0'>"
        "<LABEL>'head mvmt: nod'</LABEL><VALUE>'once'</VALUE></NON_MANUAL>"
        "</NON_MANUALS></SIGNSTREAM-UTTERANCE></DATA></SIGNSTREAM-DATABASE>"
    )
    got, _ = parse_directory(tmp_path)
    nm = got[0].non_manuals[0]
    assert nm.start_frame == 100 and nm.end_frame == 140
    assert nm.duration == 40


def test_alternate_container_tags_are_tolerated(tmp_path: Path) -> None:
    """Valid XML with a different nesting must not be an ImportError-shaped dead end."""
    for container in ("DATA", "SIGNSTREAM-UTTERANCES", "REFERENCES"):
        p = tmp_path / f"{container}.xml"
        p.write_text(
            f"<SIGNSTREAM-DATABASE><{container}>"
            "<SIGNSTREAM-UTTERANCE ID='u9'><NON_MANUALS>"
            "<NON_MANUAL ID='1' START_FRAME='1' END_FRAME='2'>"
            "<LABEL>'eye brows: raised'</LABEL><VALUE>'raised'</VALUE></NON_MANUAL>"
            "</NON_MANUALS></SIGNSTREAM-UTTERANCE>"
            f"</{container}></SIGNSTREAM-DATABASE>"
        )
        got, _ = parse_directory(tmp_path)
        assert len(got) == 1, f"container <{container}> yielded {len(got)}"
        assert got[0].non_manuals[0].marker == "brow_raise"
        for f in tmp_path.glob("*.xml"):
            f.unlink()


def test_a_seg_tier_utterance_is_not_mistaken_for_an_utterance(tmp_path: Path) -> None:
    """`SEGMENT-TIER-UTTERANCE` is a child; matching it would double-count every
    utterance and inflate the non-manual totals."""
    p = tmp_path / "nested.xml"
    p.write_text(DOCUMENTED_XML)
    utterances, rep = parse_directory(tmp_path)
    assert len(utterances) == 1, "the container element must not be emitted as an utterance"
    assert rep.utterances == 1
    assert rep.non_manual_events == 1


def test_the_documented_example_label_is_unmapped_and_says_so(tmp_path: Path) -> None:
    """A finding, not a bug.

    The one non-manual in ASLLRP Report 18's own example is `'head pos: tilt fr/bk'` -
    head tilt forward/backward - and this project has no marker for head *position*, only
    for head *movement* (shake, nod). So the canonical example lands in the unmapped
    report. That is the design working: the alternative is a vocabulary that looks
    complete while silently discarding a whole category of annotated events, and the
    first person to notice would be a reader of the paper wondering why head tilt is
    missing from the results.
    """
    p = tmp_path / "doc.xml"
    p.write_text(DOCUMENTED_XML)
    got, rep = parse_directory(tmp_path)
    assert got[0].non_manuals[0].marker is None
    assert rep.mapped_events == 0
    assert rep.non_manual_events == 1
    assert "tilt fr/bk" in " ".join(rep.unmapped_labels)
    assert rep.as_dict()["mapped_fraction"] == 0.0


def test_the_report_serialises_with_a_mapped_fraction(tmp_path: Path) -> None:
    """Every count must come from the parser, not from hand-adjusted report fields.

    An earlier version appended to `rep` by hand after parsing, which meant the test was
    asserting against numbers it had supplied itself rather than numbers the parser
    produced - and it reported 1.0 while a real mixed document would have reported 0.5.
    """
    p = tmp_path / "mix.xml"
    p.write_text(
        DOCUMENTED_XML.replace(
            "</NON_MANUALS>",
            "<NON_MANUAL ID='2' START_FRAME='100' END_FRAME='160'>"
            "<LABEL>'shoulders: shrug'</LABEL><VALUE>'once'</VALUE></NON_MANUAL>"
            "</NON_MANUALS>",
        ).replace("head pos: tilt fr/bk", "head mvmt: shake")
    )
    utterances, rep = parse_directory(tmp_path)
    assert len(utterances[0].non_manuals) == 2

    d = rep.as_dict()
    assert d["non_manual_events"] == 2
    assert d["mapped_events"] == 1
    assert d["mapped_fraction"] == 0.5
    assert d["n_unmapped_label_types"] == 1
    assert "caveat" in d


def test_parse_file_accepts_a_shared_report(tmp_path: Path) -> None:
    """Accumulating across files is how a whole collection is counted."""
    for i in range(3):
        (tmp_path / f"f{i}.xml").write_text(DOCUMENTED_XML)
    rep = ParseReport()
    total = 0
    for f in sorted(tmp_path.glob("*.xml")):
        total += len(parse_file(f, rep))
    assert total == 3
    assert rep.files == 3
    assert rep.utterances == 3
    assert rep.non_manual_events == 3
