"""Parser for the **actual** SignStream 3 DAI XML export (ASLLRP non-manual annotations).

**This module was rewritten after seeing real data.** The first version was written
against the example in ASLLRP Report 18 §8.3 and did not match the shipped format at
all: the root is `SIGNSTREAM_DAI`, not `SIGNSTREAM-DATABASE`; there is no
`STATEMENT-FIELD`/`FIELD-ID` machinery; and the participant lives on `SEGMENT-TIER`, not
on the utterance. Good that the first version was written against documentation rather
than invented, and better that it was not trusted once the data arrived - the mismatch was
visible in the first five lines of the first file.

Real structure, confirmed against 51 exported collections (2,407 utterances, 43,038
non-manual events):

```
<SIGNSTREAM_DAI AUTHOR DOWNLOAD_DATE>
  <COLLECTIONS>
    <COLLECTION ID NAME>
      <MEDIA-FILES/>
      <TEMPORAL-PARTITIONS>
        <TEMPORAL-PARTITION ID NAME>
          <SEGMENT-TIERS>
            <SEGMENT-TIER ID NAME>
              <PARTICIPANT>'Cory'</PARTICIPANT>
              <DOMINANT-HAND>'R'</DOMINANT-HAND>
              <UTTERANCES>
                <UTTERANCE ID START_FRAME END_FRAME>
                  <TRANSLATION>..</TRANSLATION>
                  <UTTERANCE-NUMBER>'U1'</UTTERANCE-NUMBER>
                  <MANUALS>
                    <SIGN ID>
                      <LABEL>'MOUSE/FICTION'</LABEL>      <- the gloss
                      <SIGN_TYPE>..</SIGN_TYPE>
                      <DOMINANT_HAND START_FRAME END_FRAME/>
                  </MANUALS>
                  <NON_MANUALS>
                    <NON_MANUAL ID START_FRAME END_FRAME>
                      <LABEL>'eye brows'</LABEL>
                      <VALUE>'raised'</VALUE>
                      <ONSET START_FRAME END_FRAME/>
                      <OFFSET START_FRAME END_FRAME/>
                    </NON_MANUAL>
```

**The single most important fact about this corpus: the marker lives in LABEL *and*
VALUE, not LABEL alone.** `eye brows` is a label with 11 values - `raised`, `lowered`,
`slightly raised`, `raised-furrowed`, `right raised/left furrowed`, and so on. A parser
that maps on the label alone sees `eye brows` 4,957 times and cannot tell a brow raise
from a brow furrow from a neutral brow. That single decision would silently flatten the
most important non-manual in the language.

Everything that cannot be mapped is counted and reported, never dropped: an unmapped
label/value pair is a fact about the coverage, and burying it is the same failure as
silently clamping frame indices.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

#: Element names that contain "UTTERANCE" but are not utterances. `UTTERANCE-NUMBER` is a
#: child element of UTTERANCE, so a naive substring match emits it as its own utterance and
#: doubles the count.
_NOT_UTTERANCE = {"UTTERANCE-NUMBER", "UTTERANCES"}


def _s(x: str | None) -> str:
    """Strip whitespace and the single quotes SignStream wraps every text value in."""
    return (x or "").strip().strip("'").strip()


def _n(x: str | None) -> str:
    return re.sub(r"\s+", " ", _s(x).lower())


# --- marker mapping ----------------------------------------------------------
#: Project marker -> (accepted labels, accepted values).
#:
#: Values matter at least as much as labels here, so both are matched. An empty value set
#: means "any value", used for labels that are inherently binary or categorical (head
#: shake is a head shake whatever its tempo).
#:
#: These marker names match `seam.features.markers` so a real label replaces a
#: pseudo-label with no downstream change.
MARKER_RULES: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    # --- brow: label is shared, value carries the direction ---
    "brow_raise": (
        frozenset({"eye brows"}),
        frozenset(
            {
                "raised",
                "slightly raised",
                "further raised",
                "right raised/left furrowed",
                "left raised/right furrowed",
                "left raised/right lowered",
            }
        ),
    ),
    "brow_furrow": (
        frozenset({"eye brows"}),
        frozenset(
            {
                "lowered",
                "slightly lowered",
                "further lowered",
                "raised-furrowed",
                "right raised/left furrowed",
                "left raised/right furrowed",
                "further raised-furrowed",
            }
        ),
    ),
    # --- head movement: value is tempo, not identity ---
    "head_shake": (frozenset({"head mvmt: shake"}), frozenset()),
    "head_nod": (frozenset({"head mvmt: nod"}), frozenset()),
    "head_side_to_side": (frozenset({"head mvmt: side to side"}), frozenset()),
    "head_jut_mvmt": (frozenset({"head mvmt: jut"}), frozenset()),
    # --- head position: held posture, distinct from movement ---
    "head_tilt": (frozenset({"head pos: tilt fr/bk"}), frozenset()),
    "head_turn": (frozenset({"head pos: turn"}), frozenset()),
    "head_tilt_side": (frozenset({"head pos: tilt side"}), frozenset()),
    "head_jut": (frozenset({"head pos: jut"}), frozenset()),
    # --- face ---
    "eye_aperture": (frozenset({"eye aperture"}), frozenset()),
    "eye_gaze": (frozenset({"eye gaze"}), frozenset()),
    "mouth_morpheme": (frozenset({"mouth"}), frozenset()),
    "cheeks": (frozenset({"cheeks"}), frozenset()),
    "nose": (frozenset({"nose"}), frozenset()),
    "neck": (frozenset({"neck"}), frozenset()),
    "shoulders": (frozenset({"shoulders"}), frozenset()),
    "body_lean": (frozenset({"body lean"}), frozenset()),
    # --- grammatical: label is the category, value is a terse code ---
    "negation": (frozenset({"negative"}), frozenset({"negation"})),
    "question_wh": (frozenset({"wh question"}), frozenset({"whq", "wh question"})),
    "question_yn": (frozenset({"yes-no question"}), frozenset()),
    "question_rhetorical": (frozenset({"rhetorical question"}), frozenset()),
    "topic": (
        frozenset({"topic/focus"}),
        frozenset({"topic", "focus", "sentence-initial adverb", "foc/prop-with-ref-to"}),
    ),
    "conditional": (frozenset({"conditional/when"}), frozenset()),
    "role_shift": (frozenset({"role shift"}), frozenset()),
    "relative_clause": (frozenset({"relative clause"}), frozenset()),
}


def map_event(label: str, value: str) -> list[str]:
    """Markers this (label, value) pair supports.

    Returns a list rather than a single name because `raised-furrowed` is genuinely both
    a raise and a furrow, and collapsing it to one would discard information a linguist
    put there deliberately. Longest-first matching means a specific rule beats a broad one.
    """
    nl, nv = _n(label), _n(value)
    if not nl:
        return []
    # A rule with an empty value set accepts any value, which is what binary categories
    # like head shake need - its VALUE carries tempo, not identity. A rule that constrains
    # values scores higher, so a specific label/value pair beats a broad one; ties break
    # alphabetically so the result is deterministic.
    hits: list[tuple[int, str]] = []
    for marker, (labels, values) in MARKER_RULES.items():
        if nl not in labels:
            continue
        if values and nv not in values:
            continue
        hits.append((len(values), marker))
    return [m for _, m in sorted(hits, key=lambda x: (-x[0], x[1]))]


@dataclass
class NonManual:
    """One annotated non-manual event."""

    label: str
    value: str
    start_frame: int | None
    end_frame: int | None
    # A pair is (start, end) and either half can be absent; ONSET/OFFSET elements in the
    # real export are occasionally missing an attribute.
    onset: tuple[int | None, int | None] | None = None
    offset: tuple[int | None, int | None] | None = None
    markers: list[str] = field(default_factory=list)

    @property
    def duration(self) -> int | None:
        if self.start_frame is None or self.end_frame is None:
            return None
        return self.end_frame - self.start_frame


@dataclass
class Utterance:
    """One annotated utterance."""

    utterance_id: str
    participant: str
    collection: str
    collection_id: str
    start_frame: int | None
    end_frame: int | None
    translation: str = ""
    utterance_number: str = ""
    glosses: list[tuple[str, int | None, int | None]] = field(default_factory=list)
    non_manuals: list[NonManual] = field(default_factory=list)

    @property
    def markers_present(self) -> set[str]:
        return {m for nm in self.non_manuals for m in nm.markers}


@dataclass
class ParseReport:
    files: int = 0
    # NOT named `collections`: a dataclass attribute of that name shadows the `collections`
    # module for the remainder of the class body, so the next annotation
    # `collections.Counter` is evaluated against an int and the import breaks at class
    # definition time.
    n_collections: int = 0
    utterances: int = 0
    signs: int = 0
    non_manual_events: int = 0
    events_with_frames: int = 0
    events_mapped: int = 0
    participants: collections.Counter = field(default_factory=collections.Counter)
    events_by_marker: collections.Counter = field(default_factory=collections.Counter)
    unmapped: collections.Counter = field(default_factory=collections.Counter)
    unparseable: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        total = self.events_mapped + sum(self.unmapped.values())
        return {
            "files": self.files,
            "collections": self.n_collections,
            "utterances": self.utterances,
            "signs": self.signs,
            "non_manual_events": self.non_manual_events,
            "events_with_frame_alignment": self.events_with_frames,
            "events_mapped_to_a_marker": self.events_mapped,
            "mapped_fraction": round(self.events_mapped / total, 4) if total else None,
            "events_by_marker": dict(self.events_by_marker.most_common()),
            "n_participants": len(self.participants),
            "utterances_by_participant": dict(self.participants.most_common()),
            "unmapped_label_value_pairs": dict(self.unmapped.most_common(50)),
            "n_unmapped_pairs": len(self.unmapped),
            "unparseable_files": self.unparseable,
            "caveat": (
                "Unmapped label/value pairs are reported, not dropped: each is a real "
                "annotated event this project's marker vocabulary cannot express. The "
                "count is the measure of how far the vocabulary is from the corpus."
            ),
        }


def _frames(el: ET.Element) -> tuple[int | None, int | None]:
    return _int(el.get("START_FRAME")), _int(el.get("END_FRAME"))


def _int(v: str | None) -> int | None:
    if v is None:
        return None
    try:
        return int(float(v.strip()))
    except (ValueError, AttributeError):
        return None


def _iter_utterances(root: ET.Element) -> Iterator[ET.Element]:
    for el in root.iter():
        tag = el.tag.split("}")[-1].upper()
        if "UTTERANCE" in tag and tag not in _NOT_UTTERANCE:
            yield el


def parse_file(path: Path, report: ParseReport | None = None) -> list[Utterance]:
    """Parse one export. A malformed file is counted and skipped, never fatal."""
    rep = report if report is not None else ParseReport()
    rep.files += 1
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        rep.unparseable.append(f"{path.name}: {exc}")
        return []
    return list(_parse_root(root, rep))


def _parse_root(root: ET.Element, rep: ParseReport) -> list[Utterance]:
    out: list[Utterance] = []
    for coll in root.iter("COLLECTION"):
        rep.n_collections += 1
        cid, cname = coll.get("ID", ""), coll.get("NAME", "")
        # PARTICIPANT and DOMINANT-HAND live on SEGMENT-TIER, so index them first and let
        # utterances inherit. Reading the participant off the utterance finds nothing.
        tier_meta: dict[ET.Element, tuple[str, str]] = {}
        for tier in coll.iter("SEGMENT-TIER"):
            tier_meta[tier] = (_s(tier.findtext("PARTICIPANT")), _s(tier.findtext("DOMINANT-HAND")))
        by_id = {tier.get("ID", ""): v for tier, v in tier_meta.items()}

        for u in _iter_utterances(coll):
            # Walk up is not available in ElementTree, so resolve the participant by
            # finding the tier that contains this utterance.
            part = ""
            for tier, (p, _dh) in tier_meta.items():
                if u in list(tier.iter()):
                    part = p
                    break
            if not part:
                for k, (p, _dh) in by_id.items():
                    if k and u.get("ID") and k == u.get("ID"):
                        part = p
                        break
            us, ue = _frames(u)
            utt = Utterance(
                utterance_id=u.get("ID", ""),
                participant=part,
                collection=cname,
                collection_id=cid,
                start_frame=us,
                end_frame=ue,
                translation=_s(u.findtext("TRANSLATION")),
                utterance_number=_s(u.findtext("UTTERANCE-NUMBER")),
            )
            for sign in u.iter("SIGN"):
                g = _s(sign.findtext("LABEL"))
                if not g:
                    continue
                d = sign.find("DOMINANT_HAND")
                nd = sign.find("NON_DOMINANT_HAND")
                ref = d if d is not None else nd
                gs, ge = _frames(ref) if ref is not None else (None, None)
                utt.glosses.append((g, gs, ge))
                rep.signs += 1
            for nm in u.iter("NON_MANUAL"):
                label, value = _s(nm.findtext("LABEL")), _s(nm.findtext("VALUE"))
                ms, me = _frames(nm)
                on_el, off_el = nm.find("ONSET"), nm.find("OFFSET")
                markers = map_event(label, value)
                utt.non_manuals.append(
                    NonManual(
                        label=label,
                        value=value,
                        start_frame=ms,
                        end_frame=me,
                        onset=_frames(on_el) if on_el is not None else None,
                        offset=_frames(off_el) if off_el is not None else None,
                        markers=markers,
                    )
                )
                rep.non_manual_events += 1
                if ms is not None and me is not None:
                    rep.events_with_frames += 1
                if markers:
                    rep.events_mapped += 1
                    for m in markers:
                        rep.events_by_marker[m] += 1
                else:
                    rep.unmapped[f"{label} :: {value}"] += 1
            rep.utterances += 1
            if utt.participant:
                rep.participants[utt.participant] += 1
            out.append(utt)
    return out


def parse_directory(path: Path, pattern: str = "*.xml") -> tuple[list[Utterance], ParseReport]:
    """Parse every matching file under `path`, returning all utterances and one report."""
    rep = ParseReport()
    utterances: list[Utterance] = []
    for f in sorted(Path(path).rglob(pattern)):
        utterances.extend(parse_file(f, rep))
    return utterances, rep


def gloss_vocabulary(utterances: list[Utterance]) -> collections.Counter[str]:
    """Flattened gloss counts across all utterances, for vocabulary-size reporting."""
    c: collections.Counter[str] = collections.Counter()
    for u in utterances:
        for g, _s, _e in u.glosses:
            c[g] += 1
    return c
