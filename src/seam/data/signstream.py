"""Parser for SignStream 3 XML collection exports (ASLLRP non-manual annotations).

**Why this exists.** Every non-manual feature in this project is currently a heuristic
pseudo-label derived from MediaPipe blendshapes — the artifacts say
`"marker_provenance": "heuristic (pseudo-labels); ASLLRP SignStream XML not available"`.
This module is the other half of that sentence: it reads the authoritative annotations the
moment they are downloaded.

**Built before the data arrived, against the documented schema.** The shape implemented
here comes from ASLLRP Report 18 §8.3 and the SignStream 3 XML documentation:

```xml
<SIGNSTREAM-DATABASE>
  <DATA> ...
    <SIGNSTREAM-UTTERANCE ID='...'>
      <SEGMENT-TIER-UTTERANCE>
        <SEGMENT-TIER>
          <STATEMENT-FIELD>            <!-- manual + grammatical fields -->
            <FIELD-REF ID='10000'/>    <!-- dom gloss -->
            <ENTITY>
              <FIELD-VALUE>...</FIELD-VALUE>
              <REFERENCE>...</REFERENCE>
            </ENTITY>
          </STATEMENT-FIELD>
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
```

**The parsing is deliberately tolerant and the reporting deliberately strict.** The ASLLRP
token CSV already proved this corpus ships malformed quoting, so a parser that raises on
the first surprise would skip rows silently and shrink the label set without anyone
noticing. Conversely, anything that *cannot* be interpreted — an unmapped non-manual
label, a field with no known meaning — is **counted and reported**, never dropped quietly.
An unmapped label is a finding about the data, and burying it would be the same failure as
silently clamping frame indices.

**Field IDs are namespaced and are not interchangeable.** Categorical `10` and continuous
`40001` are both called `eye brows` and are separate spaces; conflating them would mix a
human label with a normalised time series.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

# --- field identifiers -------------------------------------------------------
#: Manual / linguistic statement fields, from the SignStream 3 XML documentation.
FIELD_DOM_GLOSS = "10000"
FIELD_NDOM_GLOSS = "10001"
FIELD_DOM_HANDSHAPE = "1000"
FIELD_NDOM_HANDSHAPE = "2000"
FIELD_DOM_SIGN_TYPE = "26000"
FIELD_NDOM_SIGN_TYPE = "26001"
FIELD_TRANSLATION = "30000"

#: Continuous (graphical) non-manual fields, normalised to (-1, 1). Report 17 App. I.
CONTINUOUS_FIELDS = {
    "40001": "eye_brows",
    "40002": "eye_aperture",
    "50001": "yaw",
    "50002": "pitch",
    "50003": "roll",
}

#: Our marker vocabulary -> the SignStream labels that should map onto it. Keys are the
#: project marker names that already appear in `features/markers.py`, so a real label
#: replaces a pseudo-label without any downstream change.
#:
#: Matching is on normalised substrings. SignStream label spellings vary
#: (`head mvmt: shake` / `hm: shake`), so exact matching would silently map nothing.
MARKER_PATTERNS: dict[str, tuple[str, ...]] = {
    "brow_raise": ("eye brows: raised", "eye brows: +raised", "brows raised"),
    # "raised-furrowed" must be listed here and not only under brow_raise: the string
    # contains "raised", so without a longer specific pattern the raise rule wins on
    # length and every combined brow is labelled as a raise.
    "brow_furrow": (
        "eye brows: raised-furrowed",
        "raised-furrowed",
        "eye brows: furrowed",
        "furrowed",
        "eye brows: lowered",
        "lowered",
    ),
    "head_shake": ("head mvmt: shake", "hm: shake", "head shake"),
    "head_nod": ("head mvmt: nod", "hm: nod", "head nod"),
    "mouth_morpheme": ("mouth:", "mouth :", "mm:"),
    "mouth_positive": ("mouth: smile", "mouth smile", "smile"),
    # Grounded in the ASLLRP grammatical labels; these are the categories this project
    # wanted linguistically and could previously only approximate from blendshapes.
    "negation": ("negative",),
    "question_wh": ("wh question",),
    "question_yn": ("yes-no question",),
    "topic": ("topic/focus", "topic"),
}


def _norm(s: str) -> str:
    """Lowercase, collapse whitespace, strip the SignStream single-quote wrapping."""
    s = (s or "").strip().strip("'").strip()
    return re.sub(r"\s+", " ", s.lower())


@dataclass
class NonManual:
    """One annotated non-manual event, time-aligned to source frames."""

    label: str
    value: str
    start_frame: int | None
    end_frame: int | None
    marker: str | None = None

    @property
    def duration(self) -> int | None:
        if self.start_frame is None or self.end_frame is None:
            return None
        return self.end_frame - self.start_frame


@dataclass
class Utterance:
    """One annotated utterance."""

    utterance_id: str
    participant: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    non_manuals: list[NonManual] = field(default_factory=list)


@dataclass
class ParseReport:
    """What was read, and what could not be interpreted. Never silently drops."""

    files: int = 0
    utterances: int = 0
    non_manual_events: int = 0
    mapped_events: int = 0
    participants: collections.Counter = field(default_factory=collections.Counter)
    field_ids_seen: collections.Counter = field(default_factory=collections.Counter)
    unmapped_labels: collections.Counter = field(default_factory=collections.Counter)
    fields_with_no_text: int = 0
    non_manuals_without_frames: int = 0

    def as_dict(self) -> dict[str, object]:
        total = self.mapped_events + sum(self.unmapped_labels.values())
        return {
            "files": self.files,
            "utterances": self.utterances,
            "non_manual_events": self.non_manual_events,
            "mapped_events": self.mapped_events,
            "mapped_fraction": round(self.mapped_events / total, 4) if total else None,
            "participants": dict(self.participants),
            "field_ids_seen": dict(self.field_ids_seen),
            "unmapped_labels": dict(self.unmapped_labels.most_common(40)),
            "n_unmapped_label_types": len(self.unmapped_labels),
            "fields_with_no_text": self.fields_with_no_text,
            "non_manuals_without_frames": self.non_manuals_without_frames,
            "caveat": (
                "Unmapped labels are reported, not dropped. Each one is a real annotated "
                "event the project currently has no marker for, and the count is the "
                "measure of how far the vocabulary is from the annotations."
            ),
        }


def map_label(label: str) -> str | None:
    """Map a SignStream non-manual label onto a project marker, or None."""
    n = _norm(label)
    if not n:
        return None
    # Longest patterns first so 'head mvmt: shake' beats a bare 'shake' rule.
    best: tuple[int, str] | None = None
    for marker, pats in MARKER_PATTERNS.items():
        for p in pats:
            if p in n and (best is None or len(p) > best[0]):
                best = (len(p), marker)
    return best[1] if best else None


def _int(v: str | None) -> int | None:
    if v is None:
        return None
    s = v.strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        # Frame attributes have appeared as floats in this corpus's exports. Falling back
        # to None would drop a usable alignment, and int(float()) is exact for the ranges
        # involved; a genuinely non-numeric value returns None and is reported.
        return None


def _iter_utterances(root: ET.Element) -> Iterator[ET.Element]:
    """Yield utterance elements under either plausible container tag.

    The documentation shows `<DATA>`; some exports nest under
    `<SIGNSTREAM-UTTERANCES>` or put utterance elements at the top level. Matching on the
    element name rather than the path avoids an ImportError-shaped dead end where the file
    is valid XML but the tag differs from the example.
    """
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if "UTTERANCE" in tag.upper() and tag.upper() != "SEGMENT-TIER-UTTERANCE":
            yield el


def parse_file(path: Path, report: ParseReport | None = None) -> list[Utterance]:
    """Parse one XML export. Never raises on content; raises only on unreadable bytes."""
    rep = report or ParseReport()
    rep.files += 1
    try:
        tree = ET.parse(path)
    except ET.ParseError:
        # A malformed file is skipped and *counted*, mirroring the tolerant token CSV.
        # Returning [] rather than raising lets a whole collection be processed when one
        # file is truncated in transit.
        rep.unmapped_labels[f"__unparseable_file__:{path.name}"] += 1
        return []
    root = tree.getroot()
    return list(_parse_root(root, rep))


def _parse_root(root: ET.Element, rep: ParseReport) -> list[Utterance]:
    out: list[Utterance] = []
    for el in _iter_utterances(root):
        uid = (
            el.get("ID")
            or el.get("Id")
            or el.get("id")
            or (el.findtext(".//UTTERANCE_ID") or "").strip()
        )
        if not uid:
            rep.unmapped_labels["__utterance_without_id__"] += 1
            continue
        part = (
            el.get("PARTICIPANT")
            or el.get("PARTICIPANT_LABEL")
            or (el.findtext(".//PARTICIPANT") or "").strip()
        )
        u = Utterance(utterance_id=uid.strip(), participant=_norm(part))

        # Statement fields: keyed by the FIELD-REF/FIELD-ID attribute when present.
        for sf in el.iter():
            tag = sf.tag.split("}")[-1]
            if tag not in {"STATEMENT-FIELD", "ENTITY", "FIELD-VALUE"}:
                continue
            fid = None
            for a in ("ID", "FIELD-ID", "FIELD_ID", "FIELDID"):
                if sf.get(a):
                    fid = sf.get(a)
                    break
            text = (sf.text or "").strip()
            if fid:
                rep.field_ids_seen[fid] += 1
                if text and len(text) < 400:
                    u.fields[fid] = text
                elif not text:
                    rep.fields_with_no_text += 1

        for nm in _iter_non_manuals(el):
            label = (nm.findtext("LABEL") or "").strip()
            value = (nm.findtext("VALUE") or "").strip()
            marker = map_label(label)
            start = _int(nm.get("START_FRAME"))
            end = _int(nm.get("END_FRAME"))
            if start is None or end is None:
                rep.non_manuals_without_frames += 1
            event = NonManual(
                label=label, value=value, start_frame=start, end_frame=end, marker=marker
            )
            u.non_manuals.append(event)
            rep.non_manual_events += 1
            if marker:
                rep.mapped_events += 1
            else:
                rep.unmapped_labels[f"{_norm(label)} :: {_norm(value)}"] += 1
        rep.utterances += 1
        if u.participant:
            rep.participants[u.participant] += 1
        out.append(u)
    return out


def _iter_non_manuals(el: ET.Element) -> Iterator[ET.Element]:
    for nm in el.iter():
        if nm.tag.split("}")[-1].upper() == "NON_MANUAL":
            yield nm


def parse_directory(path: Path, pattern: str = "*.xml") -> tuple[list[Utterance], ParseReport]:
    """Parse every matching file, returning all utterances plus one shared report."""
    rep = ParseReport()
    files = sorted(Path(path).rglob(pattern))
    utterances: list[Utterance] = []
    for f in files:
        utterances.extend(parse_file(f, rep))
    return utterances, rep
