"""Grounding the feature set in what Deaf annotators actually wrote (M3).

This is the grounding move the plan calls for: instead of assuming a prosody and
non-manual feature set is the right one, check it against the 600 free-text cue
strings the three Deaf annotators wrote for the 200 EmoSign clips. The annotators
named sign size, speed, repetition, emphatic fingerspelling, brow and head-shake. So
the question is concrete: **do our features recover the cues they named?**

Three things this module does deliberately, each for a reason learned the hard way.

**It separates motor cues from affective interpretations.** The annotators write both
kinds of thing, and they are not the same kind of evidence. "raised eye brow" and
"bared teeth" describe a movement, and a movement is something a feature can be
expected to recover. "conveys a sense of surprise" and "signifies worry" are the
annotator's *reading* of an affect, and no feature in :mod:`seam.features.prosody` or
:mod:`seam.features.markers` is a claim about affect - those are FER's job. Treating
an affective interpretation as validation of a brow-raise feature would be a category
error, and it would manufacture agreement. So the two are parsed separately, the
affective ones are counted, and they are excluded from the motor grounding test.

**It reports coverage rather than assuming it.** A cue vocabulary that silently
discards the strings it cannot parse reports a clean result on the remainder and
hides the selection. Every parse therefore reports how many of the 600 strings were
matched, by category, and the unmatched ones are inspectable.

**The cue-to-feature mapping is declared here, in one place, before any number is
computed.** Choosing which feature to test after seeing which one works is how an
assumed feature set becomes a confirmed one without any evidence having moved.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from seam.paths import default_data_root

#: Where the annotators wrote. Three columns per clip, 200 clips, 600 strings.
CUE_COLUMNS = ("Reasoning_1", "Reasoning_2", "Reasoning_3")


#: A cue the annotators named, and what it should look like in our features.
#:
#: ``expect`` lists the seam features that a correct implementation should move for
#: this cue. It is the *test*: a cue whose expected features show nothing is a cue we
#: do not currently detect, which is a finding about the detector rather than about
#: the annotators.
@dataclass(frozen=True, slots=True)
class CueSpec:
    name: str
    patterns: tuple[str, ...]
    expect: tuple[str, ...]
    #: Why this cue should move these features, in one line. Recorded so a mapping
    #: can be argued with rather than merely trusted.
    rationale: str


#: The motor-cue vocabulary, read out of the corpus rather than assumed. The patterns
#: were written by mining the 600 strings, including the ones a first pass failed to
#: match - "eyes popped", "bared teeth", "grimace", "lower than usual" are all
#: observable cues that an initial vocabulary missed.
MOTOR_CUES: tuple[CueSpec, ...] = (
    CueSpec(
        "brow_raise",
        (
            r"\bbrow\w*\s*(?:raise|up|lift)",
            r"\brais\w*\s*(?:eye ?)?brow",
            r"\beye ?brows?\s+(?:up|raise)",
            r"\bupward\w*\s+brow",
            r"\bhigh\w*\s+brow",
            r"\barch\w*\s+brow",
        ),
        ("brow_raise",),
        "a raised brow is the single most-named non-manual cue and is what browRaise measures",
    ),
    CueSpec(
        "brow_furrow",
        (r"\bfurrow\w*", r"\bbrow\w*\s*(?:down|lower|knit)", r"\bsquint\w*", r"\bcrease\w*\s+brow"),
        ("brow_furrow",),
        "a furrowed brow is the wh-question marker and is what browDown measures",
    ),
    CueSpec(
        "head_shake",
        (r"\bshak\w*", r"\bside\s*to\s*side\b", r"\bswing\w*\s+head", r"\bhead\s+wobbl\w*"),
        ("head_shake",),
        "negation is canonically marked by a head shake, so yaw oscillation is the test",
    ),
    CueSpec(
        "head_nod",
        (r"\bnod\w*", r"\bhead\s+(?:up\s*and\s*down|bob\w*)"),
        ("head_nod",),
        "affirmation and emphasis are marked by a nod, so pitch oscillation is the test",
    ),
    CueSpec(
        "head_tilt",
        (r"\btilt\w*", r"\blean\w*", r"\bhead\s+(?:to\s+the\s+side|angle\w*)"),
        (),
        "annotators name head tilt often, but no feature in the current set measures it - "
        "a stated gap, not a miss",
    ),
    CueSpec(
        "mouth_shape",
        (
            r"\bmouth\w*",
            r"\bmouthing\b",
            r"\blip\w*",
            r"\blips\b",
            r"\bpout\w*",
            r"\bbared?\s+teeth",
            r"\bgrimac\w*",
            r"\btongue\b",
            r"\bteeth\b",
            r"\bopen(?:ed)?\s+mouth",
        ),
        ("mouth_morpheme",),
        "mouth morphemes and mouthing should move the mouth coefficient set",
    ),
    CueSpec(
        "smile",
        (r"\bsmil\w*", r"\bgrinn\w*", r"\bcheer\w*", r"\bhappy\s+face"),
        ("mouth_positive",),
        "a smile should raise the mouthSmile coefficients; mouth_positive exists as the "
        "affective control, so this is the cue that tests it",
    ),
    CueSpec(
        "eye_widen",
        (
            r"\bwid\w*\s*(?:open\s*)?ey\w*",
            r"\beyes?\s+(?:pop|wide|bulge)",
            r"\bbulging\w*",
            r"\bwide\w*\s+ey\w*",
        ),
        (),
        "eye aperture is named by annotators but no current feature measures it",
    ),
    CueSpec(
        "blink_close",
        (
            r"\bblink\w*",
            r"\beyes?\s+closed?\b",
            r"\beyes?\s+shut\b",
            r"\bsquint\w*",
            r"\bnarrow\w*\s+ey\w*",
        ),
        (),
        "blink rate and eye closure are named but not currently extracted",
    ),
    CueSpec(
        "gaze_shift",
        (
            r"\bgaz\w*",
            r"\beye\s*gaze\b",
            r"\blook\w*\s+(?:up|down|away|around|back)",
            r"\beye\s+contact\b",
        ),
        (),
        "gaze is named and the perception stage extracts a gaze proxy, but it is not in "
        "the prosody feature set",
    ),
    CueSpec(
        "sign_size",
        (
            r"\b(?:large|big|bigger)\b",
            r"\bsmall(?:er)?\b",
            r"\bsign(?:s)?\s+(?:size|are|is|were)\b",
            r"\b(?:more|less)\s+space\b",
            r"\blower\s+than\s+usual\b",
            r"\bhigher\s+than\s+usual\b",
            r"\btight\w*\s+space\b",
        ),
        ("amplitude",),
        "sign size is a hand-trajectory amplitude question, which is what prosody.amplitude "
        "measures",
    ),
    CueSpec(
        "speed_fast",
        (r"\bfast\w*", r"\bquick\w*", r"\brapid\w*", r"\bhigh\s+pace\b", r"\burgen\w*"),
        ("speed",),
        "perceived speed should raise prosody.speed",
    ),
    CueSpec(
        "speed_slow",
        (r"\bslow\w*", r"\bdeliberat\w*", r"\bmeasured\s+pace\b"),
        ("speed",),
        "perceived slowness should lower prosody.speed",
    ),
    CueSpec(
        "repetition",
        (r"\brepeat\w*", r"\brepetit\w*", r"\bagain\b", r"\btwice\b", r"\bmultiple\s+times\b"),
        ("repetition",),
        "repetition is a periodicity question, which is what prosody.repetition measures",
    ),
    CueSpec(
        "emphasis",
        (
            r"\bemphasi\w*",
            r"\bemphatic\w*",
            r"\bstron\w*\s+(?:sign|expression|movement)",
            r"\bexaggerat\w*",
            r"\binten\w*",
            r"\bforceful\w*",
        ),
        ("jerk", "volume", "amplitude"),
        "emphasis is a kinetic-amplitude question; jerk, movement volume and spatial "
        "amplitude are the features that should move",
    ),
    CueSpec(
        "fingerspelling",
        (r"\bfinger\s?spell\w*", r"\bspell\w*\s+out\b", r"\bhand\s?spell\w*", r"\bdidactic\w*"),
        (),
        "fingerspelling is named and is recognisable from handshape, but nothing in the "
        "current prosody set measures it",
    ),
    CueSpec(
        "pause_hesitation",
        (
            r"\bpause\w*",
            r"\bhesitat\w*",
            r"\bpause\w*\s+before",
            r"\bfalt\w*",
            r"\bstop\s+and\s+start",
        ),
        ("pause_ratio", "pause_mean"),
        "a described pause should show up as pause statistics",
    ),
    CueSpec(
        "body_posture",
        (r"\bposture\b", r"\bshoulder\w*", r"\blean\w*", r"\bbody\b", r"\bchest\b", r"\btorso\b"),
        (),
        "posture is named but the current feature set is hand-trajectory only",
    ),
)

#: Affective *interpretations* - the annotator's reading, not a movement. Counted and
#: reported, and excluded from the motor grounding test by construction. These are
#: the annotator's affect labels and therefore belong to FER's evaluation, not to the
#: marker or prosody features.
AFFECTIVE_PATTERNS: tuple[str, ...] = (
    r"\bconvey\w*",
    r"\bsignif\w*",
    r"\bsuggest\w*",
    r"\bimply\w*",
    r"\bindicat\w*",
    r"\bexpress\w*",
    r"\bfeel\w*",
    r"\bseem\w*",
    r"\bsurprise\w*",
    r"\bsurpris\w*",
    r"\bworr\w*",
    r"\bsad\w*",
    r"\bhapp\w*",
    r"\bangr\w*",
    r"\bfrustrat\w*",
    r"\bdisgust\w*",
    r"\bafraid\b",
    r"\bfear\b",
    r"\bexcit\w*",
    r"\bjoy\w*",
    r"\bneutra\w*",
    r"\bpositiv\w*",
    r"\bnegativ\w*",
    r"\bcurious\w*",
    r"\bconfus\w*",
    r"\bamaz\w*",
    r"\binterest\w*",
    r"\bbored?\b",
    r"\bboring\b",
    r"\bemphasi\w*",
    r"\bimpression\b",
    r"\bmood\b",
)

MOTOR_CUE_NAMES = tuple(c.name for c in MOTOR_CUES)


def labels_path(root: Path | None = None) -> Path:
    return (root or default_data_root()) / "emosign" / "emosign_dataset.csv"


def _compiled(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.I) for p in patterns]


_MOTOR_RE = {c.name: _compiled(c.patterns) for c in MOTOR_CUES}
_AFFECTIVE_RE = _compiled(AFFECTIVE_PATTERNS)


@dataclass(slots=True)
class ParsedCue:
    """One annotator string, parsed.

    ``motor`` maps cue name to the exact span that matched, so a parse can be
    checked against the sentence it came from instead of taken on trust.
    """

    text: str
    motor: dict[str, str] = field(default_factory=dict)
    affective: list[str] = field(default_factory=list)

    @property
    def motor_cues(self) -> list[str]:
        return sorted(self.motor)

    @property
    def is_affective_only(self) -> bool:
        return bool(self.affective) and not self.motor


@dataclass(slots=True)
class ClipCues:
    """The three annotator strings for one clip, merged.

    A clip's cue set is the union across its strings: the three annotators described
    the same clip, so a cue any of them named is present in it. Per-string provenance
    is kept so a clip's cues can be traced back to who said what.
    """

    utterance_id: str
    motor_cues: list[str] = field(default_factory=list)
    affective: list[str] = field(default_factory=list)
    spans: dict[str, str] = field(default_factory=dict)
    n_strings: int = 0

    def has(self, name: str) -> bool:
        return name in self.motor_cues


def parse_cue(text: str) -> ParsedCue:
    """Parse one annotator string into motor cues, plus whether it is affective.

    Returns the matched cue names and the matched span per cue, so a parse can be
    audited against the sentence it came from rather than taken on trust.
    """
    out = ParsedCue(text=text)
    for name, regs in _MOTOR_RE.items():
        for rx in regs:
            if m := rx.search(text):
                out.motor[name] = m.group(0)
                break
    out.affective = [m.group(0) for rx in _AFFECTIVE_RE if (m := rx.search(text))]
    return out


def iter_cue_strings(root: Path | None = None) -> list[tuple[str, str, str]]:
    """Every annotator string as ``(utterance_id, column, text)``.

    600 strings over 200 clips. Returned with the utterance id so a cue can be
    attributed to a clip and its landmarks joined, which is the whole point.
    """
    import pandas as pd

    path = labels_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `seam data fetch emosign_labels`")
    frame = pd.read_csv(path)
    name_col = "video_name"
    out: list[tuple[str, str, str]] = []
    for row in frame.itertuples(index=False):
        vid = str(getattr(row, name_col, ""))
        m = re.search(r"(\d+)$", vid)
        uid = m.group(1) if m else vid
        for col in CUE_COLUMNS:
            val = getattr(row, col, None)
            if val is None:
                continue
            text = str(val).strip()
            if text and text.lower() != "nan":
                out.append((uid, col, text))
    return out


def clip_cues(root: Path | None = None) -> dict[str, ClipCues]:
    """Merge the three annotator strings per clip into one cue record.

    A clip's cue set is the union across its three strings, because the annotators
    each described the same clip and a cue any of them named is present in it. The
    per-string provenance is kept so a clip's cues can be traced back to who said
    what.
    """
    merged: dict[str, ClipCues] = {}
    for uid, col, text in iter_cue_strings(root):
        parsed = parse_cue(text)
        rec = merged.setdefault(uid, ClipCues(utterance_id=uid))
        rec.n_strings += 1
        rec.affective.extend(parsed.affective)
        for name in parsed.motor:
            if name not in rec.motor_cues:
                rec.motor_cues.append(name)
            rec.spans.setdefault(name, f"{col}: {parsed.motor[name]}")
    for rec in merged.values():
        rec.motor_cues.sort()
    return merged


def coverage(root: Path | None = None) -> dict[str, object]:
    """How much of the annotators' text the vocabulary accounts for.

    Reported because a cue vocabulary that silently drops what it cannot parse
    reports a clean result on the remainder and hides the selection.
    """
    strings = iter_cue_strings(root)
    n = len(strings)
    matched = 0
    affective_only = 0
    per_cue: dict[str, int] = dict.fromkeys(MOTOR_CUE_NAMES, 0)
    for _uid, _col, text in strings:
        parsed = parse_cue(text)
        if parsed.motor:
            matched += 1
        else:
            affective_only += 1
        for c in parsed.motor:
            per_cue[c] += 1
    return {
        "n_strings": n,
        "n_with_motor_cue": matched,
        "n_affective_only": affective_only,
        "motor_coverage": round(matched / n, 4) if n else 0.0,
        "per_cue_strings": per_cue,
        "note": (
            "Strings with no motor cue are not discarded. They are the annotators' "
            "affective readings, which are ground truth for FER and are excluded from "
            "the motor grounding test rather than counted as a miss."
        ),
    }


def expectations() -> dict[str, dict[str, object]]:
    """The declared cue-to-feature map, for the methods section."""
    return {
        c.name: {
            "expects": list(c.expect),
            "testable_now": bool(c.expect),
            "rationale": c.rationale,
        }
        for c in MOTOR_CUES
    }


def unmapped_cues() -> list[str]:
    """Cues the annotators name that the current feature set cannot test.

    This is a to-do list derived from the data rather than from taste: eight of the
    cue categories Deaf annotators actually used have no feature behind them, which
    is a concrete specification for M4's feature set.
    """
    return [c.name for c in MOTOR_CUES if not c.expect]
