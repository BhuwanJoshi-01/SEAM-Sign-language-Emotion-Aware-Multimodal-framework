"""ASLLRP frame-aligned sign tokens: the M5a recognition corpus.

17,522 tokens with gloss labels and frame boundaries, plus handshape annotations and
signer-tagged source collections. This is the real supervision M5a needs — no gloss
download risk, because the labels are already here.

**The file is not valid CSV, and the fix is a parser, not a skip.**

The gloss vocabulary contains labels with a bare double quote inside a quoted field,
e.g. ``"5"ok, hey""``. Per RFC 4180 an inner quote must be doubled; here it is not, so
every CSV reader splits that row into three fields instead of one and then fails
because the row has 23 fields where the header declares 20. Measured: ~2 rows in 6,000,
about 0.03% — rare enough that skipping would look fine and would quietly remove the
most unusual glosses, which are exactly the ones a recognition model finds hardest.

So :func:`_split_row` disambiguates with the rule the writer evidently intended: a
quoted region opens at a ``"`` that follows a delimiter or the line start, and closes
at a ``"`` followed by a delimiter or the line end. A ``"`` in any other position is
data. That resolves ``"5"ok, hey""`` to the single field ``5"ok, hey`` and leaves
ordinary rows untouched.

Every malformed row is still **counted and reported** (:func:`load` returns them in
``malformed_rows``), because a loader that cannot say what it dropped is a loader whose
token count cannot be trusted.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from seam.logging import get
from seam.paths import default_data_root

log = get(__name__)

#: Where the frame-aligned tokens live.
TOKENS_REL = Path("asllrp") / "asllrp_sentence_signs_2025_06_28.csv"

#: The columns the model actually uses. The rest (handshapes, master filenames,
#: `Hidden`) are kept in the record but are not features.
FRAME_COLUMNS = (
    "Start frame of the sign video",
    "End frame of the sign video",
    "Start frame of the containing utterance",
    "End frame of the containing utterance",
)


def tokens_path(root: Path | None = None) -> Path:
    return (root or default_data_root()) / TOKENS_REL


def _split_row(line: str) -> list[str]:
    """Split one CSV row, tolerating bare inner quotes in quoted fields.

    A quoted region opens at a ``"`` preceded by a delimiter or the line start, and
    closes at a ``"`` followed by a delimiter or the line end. Any other ``"`` is
    literal content.
    """
    fields: list[str] = []
    buf: list[str] = []
    in_quotes = False
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch == '"':
            at_field_start = i == 0 or line[i - 1] == ","
            rest = line[i + 1 :]
            closes = rest.startswith(",") or rest.strip() == ""
            if not in_quotes and at_field_start:
                in_quotes = True
            elif in_quotes and closes:
                in_quotes = False
            else:
                buf.append(ch)  # a bare quote inside the value
        elif ch == "," and not in_quotes:
            fields.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        i += 1
    fields.append("".join(buf).strip())
    return fields


@dataclass(slots=True)
class SignToken:
    """One frame-aligned sign occurrence."""

    video_id: str
    gloss: str
    start_frame: int
    end_frame: int
    utterance_start: int
    utterance_end: int
    utterance_video: str
    #: The signer, recovered from the source collection, e.g. ``Rachel_2011-12-08_sc45``.
    signer: str
    source_collection: str
    sign_type: str

    @property
    def duration_frames(self) -> int:
        return max(self.end_frame - self.start_frame, 0)

    @property
    def utterance_id(self) -> str:
        """Numeric utterance id, for joining to the gloss mapping.

        The utterance video filename is ``<id>.mp4``; the id is also the join key used by
        :mod:`seam.features.syntactic` for the EmoSign clips, which is what lets a
        recognised gloss line be given its interrogative/negation context.
        """
        stem = Path(self.utterance_video).stem
        return stem if stem.isdigit() else ""


def _signer_from_underscore(collection: str) -> str:
    """``Cory_2013-6-27_sc115`` -> ``Cory``.

    The underscore form carries the name cleanly, so the name set is derived from
    these rows rather than hardcoded - a hardcoded roster would silently pool two
    people into one LOSO group the moment a new signer appears, which is the exact
    leak LOSO exists to prevent.
    """
    if "_" not in collection:
        return ""
    head = collection.split("_", 1)[0].strip()
    if not head or not head.isalpha() or not head[0].isupper():
        return ""
    return head


#: ``8-Ben-Control-of-sound-places`` -> the name is the token straight after the index.
_DASHED = re.compile(r"^\d+-([A-Z][A-Za-z]*)-")


def _signer_from_dashed(collection: str) -> str:
    """``8-Ben-Control-of-sound-places`` -> ``Ben``.

    The dashed form is ``<index>-<Name>-<topic>`` and the name is always the token
    immediately after the index, so the pattern reads it off directly.

    The limitation is real and worth stating: in isolation this returns whatever token
    follows the index, so ``1-Introduction-x`` yields ``Introduction``. There is no way
    to tell a name from a topic word in one string. The protection is at the level of
    the derived *set* — :func:`load` reports ``signers``, and a topic word leaking in
    would appear there as an extra entry, which is visible in a way a per-row failure
    is not. A test asserts the set is exactly the four people in this corpus.

    Deriving the name set only from the underscore form does **not** work: one signer in
    this corpus appears exclusively in the dashed form, so their name never entered the
    known set and all 4,389 of their tokens were left unassigned. Those rows would then
    have pooled into one anonymous LOSO group - a leak of exactly the kind the split
    exists to prevent, arriving through a parse that looked fine.
    """
    m = _DASHED.match(collection)
    return m.group(1) if m else ""


def _signer_map(collections: Sequence[str]) -> dict[str, str]:
    """Build ``source_collection -> signer`` for every distinct collection.

    Each collection is resolved by whichever form it uses, and the two name sets are
    unioned. A collection that resolves to nothing is left unmapped and reported, rather
    than being assigned a wrong signer that would corrupt a split.
    """
    out: dict[str, str] = {}
    for c in dict.fromkeys(collections):
        if not c:
            continue
        name = _signer_from_underscore(c) or _signer_from_dashed(c)
        if name:
            out[c] = name
    return out


def load(root: Path | None = None, *, limit: int | None = None) -> tuple[list[SignToken], dict]:
    """Read the token table. Returns the tokens and a provenance report.

    The report carries the row counts and the malformed-row count, so a token count in
    a results table can be traced to a specific parse rather than taken on trust.
    """
    path = tokens_path(root)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `seam data fetch asllrp_gloss_tokens`")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = _split_row(lines[0])
    idx = {name: i for i, name in enumerate(header)}
    missing = [c for c in FRAME_COLUMNS if c not in idx]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")

    def col(row: Sequence[str], name: str) -> str:
        i = idx[name]
        return row[i] if i < len(row) else ""

    raw_collections = [
        _split_row(line)[idx["Source collection"]]
        if idx["Source collection"] < len(_split_row(line))
        else ""
        for line in lines[1:]
        if line.strip()
    ]
    signer_map = _signer_map(raw_collections)
    unmapped = sorted(set(raw_collections) - set(signer_map) - {""})

    tokens: list[SignToken] = []
    malformed: list[int] = []
    empty_gloss = 0
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        if limit is not None and len(tokens) >= limit:
            break
        row = _split_row(line)
        if len(row) != len(header):
            # Counted, never silently dropped.
            malformed.append(lineno)
            continue
        gloss = col(row, "Main entry gloss label")
        if not gloss:
            empty_gloss += 1
            continue
        try:
            start = int(float(col(row, FRAME_COLUMNS[0])))
            end = int(float(col(row, FRAME_COLUMNS[1])))
            ustart = int(float(col(row, FRAME_COLUMNS[2])))
            uend = int(float(col(row, FRAME_COLUMNS[3])))
        except ValueError:
            malformed.append(lineno)
            continue
        collection = col(row, "Source collection")
        tokens.append(
            SignToken(
                video_id=col(row, "Video ID number"),
                gloss=gloss,
                start_frame=start,
                end_frame=end,
                utterance_start=ustart,
                utterance_end=uend,
                utterance_video=col(row, "Utterance video filename"),
                signer=signer_map.get(collection, ""),
                source_collection=collection,
                sign_type=col(row, "Sign type"),
            )
        )

    glosses = sorted({t.gloss for t in tokens})
    signers = sorted({t.signer for t in tokens if t.signer})
    report = {
        "path": str(path),
        "n_tokens": len(tokens),
        "n_gloss_types": len(glosses),
        "n_signers": len(signers),
        "signers": signers,
        # A guard on the dashed-form heuristic, which reads the token after the
        # collection index and so cannot distinguish a name from a topic word in a
        # single string. If a topic word ever leaked in it would show up here as an
        # implausible signer count, which is far more visible than a per-row error.
        "signer_set_plausible": len(signers) <= 12,
        "n_utterances": len({t.utterance_id for t in tokens if t.utterance_id}),
        "n_source_collections": len(set(raw_collections)),
        "unmapped_source_collections": unmapped,
        "unmapped_collection_rows": sum(1 for c in raw_collections if c and c not in signer_map),
        "malformed_rows": len(malformed),
        "malformed_line_numbers": malformed[:20],
        "empty_gloss_rows": empty_gloss,
        "parse_note": (
            "The file contains bare inner double quotes inside quoted gloss labels, so "
            "it is not valid CSV. Rows are split with a quoted-region rule that treats "
            "an inner quote as data; malformed rows are counted here rather than "
            "dropped silently."
        ),
    }
    if malformed:
        log.warning("%s: %d malformed row(s) skipped", path.name, len(malformed))
    if not signers:
        log.warning("no signer could be recovered from the source collections")
    return tokens, report


@dataclass(slots=True)
class GlossVocab:
    """Gloss string <-> id, with the frequency ordering kept for the CTC blank."""

    itos: list[str] = field(default_factory=list)
    stoi: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, gloss: str) -> int:
        return self.stoi.get(gloss, -1)

    def decode(self, idx: int) -> str:
        return self.itos[idx] if 0 <= idx < len(self.itos) else "<unk>"

    @classmethod
    def build(cls, tokens: Sequence[SignToken], *, min_count: int = 1) -> GlossVocab:
        from collections import Counter

        counts = Counter(t.gloss for t in tokens)
        # Frequency-ordered so the CTC head's most frequent target is id 0, which makes
        # a collapsed model visibly biased toward the common sign rather than
        # randomly toward an arbitrary one.
        itos = [
            g for g, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if _ >= min_count
        ]
        return cls(itos=itos, stoi={g: i for i, g in enumerate(itos)})


#: Frame rate of the timeline every ASLLRP frame index is expressed in.
#:
#: Not documented in the token table, so it was measured two independent ways on the 200
#: EmoSign utterances (2026-10-05, `scripts/check_signstream_alignment.py`):
#:
#: * utterance span over extracted frame count is bimodal at 1.00 and 1.25 - 62 clips are
#:   30 fps and 138 are 24 fps re-encodes of the same footage;
#: * human blink annotations against the eye-blink blendshape: on the 24 fps clips a
#:   frame-for-frame mapping scores AUC 0.554 and a 24/30 rescale 0.710, with the sweep
#:   over scale peaking at 0.80-0.82 and falling to chance on either side; on the 30 fps
#:   clips frame-for-frame scores 0.807 and is the peak.
SESSION_FPS = 30.0


@dataclass(slots=True)
class AlignmentReport:
    """Whether the token frame indices line up with a given set of landmark frames.

    **Corrected 2026-10-05: the mapping needs the clip's frame rate, and until now it
    did not have it.** Token indices are positions on a 30 fps session timeline
    (:data:`SESSION_FPS`). 138 of the 200 EmoSign clips are 24 fps, so the
    frame-for-frame mapping this class used to document,

        crop frame index = session frame - utterance_start + 1,

    is right only for the other 62. On a 24 fps clip it runs 25% fast: a token that
    belongs at frames 80-88 was read from frames 100-110. That is what the "10% of tokens
    overshoot their crop by a frame or two" residual was - the last fifth of every 24 fps
    utterance falling off the end - and the tokens that stayed in range were labelling
    the wrong frames. M5a was trained on that mapping. The mapping is now

        crop frame index = round((session frame - utterance_start) * clip_fps / 30) + 1.

    History, kept because each step was a measurement compared against the wrong
    reference: 2026-09-28 concluded no mapping existed (absolute indices against clip
    length); 2026-09-29 found the per-utterance offset and concluded the mapping held
    for 89.5% of tokens (in range is not the same as aligned); this correction found the
    frame-rate mismatch by testing alignment against an independent signal instead of
    against frame counts.
    """

    n_utterances: int = 0
    n_tokens: int = 0
    #: Tokens whose mapped range fits inside the clip.
    n_tokens_aligned: int = 0
    #: Tokens that run past the clip end, or start before it, once mapped.
    n_tokens_overshoot: int = 0
    #: Median of (utterance span on the session timeline) / (frames in the clip), per
    #: clip frame rate. 1.00 at 30 fps and 1.25 at 24 fps is what a 30 fps timeline means.
    span_ratio_median_by_fps: dict[str, float] = field(default_factory=dict)
    n_utterances_by_fps: dict[str, int] = field(default_factory=dict)
    #: The same tokens under the frame-for-frame mapping, for the size of the correction.
    n_tokens_aligned_one_to_one: int = 0
    aligned: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "n_utterances": self.n_utterances,
            "n_tokens": self.n_tokens,
            "n_tokens_aligned": self.n_tokens_aligned,
            "n_tokens_overshoot": self.n_tokens_overshoot,
            "aligned_fraction": round(self.n_tokens_aligned / max(self.n_tokens, 1), 4),
            "aligned_percent": round(100.0 * self.n_tokens_aligned / max(self.n_tokens, 1), 1),
            "overshoot_percent": round(100.0 * self.n_tokens_overshoot / max(self.n_tokens, 1), 1),
            "n_tokens_in_range_one_to_one": self.n_tokens_aligned_one_to_one,
            "n_utterances_by_fps": dict(self.n_utterances_by_fps),
            "span_over_frames_ratio_median_by_fps": {
                k: round(v, 3) for k, v in self.span_ratio_median_by_fps.items()
            },
            "aligned": self.aligned,
            "session_fps": SESSION_FPS,
            "mapping": MAPPING,
            "conclusion": (
                "token indices are on a 30 fps session timeline and are rescaled by the "
                "clip's own frame rate; in range is necessary and not sufficient, so the "
                "mapping is validated against an independent signal in "
                "artifacts/m3/frame_alignment.json rather than by this count"
            ),
        }


MAPPING = "crop frame index = round((session frame - utterance_start) * clip_fps / 30) + 1"


def check_alignment(
    tokens: Sequence[SignToken],
    frame_counts: Mapping[str, int],
    clip_fps: Mapping[str, float],
) -> AlignmentReport:
    """Test whether token frame indices fit clips of the given frame counts and rates.

    ``frame_counts`` maps utterance id to the number of frames actually extracted for
    it and ``clip_fps`` to the frame rate of that clip. An utterance missing from either
    is not checked. "Fits" is a necessary condition only - see :class:`AlignmentReport`.
    """
    rows = [t for t in tokens if t.utterance_id in frame_counts and t.utterance_id in clip_fps]
    if not rows:
        return AlignmentReport(aligned=False, n_tokens=0)
    aligned_n = 0
    one_to_one = 0
    ratios: dict[str, list[float]] = {}
    seen: dict[str, str] = {}
    for t in rows:
        fc = int(frame_counts[t.utterance_id])
        fps = float(clip_fps[t.utterance_id])
        span = crop_frame_range(t, clip_fps=fps)
        if span is not None and span[1] <= fc:
            aligned_n += 1
        if t.start_frame >= t.utterance_start and t.end_frame - t.utterance_start < fc:
            one_to_one += 1
        if t.utterance_id not in seen and t.utterance_end > t.utterance_start:
            key = f"{fps:g}"
            seen[t.utterance_id] = key
            ratios.setdefault(key, []).append(
                (t.utterance_end - t.utterance_start + 1) / max(fc, 1)
            )
    return AlignmentReport(
        n_utterances=len({t.utterance_id for t in rows}),
        n_tokens=len(rows),
        n_tokens_aligned=aligned_n,
        n_tokens_overshoot=len(rows) - aligned_n,
        span_ratio_median_by_fps={k: float(np.median(v)) for k, v in sorted(ratios.items())},
        n_utterances_by_fps={k: len(v) for k, v in sorted(ratios.items())},
        n_tokens_aligned_one_to_one=one_to_one,
        # Aligned when the overwhelming majority map cleanly. Not requiring 100% would
        # be wrong in the other direction - a mapping that only works half the time is
        # not a mapping - so the threshold is high and the residual is reported.
        aligned=(aligned_n / len(rows)) >= 0.80,
    )


def crop_frame_position(session_frame: float, utterance_start: float, clip_fps: float) -> float:
    """0-based, fractional position in a clip of a frame on the session timeline.

    The one place the session-to-clip conversion lives. Token bounds and SignStream
    non-manual events both go through it, so they cannot be mapped two different ways.
    """
    if clip_fps <= 0:
        raise ValueError(f"clip_fps must be positive, got {clip_fps}")
    return (float(session_frame) - float(utterance_start)) * (float(clip_fps) / SESSION_FPS)


def crop_frame_index(token: SignToken, index_in_window: int, *, clip_fps: float) -> int | None:
    """Map a session-frame position to a 1-based crop-frame index.

    ``clip_fps`` is required and has no default. The mapping is frame-for-frame only for a
    30 fps clip, and a default of 30 would silently reproduce the misalignment this
    function carried until 2026-10-05 for every 24 fps clip.

    ``None`` when the position falls before the utterance. A position past the end of
    the clip is returned as it is: the caller knows the clip length and must drop the
    token rather than clamp it, since clamping would label the wrong frame.
    """
    pos = crop_frame_position(index_in_window, token.utterance_start, clip_fps)
    rel = int(np.floor(pos + 0.5)) + 1
    return rel if rel >= 1 else None


def crop_frame_range(token: SignToken, *, clip_fps: float) -> tuple[int, int] | None:
    """1-based inclusive ``(start, end)`` crop frames of a token, or ``None`` if before it."""
    start = crop_frame_index(token, token.start_frame, clip_fps=clip_fps)
    end = crop_frame_index(token, token.end_frame, clip_fps=clip_fps)
    if start is None or end is None:
        return None
    return start, max(end, start)


def describe(report: dict) -> str:
    return (
        f"ASLLRP tokens: {report['n_tokens']} over {report['n_utterances']} utterances, "
        f"{report['n_gloss_types']} gloss types, {report['n_signers']} signers "
        f"({', '.join(report['signers'])}); "
        f"{report['malformed_rows']} malformed row(s) skipped"
    )
