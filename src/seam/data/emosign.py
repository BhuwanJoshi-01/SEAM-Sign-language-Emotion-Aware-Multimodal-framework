"""EmoSign labels, and the join that makes the affect benchmark reachable.

EmoSign is 200 utterances annotated by three Deaf native signers with professional
interpreting experience: a 7-point sentiment, ten emotion intensities, and three
free-text columns naming the visual cues they actually used. It is the only
sign-language corpus in existence with affect labels.

The labels are ungated and 43 KB. The *video* was the blocker, because EmoSign is
built on ASLLRP and ASLLRP is access-controlled. It is resolved by a join that is
not obvious and is worth stating plainly:

    EmoSign's ``video_name`` is not a filename. Its trailing numeric token is the
    ASLLRP utterance ID.

    ``Jonathan_2012-11-27_sc93_5572615``  ->  ``5572615``  ->  ``crop_original_video.mp4``

Measured on 2026-09-26: 200/200 rows join, all four signers, no exceptions. That
single fact is what makes M1/M3/M4 possible at all, so the join is asserted in
``tests/test_emosign.py`` and re-verified on every load rather than trusted.

Two further facts about the data that the plan depends on:

* Signers are **Cory 87 / Jonathan 54 / Rachel 52 / Ben 7**. Rachel appears under
  two session years, so grouping by the raw name prefix would leak her across
  LOSO folds. Signer identity is normalised to 4 people.
* The "Single Expression Set of 140 clips" that prior work evaluates on is **not
  reconstructible from the released CSV** (five subset rules tested, best match
  106). We therefore define and name our own subsets and always report the full
  200 alongside.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from seam.data.sources import get as get_resource

#: The ten EmoSign emotion-intensity columns, in the CSV's own order.
EMOTION_COLUMNS: tuple[str, ...] = (
    "joy",
    "excited",
    "surprise_pos",
    "surprise_neg",
    "worry",
    "sadness",
    "fear",
    "disgust",
    "frustration",
    "anger",
)

#: Intensity scale. 1 means "not present"; 4 is the strongest label.
INTENSITY_MIN = 1
INTENSITY_MAX = 4

#: Signers, normalised. Two naming forms exist in the release: the common
#: ``<Signer>_<year>-<mm>-<dd>_sc<session>_<id>`` and a minority
#: ``<n>-<Signer>-<topic>_<id>`` form. The key is matched anywhere in the name so
#: both resolve, and Rachel's two session years collapse to one person.
SIGNER_PATTERN = re.compile(r"(Cory|Jonathan|Rachel|Ben)")

#: The trailing numeric token is the ASLLRP utterance ID.
_UTTERANCE_ID = re.compile(r"(\d+)\s*$")


class EmoSignError(RuntimeError):
    """Raised when the label file is missing, malformed, or fails the join."""


@dataclass(frozen=True, slots=True)
class EmoSignClip:
    """One annotated utterance."""

    video_name: str
    utterance_id: str
    signer: str
    #: sentiment in -3..3
    sentiment: int
    #: emotion name -> intensity in 1..4
    intensities: dict[str, int]
    #: the three free-text Deaf-annotator cue descriptions
    cues: tuple[str, ...]
    #: True when this row's video is expected on disk
    video_expected: bool = True

    @property
    def dominant_emotion(self) -> str | None:
        """The single highest-intensity emotion, or ``None`` if there is no clear one.

        ``None`` when every emotion sits at 1 (nothing present) or when the maximum
        is tied. Both are genuine "no single expression" cases and are more
        honest than breaking the tie arbitrarily.
        """
        active = {k: v for k, v in self.intensities.items() if v > INTENSITY_MIN}
        if not active:
            return None
        top = max(active.values())
        winners = [k for k, v in active.items() if v == top]
        return winners[0] if len(winners) == 1 else None

    @property
    def n_active_emotions(self) -> int:
        return sum(1 for v in self.intensities.values() if v > INTENSITY_MIN)

    @property
    def sentiment3(self) -> str:
        """Coarse sentiment: ``negative`` / ``neutral`` / ``positive``."""
        if self.sentiment < 0:
            return "negative"
        if self.sentiment > 0:
            return "positive"
        return "neutral"

    @property
    def valence(self) -> float:
        """Signed sentiment normalised to [-1, 1]. A regression target, not a class."""
        return self.sentiment / 3.0


@dataclass(slots=True)
class EmoSign:
    """The full 200-clip label set."""

    clips: list[EmoSignClip] = field(default_factory=list)
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.clips)

    def __iter__(self) -> Iterator[EmoSignClip]:
        return iter(self.clips)

    def __getitem__(self, i: int) -> EmoSignClip:
        return self.clips[i]

    # -- selection ---------------------------------------------------------
    def by_signer(self, signer: str) -> list[EmoSignClip]:
        return [c for c in self.clips if c.signer == signer]

    def single_expression(self) -> list[EmoSignClip]:
        """Clips with exactly one emotion above baseline and no tie.

        Our own definition, named here so the paper can disclose it. It is *not*
        the 140-clip "Single Expression Set" of prior work, which we could not
        reconstruct from the released CSV.
        """
        return [c for c in self.clips if c.n_active_emotions == 1 and c.dominant_emotion]

    def with_video(self) -> list[EmoSignClip]:
        return [c for c in self.clips if c.video_expected]

    # -- derived views -----------------------------------------------------
    @property
    def signers(self) -> tuple[str, ...]:
        return tuple(sorted({c.signer for c in self.clips}))

    @property
    def signer_counts(self) -> dict[str, int]:
        counts: dict[str, int] = dict.fromkeys(self.signers, 0)
        for c in self.clips:
            counts[c.signer] += 1
        return counts

    @property
    def sentiment_distribution(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for c in self.clips:
            counts[c.sentiment] = counts.get(c.sentiment, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def emotion_distribution(self) -> dict[str, int]:
        counts = dict.fromkeys(EMOTION_COLUMNS, 0)
        for c in self.clips:
            for name, value in c.intensities.items():
                if value > INTENSITY_MIN:
                    counts[name] += 1
        return counts

    def utterance_ids(self) -> list[str]:
        return [c.utterance_id for c in self.clips]

    def describe(self) -> str:
        lines = [
            f"EmoSign: {len(self.clips)} clips from {self.source}",
            f"  signers: {', '.join(f'{s} {self.signer_counts[s]}' for s in self.signers)}",
            f"  sentiment: {self.sentiment_distribution}",
            "  emotion intensities > 1:",
            *(f"    {k:<14} {v}" for k, v in self.emotion_distribution.items()),
            f"  single-expression clips (our definition): {len(self.single_expression())}",
            f"  clips with any free-text cue: {sum(1 for c in self.clips if any(c.cues))}",
        ]
        return "\n".join(lines)


def labels_path(data_root: Path) -> Path:
    return get_resource("emosign_labels").local_path(data_root)


def load(data_root: Path, *, require_join: bool = True) -> EmoSign:
    """Load the 200-row label CSV and validate the ASLLRP join.

    ``require_join`` asserts that every row yields a numeric utterance ID. A row
    that does not is dropped loudly rather than silently: losing one clip is a
    finding, losing 200 is a bug, and a silent drop hides both.
    """
    path = labels_path(data_root)
    if not path.is_file():
        raise EmoSignError(
            f"EmoSign labels not found at {path}. Run `seam data fetch --all` first."
        )

    clips: list[EmoSignClip] = []
    unmapped: list[str] = []

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = {"video_name", "Sentiment"} - set(reader.fieldnames or ())
        if missing:
            raise EmoSignError(f"{path} is missing required columns: {sorted(missing)}")

        for row in reader:
            name = (row.get("video_name") or "").strip()
            if not name:
                continue

            uid_match = _UTTERANCE_ID.search(name)
            if uid_match is None:
                unmapped.append(name)
                continue

            signer_match = SIGNER_PATTERN.search(name)
            if signer_match is None:
                unmapped.append(name)
                continue

            intensities: dict[str, int] = {}
            for col in EMOTION_COLUMNS:
                raw = (row.get(col) or "").strip()
                try:
                    value = int(raw) if raw else INTENSITY_MIN
                except ValueError as exc:
                    raise EmoSignError(
                        f"{name}: emotion column {col!r} is {raw!r}, expected an integer"
                    ) from exc
                if not INTENSITY_MIN <= value <= INTENSITY_MAX:
                    raise EmoSignError(
                        f"{name}: emotion {col}={value} outside the "
                        f"{INTENSITY_MIN}..{INTENSITY_MAX} intensity scale"
                    )
                intensities[col] = value

            try:
                sentiment = int((row.get("Sentiment") or "0").strip())
            except ValueError as exc:
                raise EmoSignError(f"{name}: Sentiment is not an integer") from exc

            cues = tuple((row.get(f"Reasoning_{i}") or "").strip() for i in (1, 2, 3))

            clips.append(
                EmoSignClip(
                    video_name=name,
                    utterance_id=uid_match.group(1),
                    signer=signer_match.group(1),
                    sentiment=sentiment,
                    intensities=intensities,
                    cues=cues,
                )
            )

    if unmapped and require_join:
        raise EmoSignError(
            f"{len(unmapped)} of {len(clips) + len(unmapped)} EmoSign rows failed the "
            f"ASLLRP utterance-ID join. First few: {unmapped[:5]}. The join key is the "
            f"trailing numeric token of video_name; if the upstream naming changed, this "
            f"must be fixed here rather than worked around."
        )

    if not clips:
        raise EmoSignError(f"{path} contained no usable rows")

    return EmoSign(clips=clips, source=path)
