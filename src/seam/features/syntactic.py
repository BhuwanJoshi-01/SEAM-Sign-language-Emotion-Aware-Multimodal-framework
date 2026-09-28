"""Linguistic-marker supervision over real ASLLRP annotation (M3 Track A).

This module exists to supply the *syntactic* half of the marker set. The visual
half — brow raise, head shake, mouth morphemes — is already produced by
:mod:`seam.features.markers` from blendshapes and head pose. Those are
pseudo-labels: rules over facial motion, with no annotation behind them.

The syntactic half is in a different epistemic position, and the distinction is
the reason this module tracks provenance at all:

* **The input is human annotation.** The ASLLRP utterance mapping is a
  hand-written gloss of every utterance, produced by annotators. 200/200 EmoSign
  utterances join to it, so every clip this project reasons about has real
  linguistic context.
* **The rule is still a rule.** Deciding that an utterance is interrogative
  because it contains ``WHO`` is a lexical inference over that annotation, not an
  annotation of the non-manual signal. Nothing here observes a raised brow. So
  these labels are *lexically grounded* but still not gold, and they are not the
  SignStream non-manual XML that Track B is waiting on.
* **Topicalization is inferred, not read.** ASLLRP's gloss is a linear sign
  sequence with no constituent bracketing, so topic-fronting cannot be read off
  it directly. The heuristic here is weak and is labelled ``pseudo`` for that
  reason, with a confidence that reflects the weakness rather than hiding it.

Every label therefore carries a ``provenance`` string, and every table downstream
must print it. A marker without a provenance is a marker whose reliability is
unknown, which is worse than a marker known to be weak.

A note on the ``fs-`` prefix, which cost an assumption to get right: it marks
*compound* sign tokens (``fs-BEACH``, ``fs-LATE``, ``fs-OF``), not facial or
non-manual behaviour. Reading it as "facial signal" would have been a plausible
and completely wrong shortcut, and it would have put 98 token types into the
non-manual vocabulary for no reason. The lexicon below was read out of the data
rather than assumed.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from seam.logging import get
from seam.paths import default_data_root

log = get(__name__)

#: Where the human-authored glosses live.
GLOSS_MAP_REL = Path("asllrp") / "ASLLRP_utterances_mapping.txt"

#: Provenance strings. These are the only values a label may carry, and they mean
#: different things about how much the label can be trusted.
PROV_LEXICAL = "lexical-rule-over-annotated-gloss"
#: Inferred from sign order alone, with no constituent structure available.
PROV_PSEUDO = "heuristic (pseudo); no constituent structure in the gloss"

#: WH words that make an utterance interrogative. Present in the corpus: WHAT is
#: written quoted (``"WHAT"``) to mark it as an overt question word rather than a
#: contentful sign, so both spellings are matched.
WH_WORDS = frozenset(
    {"WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW", "WHICH", '"WHAT"', '"WHO"', '"WHERE"'}
)

#: Lexical negation. ``NO`` is absent from the corpus and deliberately kept: a
#: lexicon that only contains what this corpus happens to contain will silently
#: miss negation the moment the corpus changes.
NEGATIONS = frozenset(
    {
        "NOT",
        "NEVER",
        "NONE",
        "NOTHING",
        "NOBODY",
        "NOWHERE",
        "NEGATIVE",
        "CANNOT",
        "NO",
        "CAN'T",
        "DOESN'T",
        "DON'T",
        "ISN'T",
        "WON'T",
        "AIN'T",
        "WITHOUT",
    }
)

#: Indexical pointing. Not a linguistic marker in the interrogative/negation
#: sense, but it is a *reference-establishment* signal, and it is the single most
#: common non-lexical token in the corpus (1,313 occurrences). Recorded because a
#: topic hold is usually introduced by one.
REFERENCE_TOKENS = frozenset({"IX", "IX-1", "IX-2", "IX-a", "IX-b"})

#: Quoted tokens in the gloss are annotator asides, e.g. ``"perplexed"``,
#: ``"pause"``, ``"you"``. They are removed before lexical matching so an aside
#: mentioning "how" cannot make an utterance interrogative.
_ASIDE = re.compile(r'"[^"]*"')

#: A single fully-quoted token, for the exception below.
_QUOTED = re.compile(r'"([^"]*)"')

#: The bare WH words, used when a quoted form is recognised as an overt question.
WH_WORDS_BARE = frozenset({"WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW", "WHICH"})


def gloss_map_path(root: Path | None = None) -> Path:
    return (root or default_data_root()) / GLOSS_MAP_REL


def load_gloss_map(root: Path | None = None) -> dict[str, list[str]]:
    """Parse the ASLLRP utterance mapping into ``{utterance_id: [tokens]}``.

    The format is one utterance per line, ``<id>: TOKEN TOKEN ...``. Quoted
    material is kept here — it carries annotator asides that are worth having —
    and stripped later, at match time.
    """
    path = gloss_map_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `seam data fetch asllrp_utterance_map`. "
            "Without it the syntactic track has no linguistic input at all."
        )
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        tokens = value.split()
        if key.strip() and tokens:
            out[key.strip()] = tokens
    log.info("loaded %d ASLLRP gloss sequences from %s", len(out), path.name)
    return out


@dataclass(slots=True)
class SyntaxLabel:
    """One syntactic marker, with the evidence and the reliability."""

    name: str
    present: bool
    provenance: str
    #: Where the decision came from, so a label can be audited.
    evidence: tuple[str, ...] = ()
    #: 0-1. Topicalization is deliberately low-confidence; see the module docstring.
    confidence: float = 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "present": self.present,
            "provenance": self.provenance,
            "evidence": list(self.evidence),
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class UtteranceSyntax:
    """The syntactic marker set for one utterance."""

    utterance_id: str
    n_tokens: int = 0
    labels: dict[str, SyntaxLabel] = field(default_factory=dict)
    #: The gloss as read, for the audit trail.
    gloss: str = ""

    def __contains__(self, name: object) -> bool:
        lab = self.labels.get(str(name))
        return bool(lab and lab.present)

    def get(self, name: str) -> SyntaxLabel | None:
        return self.labels.get(name)

    def present(self) -> list[str]:
        return sorted(n for n, lab in self.labels.items() if lab.present)

    def provenance(self) -> dict[str, str]:
        """Provenance per marker, for printing in every downstream table."""
        return {n: lab.provenance for n, lab in self.labels.items()}

    def as_dict(self) -> dict[str, object]:
        return {
            "utterance_id": self.utterance_id,
            "n_tokens": self.n_tokens,
            "gloss": self.gloss,
            "present": self.present(),
            "labels": {n: lab.as_dict() for n, lab in self.labels.items()},
        }


def _content_tokens(tokens: Sequence[str]) -> list[str]:
    """Tokens with annotator asides removed, uppercased, compound prefixes kept.

    ``fs-`` is *not* stripped: it marks compound signs, and those compounds are
    legitimate content words. Stripping the prefix would merge distinct signs.

    One quoted form is *kept*: a quoted WH word. The corpus writes ``"WHAT"`` (34
    occurrences) to mark an overt question word as opposed to a contentful sign
    of the word "what", so those quotes are annotation rather than commentary.
    Stripping them as asides deleted the clearest interrogative evidence the
    corpus provides, and a test caught it doing so.
    """
    out: list[str] = []
    for tok in tokens:
        quoted = _QUOTED.fullmatch(tok.strip())
        if quoted and quoted.group(1).upper() in WH_WORDS_BARE:
            out.append(quoted.group(1).upper())
            continue
        for piece in _ASIDE.sub(" ", tok).split():
            out.append(piece.upper())
    return out


def _interrogative(tokens: Sequence[str]) -> SyntaxLabel:
    hits = tuple(t for t in tokens if t in WH_WORDS)
    return SyntaxLabel(
        name="interrogative",
        present=bool(hits),
        provenance=PROV_LEXICAL,
        evidence=hits,
        confidence=0.9 if hits else 1.0,
    )


def _negation(tokens: Sequence[str]) -> SyntaxLabel:
    hits = tuple(t for t in tokens if t in NEGATIONS)
    return SyntaxLabel(
        name="negation",
        present=bool(hits),
        provenance=PROV_LEXICAL,
        evidence=hits,
        confidence=0.9 if hits else 1.0,
    )


def _topicalization(tokens: Sequence[str]) -> SyntaxLabel:
    """Weak inference, and labelled as such.

    ASLLRP's gloss is a flat token sequence: there are no constituent brackets, so
    "this noun phrase was moved to the front" is not something the annotation
    states. What *is* observable is a reference being established by an indexical
    near the front of the utterance, which is how a topic is normally
    introduced. That is a correlate of topicalization, not the thing itself.

    Confidence is capped at 0.5 for exactly that reason. A downstream table that
    prints a topicalization count without the confidence is over-claiming, which
    is why :meth:`UtteranceSyntax.provenance` exists to be printed next to it.
    """
    window = tokens[:6]
    ix = tuple(t for t in window if t in REFERENCE_TOKENS)
    return SyntaxLabel(
        name="topicalization",
        present=bool(ix),
        provenance=PROV_PSEUDO,
        evidence=ix,
        confidence=0.5 if ix else 0.3,
    )


def _reference_establishment(tokens: Sequence[str]) -> SyntaxLabel:
    """Indexical reference anywhere in the utterance.

    Separated from topicalization because it is directly observed in the
    annotation rather than inferred, and it is frequent enough (1,313 corpus
    occurrences) to be worth its own column.
    """
    hits = tuple(t for t in tokens if t in REFERENCE_TOKENS)
    return SyntaxLabel(
        name="reference_establishment",
        present=bool(hits),
        provenance=PROV_LEXICAL,
        evidence=hits[:8],
        confidence=1.0,
    )


#: Markers produced by this module, in report order.
SYNTACTIC_MARKERS = (
    "interrogative",
    "negation",
    "topicalization",
    "reference_establishment",
)


def classify_tokens(tokens: Sequence[str], *, utterance_id: str = "") -> UtteranceSyntax:
    """Classify one token sequence into the syntactic marker set."""
    content = _content_tokens(tokens)
    out = UtteranceSyntax(
        utterance_id=utterance_id,
        n_tokens=len(content),
        gloss=" ".join(tokens)[:400],
    )
    out.labels = {
        "interrogative": _interrogative(content),
        "negation": _negation(content),
        "topicalization": _topicalization(content),
        "reference_establishment": _reference_establishment(content),
    }
    return out


def classify(utterance_id: str, gloss_map: dict[str, list[str]] | None = None) -> UtteranceSyntax:
    """Classify one utterance by id, loading the map if needed."""
    gm = gloss_map if gloss_map is not None else load_gloss_map()
    tokens = gm.get(str(utterance_id))
    if tokens is None:
        return UtteranceSyntax(
            utterance_id=str(utterance_id),
            labels={
                name: SyntaxLabel(name, False, PROV_PSEUDO, (), 0.0) for name in SYNTACTIC_MARKERS
            },
        )
    return classify_tokens(tokens, utterance_id=str(utterance_id))


def describe() -> dict[str, object]:
    """The module's own provenance, for the methods section of the paper."""
    return {
        "inputs": "ASLLRP utterance glosses (human annotation)",
        "markers": list(SYNTACTIC_MARKERS),
        "provenance": {
            "interrogative": PROV_LEXICAL,
            "negation": PROV_LEXICAL,
            "reference_establishment": PROV_LEXICAL,
            "topicalization": PROV_PSEUDO,
        },
        "not_yet": (
            "ASLLRP SignStream non-manual XML (Track B, needs BU access). Until then "
            "no label here is an annotation of a non-manual signal; they are lexical "
            "facts about the utterance that a non-manual signal might be expected to "
            "co-occur with."
        ),
        "fs_prefix_meaning": "compound sign token (fs-BEACH, fs-LATE), NOT facial signal",
    }
