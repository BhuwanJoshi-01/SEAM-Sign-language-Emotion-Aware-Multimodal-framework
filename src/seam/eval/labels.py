"""Agreement between two annotators of the same items, and the audit built on it.

**Why kappa and not accuracy.** These labels are heavily imbalanced — human annotation
marks 31 of 200 clips as interrogative — so a classifier that answers "no" to everything
scores 85% accuracy on the interrogative comparison and kappa 0. Reporting accuracy would
make three chance-level heuristics look like two good ones and one great one. This
module's whole purpose is to make "that label is no better than a coin" a number that
cannot be mistaken for "that label works".

`cohen_kappa` lives here rather than in `scripts/check_label_quality.py` for the same
reason `wer` lives in `seam.eval.recogniser`: a metric in a CLI script is a metric nobody
tests, and this one now decides a published claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray


def cohen_kappa(human: NDArray[np.bool_], other: NDArray[np.bool_]) -> float:
    """Cohen's kappa for two binary raters over the same items.

    Returns NaN when chance agreement is 1.0 (both raters constant and identical), which
    is undefined rather than perfect.

    Raises on a length mismatch: comparing misaligned items yields a number that looks
    exactly like a measurement, and that is precisely the kind of quiet failure this
    project keeps having to undo.
    """
    if len(human) != len(other):
        raise ValueError(
            f"label vectors differ in length: {len(human)} vs {len(other)}. "
            "They must describe the same items in the same order."
        )
    po = float(np.mean(np.asarray(human) == np.asarray(other)))
    hp, pp = float(np.mean(human)), float(np.mean(other))
    pe = hp * pp + (1 - hp) * (1 - pp)
    return float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)


def confusable_rater(a: NDArray[np.bool_], b: NDArray[np.bool_], c: NDArray[np.bool_]) -> float:
    """Fleiss' kappa for k raters over binary labels, via the confusion-count form.

    Used by the M7 preference study, where the rater count is small and the decisions are
    forced choices, so chance agreement is exactly 1/k per item.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    # Three raters by construction; the chance-agreement expression below assumes a
    # binary task with three raters, so a different rater count needs a different formula
    # rather than a changed constant.
    agree = [int(np.sum(a)), int(np.sum(b)), int(np.sum(c))]
    p_j = np.array(agree, dtype=float) / n
    p_bar = float(np.mean(p_j))
    pe = p_bar**2 + (1 - p_bar) ** 2
    po = float(np.mean((agree[0] + agree[1] + agree[2]) / n))
    return float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)


@dataclass
class Agreement:
    """One heuristic-vs-human comparison over the same clips."""

    key: str
    display: str
    heuristic_label: str
    human_markers: tuple[str, ...]
    n_human_positive: int
    n_heuristic_positive: int
    accuracy: float
    kappa: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def verdict(self) -> str:
        """useless < 0.1, weak < 0.4, usable above. Thresholds are stated, not implied."""
        if np.isnan(self.kappa):
            return "undefined"
        if self.kappa < 0.1:
            return "useless"
        if self.kappa < 0.4:
            return "weak"
        return "usable"

    def as_dict(self) -> dict[str, object]:
        return {
            # `key` is stable and unique. `heuristic_label` is NOT unique: two comparisons
            # share the heuristic `interrogative`, and a consumer keying on that alone
            # silently reads the rhetorical-question row (kappa 0.73) instead of the
            # wh/yes-no row (kappa 0.03) - the exact substitution that would hide the
            # finding.
            "key": self.key,
            "display": self.display,
            "heuristic_label": self.heuristic_label,
            "human_markers": list(self.human_markers),
            "n_human_positive": self.n_human_positive,
            "n_heuristic_positive": self.n_heuristic_positive,
            "accuracy": round(self.accuracy, 4),
            "kappa": None if np.isnan(self.kappa) else round(self.kappa, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "verdict": self.verdict,
        }


def compare(human: NDArray[np.bool_], other: NDArray[np.bool_]) -> dict[str, int]:
    """Confusion counts for one pair of binary label vectors."""
    h = np.asarray(human, dtype=bool)
    o = np.asarray(other, dtype=bool)
    return {
        "tp": int(np.sum(h & o)),
        "fp": int(np.sum(h & ~o)),
        "fn": int(np.sum(~h & o)),
        "tn": int(np.sum(~h & ~o)),
    }


def prevalence(rows: list[dict[str, bool]], prefix: str = "h_") -> dict[str, int]:
    """Count how many clips carry each human marker."""
    if not rows:
        return {}
    keys = [k for k in rows[0] if k.startswith(prefix)]
    return {k: int(sum(1 for r in rows if r[k])) for k in keys}
