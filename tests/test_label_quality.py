"""Tests for the heuristic-vs-human label comparison, and the M4 label swap.

The measurement these support is the one that decided M4's fate: three of the four
linguistic labels feeding the factorised encoder agree with human annotation at or near
chance, and swapping in human labels does **not** rescue the gate. Both halves are
load-bearing, so both are pinned here rather than left to the artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
AGREEMENT = REPO / "artifacts" / "m3" / "label_agreement.json"
HUMAN_RUN = REPO / "artifacts" / "m4" / "factorizer_human_labels.json"
HEURISTIC_RUN = REPO / "artifacts" / "m4" / "factorizer_multilabel.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        pytest.skip(f"{path.name} not built yet")
    return json.loads(path.read_text())


# --- kappa ------------------------------------------------------------------


def test_kappa_matches_a_hand_computed_case() -> None:
    """Perfect agreement on a balanced 2x2 table is 1.0."""
    from seam.eval.labels import cohen_kappa

    h = np.array([True, True, False, False])
    p = np.array([True, True, False, False])
    assert cohen_kappa(h, p) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_not_accuracy_one() -> None:
    """The whole reason kappa is used instead of accuracy.

    A classifier that says "no" for everything scores 75% accuracy on a quarter-positive
    problem and has kappa 0. Reporting accuracy here would make three broken labels look
    like one great one and two good ones.
    """
    from seam.eval.labels import cohen_kappa

    h = np.array([True, False, False, False])
    p = np.zeros(4, dtype=bool)
    assert float(np.mean(h == p)) == pytest.approx(0.75), "accuracy looks great"
    assert cohen_kappa(h, p) == pytest.approx(0.0), "kappa does not"


def test_kappa_is_negative_for_systematic_disagreement() -> None:
    """kappa is not clamped at 0; systematic error is worse than chance and shows it."""
    from seam.eval.labels import cohen_kappa

    h = np.array([True, True, False, False])
    p = np.array([False, False, True, True])
    assert cohen_kappa(h, p) < 0


def test_kappa_rejects_mismatched_lengths() -> None:
    """Comparing misaligned clips produces a number that looks like a measurement."""
    from seam.eval.labels import cohen_kappa

    with pytest.raises(ValueError, match="differ in length"):
        cohen_kappa(np.array([True, False, True]), np.array([True, False]))


# --- the measurement --------------------------------------------------------


def test_agreement_report_covers_all_four_m4_labels() -> None:
    """y_L is four slots; the report has to answer for all four."""
    d = _load(AGREEMENT)
    assert d["n_clips"] == 200, "all 200 EmoSign clips joined to human annotations"
    labels = {c["heuristic_label"] for c in d["comparisons"]}
    assert {
        "negation",
        "interrogative",
        "topicalization",
        "reference_establishment",
    } <= labels


def test_comparison_keys_are_unique() -> None:
    """Two comparisons share the heuristic label `interrogative`.

    A consumer keying on `heuristic_label` reads the rhetorical-question row (kappa 0.73)
    instead of the wh/yes-no row (kappa 0.03) - and would conclude the interrogative
    heuristic is the best of the four, when it is the worst. That is the exact
    substitution that would hide the finding, so the artifact carries a stable `key` and
    it has to be unique.
    """
    d = _load(AGREEMENT)
    keys = [c["key"] for c in d["comparisons"]]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
    heur = [c["heuristic_label"] for c in d["comparisons"]]
    assert len(heur) > len(set(heur)), (
        "expected the two interrogative comparisons to share a heuristic label; if the "
        "corpus ever distinguishes them, update this test rather than trusting it"
    )


def test_three_of_the_four_heuristic_labels_are_at_or_near_chance() -> None:
    """The finding that motivated re-running M4 with human labels.

    Pinned because it is the load-bearing claim of the whole negative result: if these
    labels were fine, the leakage would have been unexplained.
    """
    d = _load(AGREEMENT)
    by_key = {c["key"]: c["kappa"] for c in d["comparisons"]}
    assert by_key["interrogative_wh_yesno"] is not None
    assert by_key["interrogative_wh_yesno"] < 0.1, "wh/yes-no interrogative is at chance"
    assert by_key["conditional"] is not None and by_key["conditional"] < 0.1
    assert by_key["topic"] is not None and by_key["topic"] < 0.4
    # Two that are genuinely usable, so the report is not just uniformly damning - and so a
    # future edit that breaks the mapping shows up as a global collapse rather than a
    # silent improvement.
    assert by_key["negation"] is not None and by_key["negation"] > 0.5
    assert by_key["interrogative_rhetorical"] is not None
    assert by_key["interrogative_rhetorical"] > 0.5


def test_reference_establishment_over_fires() -> None:
    """93 heuristic positives against 37 human ones is a 2.5x false-positive rate."""
    d = _load(AGREEMENT)
    rec = next(c for c in d["comparisons"] if c["key"] == "conditional")
    assert rec["n_heuristic_positive"] > 2 * rec["n_human_positive"]


def test_the_report_states_its_own_caveat() -> None:
    d = _load(AGREEMENT)
    assert "kappa" in d["interpretation"].lower()


# --- the M4 label swap ------------------------------------------------------


def test_the_human_label_run_exists_and_is_a_separate_artifact() -> None:
    """Overwriting the baseline would destroy the comparison that justified the decision."""
    assert HUMAN_RUN.is_file(), "the human-label arm was not run"
    assert HEURISTIC_RUN.is_file()
    assert HUMAN_RUN != HEURISTIC_RUN


def test_both_arms_used_the_same_gate_target() -> None:
    """A comparison is only meaningful if the bar did not move."""
    h = _load(HUMAN_RUN)["runs"]["full"]["gate"]
    b = _load(HEURISTIC_RUN)["runs"]["full"]["gate"]
    assert h["separation_target"] == b["separation_target"] == 0.6
    assert h["control_min"] == b["control_min"]


def test_both_arms_pass_the_signer_control() -> None:
    """The negative result is only worth anything if the instrument still works."""
    for path in (HEURISTIC_RUN, HUMAN_RUN):
        g = _load(path)["runs"]["full"]["gate"]
        assert g["control_passed"] is True, path.name
        assert g["signer_control_max"] >= 0.80, path.name


def test_human_labels_do_not_rescue_the_gate() -> None:
    """The result that decided M4.

    Pinned as a test so that a later run cannot quietly restate it. If a future change
    *does* rescue the gate - more data, a different encoder - this test fails, and that
    failure is the signal to re-open the M4 decision rather than to edit the number.
    """
    h = _load(HUMAN_RUN)["runs"]["full"]
    b = _load(HEURISTIC_RUN)["runs"]["full"]
    assert h["gate"]["gate_passed"] is False
    assert b["gate"]["gate_passed"] is False
    assert h["gate"]["worst_cross_auc"] > h["gate"]["separation_target"]
    # A marginal change, not a rescue.
    assert h["gate"]["worst_cross_auc"] > b["gate"]["worst_cross_auc"] * 0.95


def test_the_two_arms_used_the_same_folds() -> None:
    """Different folds would make the comparison meaningless."""
    h = {f["held_out"] for f in _load(HUMAN_RUN)["runs"]["full"]["per_fold"]}
    b = {f["held_out"] for f in _load(HEURISTIC_RUN)["runs"]["full"]["per_fold"]}
    assert h == b
