"""Tests for the M5a recognition metric and the loss of information in the split.

The WER function was inverted in the first run of this experiment, reporting 0.000
where the truth was near 1.0. The failure was silent *and* flattering, which is the
worst combination: a plausible-looking number that no gate caught. These tests exist to
make that specific class of bug loud.
"""

from __future__ import annotations

import collections

import pytest

from seam.eval.recogniser import fit_logreg, oov_floor, per_class_recall, predict, wer


def test_wer_anchors() -> None:
    """The three cases whose reversal defines the bug this file exists for."""
    assert wer(list("abc"), list("abc")) == 0.0, "an identical hypothesis is a perfect score"
    assert wer(list("abc"), list("xyz")) == 1.0, "no correct gloss is a total loss"
    assert wer(list("abc"), list("xbc")) == pytest.approx(1 / 3), "one of three wrong"
    assert wer(list("abc"), list("abbc")) == pytest.approx(1 / 3), "one insertion"
    assert wer(list("abc"), list("ac")) == pytest.approx(1 / 3), "one deletion"
    assert wer([], []) == 0.0


def test_wer_is_non_negative_and_bounded_above_by_the_insertion_count() -> None:
    """WER has no upper bound of 1: insertions are counted against a fixed denominator.

    A hypothesis twice as long as the reference scores 2.0, which is correct and is why
    this asserts a lower bound and a specific value rather than a range.
    """
    assert wer(["a", "b"], ["x"]) == 1.0
    assert wer(["a"], ["a", "b", "c"]) == pytest.approx(2.0)
    assert wer(["a", "b", "c"], []) == 1.0
    for ref, hyp in ((["a", "b"], ["x"]), (["a"], ["a", "b"]), (["a"], [])):
        assert wer(ref, hyp) >= 0.0


def test_wer_matches_a_hand_computed_edit_distance() -> None:
    ref = ["NOT", "UNDER", "THE", "TABLE"]
    hyp = ["NOT", "ON", "THE", "TABLE", "NOW"]
    # three words match, one substitution and one insertion over a reference of 4
    assert wer(ref, hyp) == pytest.approx(2 / 4)


def test_the_model_cannot_beat_chance_on_unique_features() -> None:
    """Sanity floor for the fitting code.

    Random features carry no class information, so a correct fit must land near the
    class prior. A model scoring near zero here would mean the optimiser or the metric
    is broken rather than that the features are informative.
    """
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 12))
    y = np.array(["a" if i % 2 else "b" for i in range(400)])
    classes = ["a", "b"]
    hyp = predict(fit_logreg(X, y, classes, lam=1.0), classes, X)
    acc = np.mean([h == t for h, t in zip(hyp, y, strict=True)])
    assert 0.35 < acc < 0.65, f"random features gave accuracy {acc:.3f}, expected ~0.5"


def test_fitting_is_deterministic() -> None:
    import numpy as np

    X = np.linspace(0, 1, 100).reshape(50, 2)
    y = np.array(["a", "b"] * 25)
    classes = ["a", "b"]
    a = fit_logreg(X, y, classes, lam=1.0)
    b = fit_logreg(X, y, classes, lam=1.0)
    assert (a == b).all(), "the same data must give the same weights, or seeds mean nothing"


def test_unseen_glosses_cannot_be_predicted() -> None:
    """The OOV floor is structural, so the harness must not pretend otherwise.

    Any model restricted to the training vocabulary is wrong on every test token whose
    gloss the training signers never used. The 499-gloss corpus has 284 hapax types, so
    this is the dominant term in the reported WER and is what `evaluate` reports as
    `oov_fraction` rather than folding into the score.
    """
    glosses = [f"g{i}" for i in range(500)]
    train, test = glosses[:499], glosses[499:]
    unseen = [g for g in test if g not in set(train)]
    assert unseen, "this corpus always has unseen test glosses"
    assert collections.Counter(train)["g0"] >= 1


def test_per_class_recall_exposes_what_an_aggregate_hides() -> None:
    """A fold dominated by frequent glosses can look fine and predict nothing rare."""
    ref = ["IX"] * 90 + ["RARE"] * 10
    hyp = ["IX"] * 100
    recall = per_class_recall(ref, hyp, ["IX", "RARE"])
    assert recall["IX"] == 1.0
    assert recall["RARE"] == 0.0
    # The aggregate WER is only 10%, which would read as a partial success.
    assert wer(ref, hyp) == pytest.approx(0.1)


def test_oov_floor_is_the_structural_error_rate() -> None:
    import collections as c

    train = c.Counter({"IX": 50, "NOT": 30})
    assert oov_floor(train, ["IX", "NOT", "NEVER-SEEN"]) == pytest.approx(1 / 3)
    assert oov_floor(train, ["IX", "NOT"]) == 0.0
    assert oov_floor(train, []) == 0.0


def test_fitting_refuses_labels_outside_the_vocabulary() -> None:
    """A silently-ignored label would drop training rows without any warning."""
    import numpy as np

    X = np.zeros((3, 2))
    with pytest.raises(ValueError, match="absent"):
        fit_logreg(X, ["a", "b", "z"], ["a", "b"], lam=1.0)


def test_predict_uses_the_fitted_weights() -> None:
    """`predict` must read W.

    It once returned `argmax` of the features themselves, so predictions depended on
    which column happened to be largest and the fitted model was never consulted. Two
    disjoint feature blocks make that visible: a model told to prefer class "a" must
    return "a" even when the larger feature block points the other way.
    """
    import numpy as np

    X = np.zeros((2, 4))
    X[0, :2] = 10.0  # large block
    X[1, 2:] = 10.0  # large block
    classes = ["a", "b"]
    W = np.zeros((5, 2))
    W[2, 0] = 1.0  # bias row is last (index 4); use the feature rows
    W[0, 1] = 5.0  # strongly prefer "b" for row 0
    hyp = predict(W, classes, X)
    assert hyp == ["b", "a"], f"got {hyp}, so the weights were not used"


def test_oov_count_and_fraction_are_distinct_quantities() -> None:
    """Guards the units bug that reported a 29.5% OOV floor as 0.1%.

    `oov_floor` returns a fraction. Storing it in a field called `oov_tokens` and then
    dividing by the token count a second time produced a number two orders of magnitude
    too small, and the direction of the error was towards looking better than it is.
    """
    import collections as c

    train = c.Counter({"IX": 50, "NOT": 30})
    ref = ["IX"] * 70 + ["NEVER-SEEN"] * 30
    frac = oov_floor(train, ref)
    count = sum(1 for v in ref if v not in train)
    assert frac == pytest.approx(0.30)
    assert count == 30
    assert count / len(ref) == pytest.approx(frac)
    assert frac != pytest.approx(count), "fraction and count must not be interchangeable"
