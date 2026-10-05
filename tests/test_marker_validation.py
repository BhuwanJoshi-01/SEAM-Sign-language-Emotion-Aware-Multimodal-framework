"""Validating the visual markers against human frame labels: the parts that can mislead.

The experiment is `scripts/validate_markers.py`. What is pinned here is the feature
builder (which must not read a label or another clip), the verdict rule, and the two
results the rest of the project now leans on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_markers as V

ARTIFACT = Path(__file__).resolve().parents[1] / "artifacts" / "m3" / "marker_validation.json"


def _rotations(yaw: np.ndarray) -> np.ndarray:
    rot = np.tile(np.eye(4), (len(yaw), 1, 1))
    c, s = np.cos(yaw), np.sin(yaw)
    rot[:, 0, 0], rot[:, 0, 1] = c, -s
    rot[:, 1, 0], rot[:, 1, 1] = s, c
    return rot


def test_rolling_std_is_centred_and_shrinks_at_the_edges() -> None:
    x = np.arange(10, dtype=float)[:, None]
    out = V.rolling_std(x, half=1)
    assert out.shape == x.shape
    assert out[5, 0] == pytest.approx(np.std([4.0, 5.0, 6.0]))
    assert out[0, 0] == pytest.approx(np.std([0.0, 1.0]))


def test_features_see_an_oscillation_that_no_single_frame_shows() -> None:
    """A head shake is yaw going back and forth; the rolling sd is what registers it."""
    t = np.arange(60) / 24.0
    still = V.frame_features(np.zeros((60, 52)), _rotations(np.zeros(60)), 24.0)
    shake = V.frame_features(np.zeros((60, 52)), _rotations(0.3 * np.sin(2 * np.pi * 3 * t)), 24.0)
    assert still.shape == shake.shape == (60, 61)
    wobble = slice(58, 61)
    assert np.abs(still[:, wobble]).max() < 1e-9
    assert shake[10:50, wobble].max() > 0.1


def test_features_are_centred_on_the_clip_and_nothing_else() -> None:
    """A signer's resting face must not be a feature; removing it reads no label."""
    rng = np.random.default_rng(0)
    bs = rng.uniform(0, 0.3, size=(40, 52))
    rot = _rotations(np.zeros(40))
    a = V.frame_features(bs, rot, 24.0)
    b = V.frame_features(bs + 0.2, rot, 24.0)  # same motion, different resting face
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_the_verdict_is_decided_by_the_worst_large_fold() -> None:
    assert V.verdict({"Cory": 0.84, "Jonathan": 0.82, "Rachel": 0.88, "Ben": 0.40}) == "validated"
    assert V.verdict({"Cory": 0.84, "Jonathan": 0.79, "Rachel": 0.88}) == "usable with caution"
    assert V.verdict({"Cory": 0.60, "Jonathan": 0.90, "Rachel": 0.90}) == "not validated"
    assert V.verdict({"Cory": 0.9, "Rachel": 0.9}).startswith("no verdict")


def test_the_gate_is_the_registered_one() -> None:
    assert (V.VALIDATED, V.CAUTION) == (0.80, 0.70)
    assert V.DECIDING_FOLDS == ("Cory", "Jonathan", "Rachel")
    assert V.MARKERS == ("brow_raise", "brow_furrow", "head_shake", "head_nod")


def _artifact() -> dict:
    if not ARTIFACT.is_file():
        pytest.skip("marker_validation.json not built; run scripts/validate_markers.py")
    return json.loads(ARTIFACT.read_text())


def test_brow_raise_is_the_one_marker_validated_against_human_frames() -> None:
    """The first instrument in this project checked against an annotator, and it holds.

    M3 called brow_raise "degenerate" because it fires on 86% of clips. The annotators
    mark a brow raise on 169 of 200 clips, so the clip-level rate was never the defect:
    frame by frame the three-blendshape mean follows the human track.
    """
    m = _artifact()["markers"]
    assert m["brow_raise"]["verdict_heuristic"] == "validated"
    assert (
        min(m["brow_raise"]["heuristic_within_clip_auc_by_fold"][f] for f in V.DECIDING_FOLDS)
        >= 0.80
    )
    for other in ("brow_furrow", "head_shake", "head_nod"):
        assert m[other]["verdict_heuristic"] == "not validated", other


def test_the_head_markers_are_unreadable_even_with_supervision() -> None:
    """Not a threshold problem: a fitted detector on head pose does not find them either."""
    m = _artifact()["markers"]
    for marker in ("head_shake", "head_nod"):
        assert m[marker]["verdict_supervised"] == "not validated"
        heur = m[marker]["heuristic_within_clip_auc_by_fold"]
        assert all(abs(heur[f] - 0.5) < 0.02 for f in V.DECIDING_FOLDS), (
            f"{marker}: the heuristic signal should be at chance, it is blind"
        )
