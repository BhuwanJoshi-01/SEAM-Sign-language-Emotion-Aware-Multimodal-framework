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


def test_the_offline_head_shake_marker_is_no_longer_at_chance() -> None:
    """0.50 was read as "head movement cannot be seen". It was the wrong axis.

    `markers.signals` read a tilt for head_shake until 2026-10-05 and scored exactly 0.50 on
    every fold. Corrected, the same row is above chance on each large fold. It stays below
    the page's level score, as it should: the offline marker is an on/off event with one
    hard threshold, which ranks frames more coarsely than the continuous level does.
    """
    shake = _artifact()["markers"]["head_shake"]
    offline = shake["heuristic_within_clip_auc_by_fold"]
    level = shake["revised_page_within_clip_auc_by_fold"]
    for fold in V.DECIDING_FOLDS:
        assert offline[fold] > 0.52, fold
        assert offline[fold] < level[fold], fold


def test_head_shake_is_readable_on_the_right_axis_but_does_not_pass_the_gate() -> None:
    """Above chance on every large fold; under 0.80 on every one. Both halves are the result."""
    shake = _artifact()["markers"]["head_shake"]
    revised = shake["revised_page_within_clip_auc_by_fold"]
    for fold in V.DECIDING_FOLDS:
        assert 0.65 < revised[fold] < V.VALIDATED, fold
    assert shake["verdict_revised_page"] == "not validated"
    assert shake["verdict_compact"] == "not validated"


def test_the_angle_columns_are_what_their_names_say() -> None:
    """Written to the artifact on every run, so the names cannot drift from the head again."""
    check = _artifact()["axis_check"]["median_abs_correlation"]
    rows = {"turn": None, "nod": None, "tilt": None}
    for motion, row in check.items():
        for name in rows:
            if f"a {name})" in motion:
                rows[name] = row
    assert all(rows.values()), rows
    assert rows["turn"]["markers.yaw (read as head_shake)"] > 0.8
    assert rows["nod"]["markers.pitch"] > 0.75
    assert rows["tilt"]["markers.roll"] > 0.9
    for name, row in rows.items():
        assert row[f"head_motion.{name}"] == max(
            row[k] for k in row if k.startswith("head_motion.")
        ), name


def test_head_nod_stays_weak_on_the_right_axis() -> None:
    nod = _artifact()["markers"]["head_nod"]
    revised = nod["revised_page_within_clip_auc_by_fold"]
    assert min(revised[f] for f in V.DECIDING_FOLDS) < 0.60
    assert nod["verdict_revised_page"] == "not validated"
    assert nod["verdict_tuned"] == "not validated"


def test_brow_furrow_was_not_rescued_by_any_alternative_signal() -> None:
    furrow = _artifact()["markers"]["brow_furrow"]
    for key in ("verdict_browser", "verdict_tuned", "verdict_compact", "verdict_heuristic"):
        assert furrow[key] == "not validated", key


def test_the_revised_detector_was_chosen_without_the_held_out_signer() -> None:
    """Every fold records what the other three signers chose, from the declared grid."""
    m = _artifact()["markers"]
    names = set(V.LIVE_PARAMS)
    for marker in ("head_shake", "head_nod"):
        choice = m[marker]["revised_page_choice_by_fold"]
        assert set(choice) >= set(V.DECIDING_FOLDS)
        assert set(choice.values()) <= names
        assert m[marker]["shipped_in_page"]["name"] in names
        assert "not held-out" in m[marker]["shipped_in_page"]["note"]


# --- the second design: nothing in it may read a label --------------------------------------


def _clip(n: int = 72, fps: float = 24.0, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    turn = 0.2 * np.sin(2 * np.pi * 2.5 * t)
    rot = np.tile(np.eye(4), (n, 1, 1))
    rot[:, 0, 0], rot[:, 0, 2] = np.cos(turn), np.sin(turn)
    rot[:, 2, 0], rot[:, 2, 2] = -np.sin(turn), np.cos(turn)
    landmarks = rng.uniform(0.2, 0.8, size=(n, 553, 3))
    return {
        "fps": fps,
        "bs": rng.uniform(0, 0.6, size=(n, 52)),
        "angles": V.HM.head_angles(rot),
        "geometry": V.face_geometry(landmarks, np.ones((n, 4), dtype=bool)),
    }


def test_no_detector_in_the_second_design_can_see_a_label() -> None:
    """The clip handed over has no `y`. A signal that needed one would raise here."""
    clip = _clip()
    families = [*V.CANDIDATES.values(), V.BROWSER, *(V.live_family(m) for m in V.LIVE_AXES)]
    for family in families:
        for name, signal in family.items():
            out = signal(clip)
            assert out.shape == (72,), name
            assert np.isfinite(out).all(), name
    for marker in V.MARKERS:
        assert np.isfinite(V.compact_features(clip, marker)).all()
    assert V.compact_features(clip, "brow_furrow").shape == (72, 16)
    assert V.compact_features(clip, "head_shake").shape == (72, 10)


def test_the_tuned_choice_is_made_on_the_clips_it_is_given_and_no_others() -> None:
    """Planted: on these clips only one family member follows the label, and it is chosen."""
    clips = []
    for seed in range(4):
        c = _clip(seed=seed)
        c["y"] = {"brow_furrow": c["bs"][:, V.VM._IDX["browInnerUp"]] < 0.3}
        clips.append(c)
    assert V.choose_candidate(clips, "brow_furrow") == "inner-brow-up blendshape, negated"


def test_adaptive_level_is_the_pages_rescale() -> None:
    """Baseline at the 20th percentile, full scale at the 95th, never under the floor."""
    x = np.concatenate([np.full(40, 0.2), np.full(40, 0.8)])
    level = V.adaptive_level(x)
    assert level[:40].max() == 0.0
    assert level[40] == pytest.approx(1.5), "a jump over a flat baseline saturates"
    assert level.min() >= 0.0 and level.max() <= 1.5
    flat = V.adaptive_level(np.full(30, 0.4))
    assert flat.max() == 0.0


def test_frames_without_a_face_do_not_invent_brow_geometry() -> None:
    landmarks = np.random.default_rng(0).uniform(0.2, 0.8, size=(10, 553, 3))
    presence = np.ones((10, 4), dtype=bool)
    landmarks[3], presence[3, 0] = 0.0, False
    geo = V.face_geometry(landmarks, presence)
    seen = np.delete(np.arange(10), 3)
    for key, value in geo.items():
        assert value[3] == pytest.approx(float(np.median(value[seen]))), key


def test_the_operating_point_is_set_on_unmarked_frames_only() -> None:
    clip = _clip()
    clip["y"] = {"head_shake": np.arange(72) >= 36}
    signal = V.live_signal("head_shake", V.LIVE_GRID[0])
    thr = V.threshold_at_false_alarm([clip], "head_shake", signal)
    quiet = signal(clip)[:36]
    assert (quiet >= thr).mean() == pytest.approx(V.LIVE_FALSE_ALARM, abs=0.03)
    point = V.operating_point([clip], "head_shake", signal, thr)
    assert set(point) == {
        "threshold_deg_rms",
        "level_only",
        "event_with_swing_gate",
        "old_page_event",
    }
    gated, level = point["event_with_swing_gate"], point["level_only"]
    assert gated["hit_rate"] <= level["hit_rate"], "a gate can only remove detections"
    assert gated["false_alarm_rate"] <= level["false_alarm_rate"]
