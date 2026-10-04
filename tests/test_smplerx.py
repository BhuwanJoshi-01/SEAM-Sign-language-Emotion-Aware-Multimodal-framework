"""Tests for the SMPLer-X adapter.

Scoped to the pure functions - the ones that can be tested without the licence-gated
model, the GPU, or the read-only SMPLer-X tree. The parts that need those (actually
running the regressor) are exercised by ``scripts/make_avatar_demo.py``, which is the only
honest way to test them.

The rotation algebra gets the most attention here because it is where the bugs were. Every
one of these was found by rendering a frame and looking at it, which is the slowest
possible way to find a sign error.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from seam.avatar.synthesis import SmplxFrame
from seam.perception.smplerx import (
    SmplerxUnavailable,
    _aa_to_matrix,
    _matrix_to_aa,
    _minimal_rotation,
    locate,
    motion_energy,
    parse,
    repair_outliers,
    upright,
)

SMPLERX_ROOT = "/nonexistent/smplerx"


# ── rotation algebra ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "aa",
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, np.pi / 2],
        [0.3, -0.2, 0.1],
        [np.pi, 0.0, 0.0],
        [0.0, np.pi, 0.0],
    ],
)
def test_matrix_aa_roundtrip(aa: list[float]) -> None:
    a = np.asarray(aa, dtype=np.float64)
    back = _matrix_to_aa(_aa_to_matrix(a))
    assert np.allclose(_aa_to_matrix(back), _aa_to_matrix(a), atol=1e-9)


def test_minimal_rotation_maps_a_onto_b() -> None:
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        if np.linalg.norm(a) < 1e-6 or np.linalg.norm(b) < 1e-6:
            continue
        R = _minimal_rotation(a, b)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
        # A rotation, not a reflection.
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


def test_minimal_rotation_opposite_vectors() -> None:
    """The antiparallel case is the one that skips the cross product, so it is separate."""
    R = _minimal_rotation(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))
    assert np.allclose(R @ np.array([1.0, 0, 0]), [-1.0, 0, 0], atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-9)


# ── parsing ───────────────────────────────────────────────────────────────────


def _raw(n: int = 3, **over: object) -> dict:
    frame = {
        "detected": True,
        "global_orient": [0.0, 0.0, 0.0],
        "body_pose": [0.0] * 63,
        "left_hand_pose": [0.0] * 45,
        "right_hand_pose": [0.0] * 45,
        "jaw_pose": [0.0] * 3,
        "betas": [0.0] * 10,
        "expression": [0.0] * 10,
        "cam_trans": [0.0, 0.0, 0.0],
    }
    frame.update(over)  # type: ignore[arg-type]
    return {"n_frames": n, "frames": [dict(frame) for _ in range(n)]}


def test_parse_shapes(tmp_path) -> None:
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(_raw()))
    r = parse(p)
    assert len(r.frames) == 3
    f = r.frames[0]
    assert f.body_pose.shape == (21, 3)
    assert f.left_hand_pose.shape == (15, 3)
    assert f.right_hand_pose.shape == (15, 3)
    assert f.jaw_pose.shape == (3,)
    assert f.expression.shape == (10,)
    assert r.coverage == 1.0


def test_parse_rejects_wrong_body_pose_length(tmp_path) -> None:
    """A short body_pose reshaped to (21, 3) still renders - a different wrong person.

    So the shape check is the only thing between a schema change and a plausible lie.
    """
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(_raw(1, body_pose=[0.0] * 60)))
    with pytest.raises(ValueError, match="body_pose"):
        parse(p)


def test_parse_rejects_wrong_hand_length(tmp_path) -> None:
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(_raw(1, left_hand_pose=[0.0] * 30)))
    with pytest.raises(ValueError, match="hand pose"):
        parse(p)


def test_undetected_frames_are_counted_not_filled(tmp_path) -> None:
    """A gap must be reported. Interpolating it silently would be inventing data."""
    d = _raw(3)
    d["frames"][1]["detected"] = False
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(d))
    r = parse(p)
    assert r.n_detected == 2
    assert r.n_nobox == 1
    assert r.n_nobox_frames == [1]
    assert r.coverage == pytest.approx(2 / 3)


# ── motion energy ─────────────────────────────────────────────────────────────


def test_motion_energy_flags_a_static_sequence(tmp_path) -> None:
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(_raw(10)))
    m = motion_energy(parse(p))
    assert m["mean_joint_speed"] == pytest.approx(0.0)
    assert m["body_range_rad"] == pytest.approx(0.0)


def test_motion_energy_detects_motion(tmp_path) -> None:
    d = _raw(10)
    for i, fr in enumerate(d["frames"]):
        fr["body_pose"] = [0.0] * 63
        fr["body_pose"][0] = 0.1 * i
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(d))
    assert motion_energy(parse(p))["mean_joint_speed"] > 0.0


# ── upright / repair ──────────────────────────────────────────────────────────


def test_upright_puts_the_head_above_the_pelvis(tmp_path) -> None:
    """SMPLer-X arrives upside down; this is the invariant that proves it was fixed.

    On a real clip global_orient x averages 129 deg and the head lands at y = -0.69 m.
    """
    from seam.perception.smplerx import check_upright

    d = _raw(4)
    for fr in d["frames"]:
        fr["global_orient"] = [2.2, 0.0, 0.0]  # ~126 deg about x
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(d))
    r = parse(p)
    assert check_upright(r)["frames_head_below_pelvis"] > 0
    u = upright(r)
    stats = check_upright(u)
    assert stats["frames_head_below_pelvis"] == 0
    assert stats["head_above_pelvis_min_m"] > 0.0


def test_upright_does_not_change_the_articulated_pose(tmp_path) -> None:
    """It is a coordinate correction. If it moved joints, motion energy would change."""
    d = _raw(6)
    for i, fr in enumerate(d["frames"]):
        fr["body_pose"] = [0.0] * 63
        fr["body_pose"][0] = 0.05 * i
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(d))
    r = parse(p)
    u = upright(r)
    for a, b in zip(r.frames, u.frames, strict=True):
        assert np.allclose(a.body_pose, b.body_pose)
        assert np.allclose(a.left_hand_pose, b.left_hand_pose)
    assert motion_energy(u)["mean_joint_speed"] == pytest.approx(
        motion_energy(r)["mean_joint_speed"]
    )


def test_repair_outliers_finds_collapsed_frames(tmp_path) -> None:
    """Detection is on head-above-pelvis, not deviation from the median pose.

    A median-based test found zero outliers on the very clip that visibly has three,
    because signing moves far from the median by definition.
    """
    d = _raw(12)
    for i, fr in enumerate(d["frames"]):
        # 9 good frames upright, then 3 collapsed (near-zero global rotation + bent neck).
        fr["global_orient"] = [0.0, 0.0, 0.0] if i < 9 else [2.0, 0.0, 0.0]
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(d))
    r = parse(p)
    fixed, bad = repair_outliers(upright(r))
    assert bad == [9, 10, 11]
    assert fixed.frames[9].body_pose.shape == (21, 3)


def test_repair_leaves_a_clean_clip_alone(tmp_path) -> None:
    d = _raw(6)
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(d))
    r = parse(p)
    _, bad = repair_outliers(upright(r))
    assert bad == []


def test_uniformly_collapsed_clip_is_not_flagged(tmp_path) -> None:
    """The detector is median-relative, so it cannot catch a wholly collapsed clip.

    If every frame is equally bad then the median is equally bad and no frame falls below
    half of it. This is a real limitation, not a passing test: `repair_outliers` repairs
    *outliers*, and a clip where the regressor failed from end to end is invisible to it.
    Such a clip still renders, just wrongly, so the demo's uprightness numbers - not this
    function - are what would catch it.
    """
    d = _raw(4)
    for fr in d["frames"]:
        body = [0.0] * 63
        body[11 * 3] = 2.6  # neck, joint index 11 in body_pose
        fr["body_pose"] = body
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(d))
    _, bad = repair_outliers(upright(parse(p)))
    assert bad == []


def test_repair_interpolates_across_a_short_gap(tmp_path) -> None:
    """A collapse in the middle is held between its two good neighbours."""
    d = _raw(9)
    for i, fr in enumerate(d["frames"]):
        # A per-frame root tilt, which upright()'s single clip-wide correction cannot hide.
        # (A clip bent only in the *neck* stays bent: upright derives its "up" from the
        # median pose, so a uniformly bent clip has a bent median and no correction.)
        fr["global_orient"] = [0.0, 0.0, 0.0] if i not in (4, 5) else [2.0, 0.0, 0.0]
        body = [0.0] * 63
        body[0] = 0.1 * i  # so the frames are distinguishable and interpolation is visible
        fr["body_pose"] = body
    p = tmp_path / "raw.json"
    p.write_text(json.dumps(d))
    fixed, bad = repair_outliers(upright(parse(p)))
    assert bad == [4, 5]
    # Interpolated, so strictly between the neighbours rather than copied from either.
    between = fixed.frames[4].body_pose[0, 0]
    assert 0.3 < between < 0.5, f"expected a blend near 0.4, got {between}"


# ── locating the read-only tree ───────────────────────────────────────────────


def test_locate_reports_the_missing_piece(tmp_path) -> None:
    with pytest.raises(SmplerxUnavailable, match="does not exist"):
        locate(root=tmp_path / "absent")


def test_locate_reports_a_missing_runner(tmp_path) -> None:
    (tmp_path / "main").mkdir()
    with pytest.raises(SmplerxUnavailable, match="runner missing"):
        locate(root=tmp_path)


def test_locate_reports_a_missing_checkpoint(tmp_path) -> None:
    (tmp_path / "main").mkdir()
    (tmp_path / "main" / "nsl_runner.py").write_text("")
    with pytest.raises(SmplerxUnavailable, match="checkpoint missing"):
        locate(root=tmp_path)


def test_smplxframe_defaults_are_neutral_not_garbage() -> None:
    """A SmplxFrame must default to a neutral pose, not to uninitialised memory."""
    f = SmplxFrame()
    assert np.all(f.body_pose == 0)
    assert np.all(f.expression == 0)
    assert f.body_pose.shape == (21, 3)
