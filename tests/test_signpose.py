"""Tests for the signing-space feature extractor.

Each test pins one claim the module makes in its docstring, because the failure modes
here are silent: a sign-space bug produces a model that still trains, still reports a
plausible accuracy, and is simply wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.features.signpose import (
    FACE_POINTS,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    POSE_POINTS,
    R_ELBOW,
    R_HIP,
    R_SHOULDER,
    R_WRIST,
    TOTAL_POINTS,
    UPPER_BODY,
    feature_dim,
    signing_space,
    split_landmarks,
    token_feature,
)


def _pose(t: int = 4, seed: int = 0) -> np.ndarray:
    """A plausible (T, 33, 3) pose: upright, y down as MediaPipe emits it."""
    rng = np.random.default_rng(seed)
    p = np.zeros((t, POSE_POINTS, 3))
    p[:, :, 1] = rng.normal(0.35, 0.01, size=(t, POSE_POINTS))
    # MediaPipe y grows downward, so a smaller y is higher in the image. The nose must
    # sit above the shoulder line and the hips below it, or the frame's head-up
    # disambiguation resolves the wrong way and the sign space comes out upside down.
    p[:, NOSE, 1] = 0.10
    p[:, L_HIP, 1] = p[:, R_HIP, 1] = 0.65
    # MediaPipe labels shoulders *anatomically*. For a front-facing, unmirrored camera
    # the signer's left shoulder is on the image's right, so it has the larger x. An
    # earlier version of this fixture put the left shoulder at negative x - the layout of
    # a selfie-mirrored capture - which made the sign-space x axis point at the signer's
    # left and made the mirror test fail for a reason that had nothing to do with the
    # mirror test.
    p[:, L_SHOULDER, 0], p[:, R_SHOULDER, 0] = 0.15, -0.15
    p[:, L_SHOULDER, 1] = p[:, R_SHOULDER, 1] = 0.30
    p[:, L_ELBOW, 1], p[:, R_ELBOW, 1] = 0.22, 0.22
    p[:, L_WRIST, 0], p[:, L_WRIST, 1] = 0.25, 0.15
    p[:, R_WRIST, 0], p[:, R_WRIST, 1] = -0.25, 0.15
    p[:, :, 2] = rng.normal(0.0, 0.02, size=(t, POSE_POINTS))
    return p


def _landmarks(pose: np.ndarray) -> np.ndarray:
    t = pose.shape[0]
    lm = np.zeros((t, TOTAL_POINTS, 3))
    lm[:, FACE_POINTS : FACE_POINTS + POSE_POINTS] = pose
    lm[:, FACE_POINTS + POSE_POINTS : FACE_POINTS + POSE_POINTS + 21] = 0.01
    lm[:, -21:] = 0.01
    return lm


def test_layout_offsets_are_the_documented_ones() -> None:
    assert (FACE_POINTS, POSE_POINTS, TOTAL_POINTS) == (478, 33, 553)
    face, pose, lh, rh = split_landmarks(_landmarks(_pose()))
    assert face.shape[1] == 478
    assert pose.shape[1] == 33
    assert lh.shape[1] == rh.shape[1] == 21


def test_split_refuses_a_wrongly_sized_array() -> None:
    """A silent re-slice would produce plausible garbage from a changed extractor."""
    with pytest.raises(ValueError, match="553"):
        split_landmarks(np.zeros((2, 468, 3)))


def test_origin_is_the_shoulder_midpoint() -> None:
    sp = signing_space(_pose())
    centre = 0.5 * (sp[:, UPPER_BODY.index(L_SHOULDER)] + sp[:, UPPER_BODY.index(R_SHOULDER)])
    assert np.allclose(centre, 0.0, atol=1e-6)


def test_shoulder_width_is_the_unit_of_scale() -> None:
    """Shoulders must sit at +/- 0.5 once scaled, or the scale is arbitrary."""
    sp = signing_space(_pose())
    i, j = UPPER_BODY.index(L_SHOULDER), UPPER_BODY.index(R_SHOULDER)
    # Sign, not order: the levelling rotation makes the left-to-right axis point along
    # +x, so which of the two shoulders ends up on the left is a property of the input
    # pose, not of the frame. The distance is what the scale claim is about.
    assert np.allclose(np.abs(sp[:, j, 0] - sp[:, i, 0]), 1.0, atol=1e-6)


def test_invariant_to_translation_scale_and_in_plane_rotation() -> None:
    """The whole point of the body-relative frame.

    The same sign recorded at a different distance, height or body roll must produce the
    same features; if it does not, the model is partly a camera-position detector.
    """
    base = _pose()
    fb = token_feature(_landmarks(base), 1, 4)

    shifted = base.copy()
    shifted[..., 0] += 0.4
    shifted[..., 1] -= 0.25
    assert np.allclose(token_feature(_landmarks(shifted), 1, 4), fb, atol=1e-6)

    scaled = base * 2.0
    assert np.allclose(token_feature(_landmarks(scaled), 1, 4), fb, atol=1e-6)

    th = np.deg2rad(23.0)
    rot = base.copy()
    c, s = np.cos(th), np.sin(th)
    x, y = rot[..., 0].copy(), rot[..., 1].copy()
    rot[..., 0] = x * c - y * s
    rot[..., 1] = x * s + y * c
    assert np.allclose(token_feature(_landmarks(rot), 1, 4), fb, atol=1e-6)


def test_up_is_up() -> None:
    """A wrist above the shoulders must have positive y.

    MediaPipe emits y downward; if the flip were missing, every vertical movement would
    be inverted and the avatar's hands would move opposite to the signer.
    """
    p = _pose()
    p[:, L_WRIST, 1] = 0.05  # smaller y = higher in the image
    sp = signing_space(p)
    assert np.all(sp[:, UPPER_BODY.index(L_WRIST), 1] > 0)


def test_mirroring_keeps_the_frame_upright_but_reverses_the_x_convention() -> None:
    """A limitation, pinned rather than papered over.

    An earlier version of this test asserted that mirroring the image leaves the
    features unchanged. It cannot, and no bug fix delivers it. In a single 2-D view, a
    signer captured by a front-facing camera and the same signer seen through a
    mirrored one produce the *same* image modulo handedness - there is no third
    dimension to disambiguate them. The frame therefore resolves the ambiguity one way,
    and the honest statement is what that resolution costs: the pose stays upright and
    y stays up, but +x names the signer's left rather than their right.

    Consequence for M5a: a mirrored capture contributes a systematic x-sign flip to the
    features. This is why the frame is not a license to ignore capture geometry, and why
    a mirrored/selfie source needs either an explicit un-mirror step or per-clip
    provenance before its tokens are used. The alternative - flipping x on a
    per-clip basis by some rule - would silently corrupt correctly-captured clips.
    """
    p = _pose()
    mirrored = p.copy()
    mirrored[..., 0] *= -1
    a, b = signing_space(p), signing_space(mirrored)
    nose = UPPER_BODY.index(NOSE)
    li, ri = UPPER_BODY.index(L_WRIST), UPPER_BODY.index(R_WRIST)
    assert np.all(b[:, nose, 1] > 0), "mirroring must not put the head down"
    assert np.allclose(a[:, nose, 1], b[:, nose, 1], atol=1e-6)
    assert np.all(a[:, li, 0] < 0) and np.all(a[:, ri, 0] > 0)
    assert np.all(b[:, li, 0] > 0) and np.all(b[:, ri, 0] < 0), (
        "the x convention reverses under mirroring - the documented limitation"
    )


def test_x_is_the_signers_right_for_an_asymmetric_pose() -> None:
    """With a known asymmetric pose, the signer's right wrist is the one further out."""
    p = _pose()
    p[:, R_WRIST, 0] += 0.3
    sp = signing_space(p)
    assert np.all(sp[:, UPPER_BODY.index(R_WRIST), 0] > sp[:, UPPER_BODY.index(L_WRIST), 0])


def test_lever_and_lean_are_available_as_choices() -> None:
    """level=False must keep torso lean, or the choice is not actually a choice."""
    p = _pose()
    p[:, R_SHOULDER, 1] -= 0.2  # tilt one shoulder down
    lev, raw = signing_space(p, level=True), signing_space(p, level=False)
    assert not np.allclose(lev, raw)
    assert np.std(raw[..., 1]) > np.std(lev[..., 1])


def test_degenerate_shoulder_width_does_not_explode() -> None:
    """A dropout frame must not be scaled by ~0 and multiplied into a huge offset."""
    p = _pose()
    p[1, L_SHOULDER] = p[1, R_SHOULDER]  # zero shoulder width on one frame
    sp = signing_space(p)
    assert np.isfinite(sp).all()
    assert np.abs(sp).max() < 10.0
    assert np.all(sp[1] == 0.0), "a dropout frame is 'no measurement', not a huge one"
    assert not np.all(sp[0] == 0.0), "the good frames must survive"


def test_overshoot_raises_rather_than_clamping() -> None:
    """The ~10% of tokens past their crop must fail loudly, not be relabelled."""
    lm = _landmarks(_pose(t=6))
    with pytest.raises(IndexError, match="outside"):
        token_feature(lm, 5, 9)
    with pytest.raises(IndexError, match="outside"):
        token_feature(lm, 0, 3)


def test_one_based_inclusive_frames() -> None:
    """Off-by-one here silently shifts every token by a frame."""
    lm = _landmarks(_pose(t=6))
    a = token_feature(lm, 2, 4)
    b = token_feature(lm, 1, 3)
    assert not np.allclose(a, b)


def test_duration_is_appended_separately() -> None:
    """Duration must be its own scalar, so a model can be tested without it.

    On this corpus clip duration correlates with syntactic class, so leaving it mixed
    into the pose features would let a duration classifier pass as a sign recogniser.
    """
    lm = _landmarks(_pose(t=12))
    for start, end in ((1, 3), (1, 9), (4, 5)):
        f = token_feature(lm, start, end)
        assert f[-1] == pytest.approx((end - start + 1) / 10.0), (
            "the trailing scalar must be exactly the token duration, on its own"
        )
    assert token_feature(lm, 1, 9)[-1] > token_feature(lm, 1, 3)[-1]
    # feature_dim counts the duration as one extra column, so a model can drop it.
    assert feature_dim("upper") == len(UPPER_BODY) * 3 * 3 + 1


def test_feature_dim_matches_what_is_produced() -> None:
    lm = _landmarks(_pose(t=6))
    for part in ("upper", "upper+hands"):
        assert token_feature(lm, 1, 4, part=part).shape[0] == feature_dim(part)  # type: ignore[arg-type]


def test_hands_are_wrist_relative() -> None:
    """Finger shape is defined against the hand, not the shoulder or the body tracker.

    Translating the whole hand mesh must leave the feature unchanged. Anchoring on the
    body pose's wrist estimate instead of the hand mesh's own wrist (point 0) would fail
    this, because the body estimate does not move with the mesh - so whole-hand travel
    would leak into the finger coordinates.
    """
    lm = _landmarks(_pose(t=6))
    base = token_feature(lm, 1, 4, part="upper+hands")
    moved = lm.copy()
    h0 = FACE_POINTS + POSE_POINTS
    moved[:, h0 : h0 + 21, :] += np.array([0.3, -0.2, 0.1])
    assert np.allclose(token_feature(moved, 1, 4, part="upper+hands"), base, atol=1e-6)
