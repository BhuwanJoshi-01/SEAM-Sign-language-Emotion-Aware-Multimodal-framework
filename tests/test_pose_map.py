"""Tests for the MediaPipe -> SMPL-24 mapping and its joint normalisation.

These three functions - :func:`mediapipe_to_smpl24`, :func:`normalise_to_rest_frame` and
the private :func:`_spine_shape` that sits between them - had **no test coverage at all**
before this file existed, and that is why two of the defects they carried reached a
rendered video: a spine whose curvature was inflated 1.48x, and legs whose depth was four
times their vertical drop. Both produced geometry that rendered, which is exactly the case
a shape check cannot catch.

The tests here are written against the *properties* the functions must have, not against
recorded numbers, so they keep working if the model file changes:

* the mapped skeleton is anatomically ordered, at every landmark scale;
* normalising into the model's rest frame leaves a rest-pose input unchanged;
* the shaped spine's curvature matches the rest pose's, not an inflated version of it;
* depth damping never lengthens a bone.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from seam.avatar.pose_map import (
    N_SMPL_JOINTS,
    _spine_shape,
    anatomical_report,
    damp_depth_noise,
    mediapipe_to_smpl24,
    normalise_to_rest_frame,
)
from seam.avatar.synthesis import load_smplx, model_rest_pose

#: A model directory, if the licence-gated file is available in this checkout. These
#: tests are valuable without it (they use a synthetic stand-in rest pose) but the most
#: important one - that the spine's curvature is not inflated - only means something
#: against real proportions, because the inflation factor *is* a property of the real
#: ratio between pelvis->spine3 and pelvis->shoulders.
# Candidate locations for the licence-gated SMPL-X model, in priority order.
#
# `SEAM_SMPLX_MODEL` is the project-wide convention and comes first: the original list
# held two hardcoded paths, one of them a Windows desktop belonging to whoever wrote the
# test, so these checks silently skipped on every other machine and on CI. A test that
# cannot run is not a test - and one that skips quietly is worse, because it reads as
# coverage in a report. The original paths are kept so the file still works on the
# machine it was written on.
_MODEL_CANDIDATES: tuple[Path, ...] = tuple(
    p
    for p in (
        Path(v).expanduser()
        for v in ([os.environ["SEAM_SMPLX_MODEL"]] if os.environ.get("SEAM_SMPLX_MODEL") else [])
    )
) + (
    Path("C:/Users/nancy/Desktop/.models/smplx"),
    Path.home() / ".models" / "smplx",
)


def _rest_pose_or_skip() -> np.ndarray:
    for candidate in _MODEL_CANDIDATES:
        if candidate.exists():
            try:
                return model_rest_pose(load_smplx(candidate))
            except Exception:  # pragma: no cover - depends on the local file
                continue
    pytest.skip("SMPL-X model not available; real-proportion checks need it")


def _fake_landmarks(scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """A plausible 553-landmark frame in an arbitrary unit, standing upright.

    Built rather than downloaded so the test runs anywhere. Layout follows `PART_ORDER`:
    pose 0..32, left hand 33..53, right hand 54..74, face 75..552. The hands and face are
    filled because the full frame is what the perception stage produces; note that
    :func:`mediapipe_to_smpl24` consumes only the **pose** slice ``[0:33]``, so the tests
    for it pass ``lm[0:33]``.
    """
    rng = np.random.default_rng(seed)
    lm = np.zeros((553, 3), dtype=np.float64)

    # Pose block, in MediaPipe's ordering - i.e. **image** coordinates, y growing down.
    # MediaPipe's own "left" is the subject's left in a mirrored camera feed, so the
    # indices are entered as the mapper expects to read them, not as the anatomy reads.
    px, py = 0.0, 0.0
    lm[0, :2] = (px, py - 1.55 * scale)  # nose (above the pelvis in image coords)
    lm[7, :2] = (px + 0.12 * scale, py - 1.50 * scale)  # left ear
    lm[8, :2] = (px - 0.12 * scale, py - 1.50 * scale)  # right ear
    lm[11, :2] = (px + 0.18 * scale, py - 0.95 * scale)  # left shoulder
    lm[12, :2] = (px - 0.18 * scale, py - 0.95 * scale)  # right shoulder
    lm[13, :2] = (px + 0.26 * scale, py - 0.65 * scale)  # left elbow
    lm[14, :2] = (px - 0.26 * scale, py - 0.65 * scale)  # right elbow
    lm[15, :2] = (px + 0.32 * scale, py - 0.36 * scale)  # left wrist
    lm[16, :2] = (px - 0.32 * scale, py - 0.36 * scale)  # right wrist
    lm[23, :2] = (px + 0.11 * scale, py - 0.30 * scale)  # left hip
    lm[24, :2] = (px - 0.11 * scale, py - 0.30 * scale)  # right hip
    lm[25, :2] = (px + 0.11 * scale, py + 0.20 * scale)  # left knee
    lm[26, :2] = (px - 0.11 * scale, py + 0.20 * scale)  # right knee
    lm[27, :2] = (px + 0.11 * scale, py + 0.68 * scale)  # left ankle
    lm[28, :2] = (px - 0.11 * scale, py + 0.68 * scale)  # right ankle
    lm[29, :2] = (px + 0.11 * scale, py + 0.74 * scale)  # left heel
    lm[30, :2] = (px - 0.11 * scale, py + 0.74 * scale)  # right heel
    lm[31, :2] = (px + 0.11 * scale, py + 0.76 * scale)  # left foot index
    lm[32, :2] = (px - 0.11 * scale, py + 0.76 * scale)  # right foot index

    # Hands: five points per finger, a short chain outward from the wrist. Filled so the
    # fixture is a complete frame, even though the body mapper ignores this block.
    for base, wx, wy, side in ((33, 0.32, -0.36, 1.0), (54, -0.32, -0.36, -1.0)):
        for finger in range(5):
            for joint in range(4):
                lm[base + finger * 4 + joint, :2] = (
                    (wx + side * 0.02 * joint) * scale,
                    (wy + 0.01 * finger + 0.01 * joint) * scale,
                )
    # Face: a ring, so any derived centroid is defined.
    for i in range(75, 553):
        ang = 2.0 * np.pi * (i - 75) / (553 - 75)
        lm[i, :2] = (
            0.10 * np.cos(ang) * scale,
            (-1.58 + 0.10 * np.sin(ang)) * scale,
        )

    lm += rng.normal(scale=1e-6 * scale, size=lm.shape)
    return lm


# --- the mapping produces a body --------------------------------------------


@pytest.mark.parametrize("scale", [0.3, 1.0, 7.5])
def test_mapped_body_is_anatomically_ordered_at_every_scale(scale: float) -> None:
    """The mapper must not depend on the units its input arrives in.

    Landmarks come back from the perception stage in normalised image coordinates and
    from a different pipeline in pixels; a mapping that only works at one scale is a
    mapping that will be fed the other one eventually.
    """
    body = mediapipe_to_smpl24(_fake_landmarks(scale)[0:33], flip_y=True)
    assert body.shape == (N_SMPL_JOINTS, 3)
    report = anatomical_report(body)
    failed = [k for k, v in report["checks"].items() if not v]
    assert report["passed"], f"anatomy failed at scale {scale}: {failed}"


def test_mapping_flips_y_so_the_body_stands_the_right_way_up() -> None:
    """Image y grows downward; SMPL y grows upward.

    ``flip_y=True`` negates the input's y before mapping. The mapper itself then passes
    y through unchanged, so the *output* is y-up only when the input was already y-down
    (which is what an image gives you). ``make_stimuli`` flips the landmarks itself and
    therefore calls with ``flip_y=False`` - both routes are checked here because getting
    this wrong inverts the body, and an inverted body renders.
    """
    lm = _fake_landmarks()[0:33]  # image coordinates: y grows downward
    upright = mediapipe_to_smpl24(lm, flip_y=True)
    inverted = mediapipe_to_smpl24(lm, flip_y=False)
    assert upright[15, 1] > upright[0, 1], "with flip_y the head ends up above the pelvis"
    assert inverted[15, 1] < inverted[0, 1], "without it the head is below"

    # The route make_stimuli takes: flip first, then ask the mapper not to.
    pre_flipped = lm.copy()
    pre_flipped[:, 1] *= -1.0
    assert np.allclose(mediapipe_to_smpl24(pre_flipped, flip_y=False), upright, atol=1e-12), (
        "flipping before the call must be equivalent to flip_y=True"
    )


def test_mapper_refuses_a_full_landmark_frame() -> None:
    """Feeding all 553 landmarks must fail loudly, not scramble the body.

    The landmark order is not self-describing, so a caller that forgets to slice the pose
    block gets a body built from face points - which renders, and is wrong. The guard is
    the only thing standing between that mistake and a stimulus set.
    """
    with pytest.raises(ValueError, match="33, 3"):
        mediapipe_to_smpl24(_fake_landmarks(), flip_y=True)


# --- the spine's curvature --------------------------------------------------


def test_shaped_spine_has_the_rest_poses_curvature_not_an_inflated_one() -> None:
    """Regression: the spine's bend used to be scaled 1.48x too large.

    `_spine_shape` measures the rest spine's offsets from the straight pelvis->chest line
    and reapplies them to the tracked line. `chest` in the mapper is the *shoulder
    midpoint*, but the offsets were originally measured from the rest pose's
    pelvis->**spine3** line. Those differ in length - 0.435 m against 0.294 m on
    SMPLX_NEUTRAL, a ratio of 1.48 - so the curvature came out 1.48x too strong, pushing
    spine3 past the shoulder line and leaving the retargeter a 50-degree correction to
    make in the spine3 slot. That was the residual torso lean.

    The property tested: applied to the rest pose itself, the shaped spine must reproduce
    the rest pose's own spine. Any change of scale in the offsets breaks it.
    """
    rest = _rest_pose_or_skip()
    shaped = _spine_shape(rest[0], (rest[16] + rest[17]) / 2.0, rest)
    assert shaped.shape == (3, 3)
    for k, joint in enumerate((3, 6, 9)):
        err = float(np.linalg.norm(shaped[k] - rest[joint]))
        assert err < 1e-9, (
            f"shaped spine joint {joint} is {err:.4f} m from the rest pose's own joint; "
            "the rest curvature is being rescaled"
        )


def test_shaped_spine_is_centred_on_the_tracked_line() -> None:
    """A shifted pelvis/chest must move the whole spine with it, not distort it."""
    rest = _rest_pose_or_skip()
    pelvis = rest[0] + np.array([0.5, 0.2, -0.3])
    chest = (rest[16] + rest[17]) / 2.0 + np.array([0.5, 0.2, -0.3])
    shaped = _spine_shape(pelvis, chest, rest)
    expected = _spine_shape(rest[0], (rest[16] + rest[17]) / 2.0, rest) + np.array([0.5, 0.2, -0.3])
    assert np.allclose(shaped, expected, atol=1e-9)


def test_shaped_spine_without_a_rest_pose_is_a_straight_line() -> None:
    """No reference means no claim about curvature - the honest default is a straight line."""
    pelvis = np.zeros(3)
    chest = np.array([0.0, 0.9, 0.0])
    shaped = _spine_shape(pelvis, chest, None)
    # All three points collinear with pelvis and chest.
    for point in shaped:
        cross = np.cross(chest - pelvis, point - pelvis)
        assert np.linalg.norm(cross) < 1e-12


# --- depth damping ----------------------------------------------------------


def test_depth_damping_never_lengthens_a_bone() -> None:
    """Damping may only pull a bone's depth inwards.

    A rule that could lengthen a bone would change the avatar's proportions, and the
    failure mode - a leg that is slightly too long in unusual frames - is hard to see
    and impossible to attribute later.

    Only the thigh and shin are damped; the foot is deliberately left alone (it points
    forward, so depth legitimately dominates it - see the docstring). The foot's
    ``(10, 7)`` and ``(11, 8)`` entries are therefore not in the list, and asserting a
    bound on them here would be asserting the opposite of the intended behaviour.
    """
    rng = np.random.default_rng(7)
    joints = np.zeros((24, 3), dtype=np.float64)
    joints[0] = (0.0, 0.30, 0.0)
    joints[1] = (-0.11, 0.30, 0.0)
    joints[2] = (0.11, 0.30, 0.0)
    # Legs with a deliberately absurd depth, like MediaPipe's.
    joints[4] = (-0.11, 0.10, -0.80)
    joints[5] = (0.11, 0.10, -0.80)
    joints[7] = (-0.11, -0.40, -1.60)
    joints[8] = (0.11, -0.40, -1.60)
    joints[10] = (-0.11, -0.45, -1.70)
    joints[11] = (0.11, -0.45, -1.70)
    joints[3:] += rng.normal(scale=0.01, size=(21, 3))

    out = damp_depth_noise(joints, depth_share=0.25)
    for child, parent in ((4, 1), (5, 2), (7, 4), (8, 5)):
        before = float(np.linalg.norm(joints[child] - joints[parent]))
        after = float(np.linalg.norm(out[child] - out[parent]))
        assert after <= before + 1e-9, (
            f"bone {parent}->{child} grew from {before:.4f} to {after:.4f}"
        )


def test_depth_damping_leaves_the_foot_alone() -> None:
    """The foot bone must pass through untouched.

    The foot runs forward from the ankle, so a cap calibrated on the thigh flattens it.
    Measured on ``SMPLX_NEUTRAL`` the ankle->foot depth share is 1.58 and 1.70 against
    0.025-0.079 for the thigh and shin; capping it moved the model's own rest pose by
    0.101 m - a 10 cm error injected into a pose that needed none.
    """
    joints = np.zeros((24, 3), dtype=np.float64)
    joints[7] = (-0.11, -0.40, -1.60)
    joints[8] = (0.11, -0.40, -1.60)
    joints[10] = (-0.11, -0.42, -1.00)  # foot pointing forward: depth > height
    joints[11] = (0.11, -0.42, -1.00)
    out = damp_depth_noise(joints, depth_share=0.25)
    for joint in (10, 11):
        assert np.allclose(out[joint], joints[joint]), (
            f"foot joint {joint} was modified by the thigh/shin depth cap"
        )


def test_depth_damping_leaves_the_xy_projection_alone() -> None:
    """Only depth may change; the limb must not slide sideways in the picture."""
    joints = np.zeros((24, 3), dtype=np.float64)
    joints[1] = (-0.11, 0.30, 0.0)
    joints[4] = (-0.13, 0.12, -0.90)
    out = damp_depth_noise(joints, depth_share=0.25)
    assert np.allclose(out[1, :2], joints[1, :2])
    assert np.allclose(out[4, :2], joints[4, :2])


def test_depth_damping_brings_absurd_depth_within_the_cap() -> None:
    """The whole point: a leg bone may not be mostly depth."""
    joints = np.zeros((24, 3), dtype=np.float64)
    joints[1] = (-0.11, 0.30, 0.0)
    joints[4] = (-0.11, 0.10, -0.80)  # depth 4x the vertical drop
    out = damp_depth_noise(joints, depth_share=0.25)
    bone = out[4] - out[1]
    planar = float(np.hypot(bone[0], bone[1]))
    assert abs(float(bone[2])) <= 0.25 / np.sqrt(1 - 0.25**2) * planar + 1e-9


# --- normalisation ----------------------------------------------------------


def test_normalising_the_rest_pose_returns_that_pose_up_to_the_pelvis_offset() -> None:
    """Normalising into a frame is idempotent on that frame's *shape*.

    This is the property that makes "no rotation" mean "standing in the rest pose". If it
    fails, every joint receives a spurious rotation in every frame, which is how the very
    first stimulus videos came out lying on their backs.

    The output is not expected to equal the input joint-for-joint: step 1 translates the
    pelvis to the origin, so the whole body is shifted by ``-rest[0]``. What must be
    preserved is every bone, i.e. ``out[j] - out[0]`` must equal ``rest[j] - rest[0]``.
    Asserting full equality instead would be asserting that the function does not do the
    one thing its docstring says it does.
    """
    rest = _rest_pose_or_skip()
    out = normalise_to_rest_frame(rest, rest=rest)
    assert np.allclose(out[0], 0.0, atol=1e-9), "pelvis must land at the origin"
    # Compare everything relative to the pelvis, which is the translation-invariant part.
    assert np.allclose(out - out[0], rest - rest[0], atol=1e-6)


def test_normalising_does_not_scale_an_already_correct_body() -> None:
    """Feeding the rest pose back in must not shrink or stretch it.

    The scale factor is computed from the tracked body's y-extent against the rest pose's.
    For the rest pose those are the same number by definition, so the factor is 1. A
    mistake here - measuring height over the wrong axis, or normalising by the wrong
    joint - rescales every stimulus, and a body that is uniformly 10% small is not
    obviously wrong in a single frame.
    """
    rest = _rest_pose_or_skip()
    out = normalise_to_rest_frame(rest, rest=rest)
    bones_in = np.linalg.norm(rest[1:] - rest[0], axis=1)
    bones_out = np.linalg.norm(out[1:] - out[0], axis=1)
    assert np.allclose(bones_in, bones_out, rtol=1e-6, atol=1e-9)


def test_normalisation_survives_a_pure_scale_and_translation() -> None:
    """Scaling and moving the tracked body must not change the pose it describes."""
    rest = _rest_pose_or_skip()
    moved = rest * 100.0 + np.array([50.0, -20.0, 7.0])
    out = normalise_to_rest_frame(moved, rest=rest)
    assert np.allclose(out - out[0], rest - rest[0], atol=1e-4)


def test_normalisation_rejects_the_wrong_joint_count() -> None:
    with pytest.raises(ValueError, match="expected"):
        normalise_to_rest_frame(np.zeros((10, 3)))
