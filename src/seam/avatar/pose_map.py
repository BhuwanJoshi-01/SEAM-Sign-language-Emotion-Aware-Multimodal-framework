"""MediaPipe pose -> SMPL-X 24-joint order, for the avatar path.

**Why this module exists.** The avatar chain (``landmark .npz -> SMPL-X parameters ->
mesh``) was producing geometry whose joints were anatomically impossible: the head was
driven by the wrist, the spine sat below the hips, and the whole skeleton occupied a
small blob. Three independent defects, all invisible to the test suite:

1. **The landmark layout disagreed between two modules.** The extractor writes the 553
   points in ``PART_ORDER = ("pose", "left_hand", "right_hand", "face")`` - pose at
   ``0..32``, face at ``75..552``. ``seam.features.signpose.split_landmarks`` assumed the
   opposite, face-first: face ``0..477``, pose ``478..510``. So the block handed to
   :func:`seam.avatar.synthesis.retarget_body` as "pose" was in fact a slice of *face*
   landmarks. Measured on a real clip: the real pose block is ordered nose -> shoulder ->
   hip -> ankle (0.24 / 0.42 / 0.83 / 1.41 in image y), while what the splitter returned
   was not ordered at all, differing from the real block by up to 1.78.

2. **MediaPipe's 33 joints are not SMPL's 24.** Even reading the correct block,
   ``retarget_body`` indexes the first 24 rows as if they were SMPL joints. MediaPipe
   index 15 is the left wrist; SMPL index 15 is the head. Nothing remapped them.

3. **Image y grows downward; the rest pose's y grows upward.** ``retarget_body`` compares
   bone directions against :func:`seam.avatar.synthesis.canonical_rest_pose`, which is
   built y-up. The raw landmarks were never flipped, so every bone direction was compared
   upside-down.

**What this module does not do.** It does not invent the joints MediaPipe lacks. MediaPipe
Pose has no pelvis, no spine chain and no collars, so those are *derived* - by midpoint or
interpolation between landmarks that do exist - and that derivation is stated here rather
than hidden, because it is the one part of the mapping that is a modelling choice rather
than a correspondence.
"""

from __future__ import annotations

import numpy as np

from seam.avatar.synthesis import SMPL_BODY_JOINTS, canonical_rest_pose

# --- MediaPipe BlazePose landmark indices we use ---------------------------
NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

N_MEDIAPIPE_POSE = 33
N_SMPL_JOINTS = 24

#: SMPL joint order, for reference and for tests. ``pelvis`` is the root.
SMPL_JOINT_NAMES = ("pelvis", *SMPL_BODY_JOINTS)

#: SMPL joint indices in the leg chain, both sides: hip, knee, ankle, foot.
#: Used by :func:`normalise_to_rest_frame` to damp the depth these joints are given by
#: monocular tracking - see the note at step 4 of its docstring.
LEG_JOINTS: tuple[int, ...] = (1, 2, 4, 5, 7, 8, 10, 11)

#: SMPL index of the neck. Named rather than written as ``12`` because the torso axis
#: used to upright the tracked body is built from it, and a reader checking that choice
#: should not have to count through a table of indices to find out which joint it is.
NECK = 12

#: Joints copied straight from one MediaPipe landmark. Index is the SMPL joint.
_DIRECT: dict[int, int] = {
    1: LEFT_HIP,
    2: RIGHT_HIP,
    4: LEFT_KNEE,
    5: RIGHT_KNEE,
    7: LEFT_ANKLE,
    8: RIGHT_ANKLE,
    10: LEFT_FOOT_INDEX,
    11: RIGHT_FOOT_INDEX,
    16: LEFT_SHOULDER,
    17: RIGHT_SHOULDER,
    18: LEFT_ELBOW,
    19: RIGHT_ELBOW,
    20: LEFT_WRIST,
    21: RIGHT_WRIST,
}


def _mean(p: np.ndarray, idx: list[int]) -> np.ndarray:
    """Mean of several landmarks, keeping the leading batch axes."""
    return np.mean(p[..., idx, :], axis=-2)


def mediapipe_to_smpl24(
    pose: np.ndarray, *, flip_y: bool = True, rest: np.ndarray | None = None
) -> np.ndarray:
    """``(..., 33, 3)`` MediaPipe pose -> ``(..., 24, 3)`` in SMPL joint order.

    ``flip_y`` negates the second axis, converting the image convention (y grows
    downward) to the world convention :func:`canonical_rest_pose` is written in (y grows
    upward). Leave it on for raw extractor output.

    ``rest`` is the rest pose of the model the result will drive. It is used only for the
    *shape* of the derived spine; pass it whenever a model is loaded. See
    :func:`_spine_shape` for why a straight spine is not good enough.

    The seven joints MediaPipe has no landmark for are derived:

    - ``pelvis`` (0) - midpoint of the two hips
    - ``chest`` - midpoint of the two shoulders; this is where spine3 *aims*
    - ``spine1`` (3) / ``spine2`` (6) / ``spine3`` (9) - along pelvis -> chest, following
      the rest pose's own spine curvature when ``rest`` is given (see
      :func:`_spine_shape`). A straight-line spine makes the retargeter bend the model's
      curved spine flat, which leans the whole torso.
    - ``neck`` (12) - midway between the chest and the head
    - ``left_collar`` (13) / ``right_collar`` (14) - midway from the chest to each
      shoulder, which is what a collar bone spans
    - ``head`` (15) - midpoint of the two ears, which is a stabler centre than the nose,
      which sits forward of the skull's rotation centre

    Hand joints (22, 23) use the mean of pinky, index and thumb, i.e. the palm centre.
    """
    p = np.asarray(pose, dtype=np.float64)
    if p.shape[-2] != N_MEDIAPIPE_POSE or p.shape[-1] != 3:
        raise ValueError(
            f"pose must be (..., {N_MEDIAPIPE_POSE}, 3) in MediaPipe order, got "
            f"{p.shape}. Feeding anything else silently produces a scrambled body: the "
            "landmark order is not self-describing."
        )
    if flip_y:
        # A copy, so the caller's array is not mutated behind their back.
        p = p.copy()
        p[..., 1] = -p[..., 1]

    batch = p.shape[:-2]
    out = np.zeros((*batch, N_SMPL_JOINTS, 3), dtype=np.float64)

    for smpl_idx, mp_idx in _DIRECT.items():
        out[..., smpl_idx, :] = p[..., mp_idx, :]

    left_hip = p[..., LEFT_HIP, :]
    right_hip = p[..., RIGHT_HIP, :]
    pelvis = (left_hip + right_hip) / 2.0
    out[..., 0, :] = pelvis

    chest = (p[..., LEFT_SHOULDER, :] + p[..., RIGHT_SHOULDER, :]) / 2.0

    spine = _spine_shape(pelvis, chest, rest)
    out[..., 3, :] = spine[..., 0, :]
    out[..., 6, :] = spine[..., 1, :]
    out[..., 9, :] = spine[..., 2, :]

    head = _mean(p, [LEFT_EAR, RIGHT_EAR])
    out[..., 15, :] = head
    out[..., 12, :] = (chest + head) / 2.0

    out[..., 13, :] = (chest + p[..., LEFT_SHOULDER, :]) / 2.0
    out[..., 14, :] = (chest + p[..., RIGHT_SHOULDER, :]) / 2.0

    out[..., 22, :] = _mean(p, [LEFT_PINKY, LEFT_INDEX, LEFT_THUMB])
    out[..., 23, :] = _mean(p, [RIGHT_PINKY, RIGHT_INDEX, RIGHT_THUMB])

    return out


def _spine_shape(pelvis: np.ndarray, chest: np.ndarray, rest: np.ndarray | None) -> np.ndarray:
    """Place spine1 (3), spine2 (6) and spine3 (9) between ``pelvis`` and ``chest``.

    Returns three positions in the order pelvis-side to chest-side.

    **Why this is not simply a straight line.** The obvious derivation - one third and two
    thirds along ``pelvis -> chest``, with spine3 at the chest - puts the whole spine on a
    single straight segment. That is wrong for two reasons.

    First, it is not the shape a real spine has: the model's own rest joints curve. Second,
    and this is the one that shows up in the render, :func:`retarget_body` measures each
    spine bone's *direction* against the rest pose's. A straight tracked spine reports the
    same direction for spine1, spine2 and spine3, while the rest pose's differ by up to
    thirty degrees - so the retargeter is asked to bend a straight spine into a zig-zag or
    vice versa, and the accumulated result leans the whole torso, measured at 18-20 degrees
    on clips whose signers stand within 3 degrees of vertical.

    So when the rest pose is known, the tracked spine is given the rest pose's own
    *shape*: the rest spine's offsets from the straight pelvis->chest line are measured,
    then re-applied to the tracked line, scaled by its length. The tracked spine then
    differs from the rest spine only by the whole-body orientation, which is exactly what
    the retargeter is supposed to be measuring. With no rest pose available the offsets
    are zero and this reduces to the straight line - the honest answer when there is
    nothing to say what shape the spine should be.

    **Both lines must be the same line, and that is what the shoulder midpoint is for.**
    ``chest`` here is the *shoulder midpoint* - the mapper's only direct measurement of
    where the torso ends, since MediaPipe has no spine landmark at all. The rest pose's
    offsets therefore have to be measured from the rest pose's own pelvis->shoulder line,
    not from its pelvis->spine3 line. Those two lines are not the same length: on
    ``SMPLX_NEUTRAL`` the pelvis->shoulder midpoint is 0.435 m while pelvis->spine3 is
    0.294 m, a ratio of 1.48. Measuring curvature against the short line and reapplying it
    against the long one inflates the spine's bend by that factor, which overshoots
    spine3 past the shoulder line and leaves the retargeter a 30-to-50-degree correction
    to make in the spine3 slot - the residual torso lean. Measured before this fix:
    shaped spine2 sat 0.114 m off the straight line where the rest pose's sat 0.031 m.
    """
    span = chest - pelvis
    if rest is None:
        # shape (..., 3, 3) - the three spine joints
        frac = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0])[..., None]
        return pelvis[..., None, :] + frac * span[..., None, :]

    # Rest-pose spine, expressed as offsets from its own straight pelvis->chest line.
    # `chest` for the rest pose is the shoulder midpoint, matching the tracked `chest`
    # above; using `rest[9]` here instead is the 1.48x inflation described above.
    r_pelvis = rest[0]
    r_chest = (rest[16] + rest[17]) / 2.0
    r_span = r_chest - r_pelvis
    r_len = float(np.linalg.norm(r_span))
    frac = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0])[:, None]
    if r_len < 1e-9:
        return pelvis[..., None, :] + frac[None, ...] * span[..., None, :]

    # The rest spine's curvature, as offsets from the rest straight line, normalised by
    # the rest torso length so they are dimensionless fractions rather than metres.
    offsets = (np.stack([rest[3], rest[6], rest[9]], axis=0) - (r_pelvis + frac * r_span)) / r_len

    # Re-apply that same curvature to the tracked straight line, in the tracked torso's
    # own scale. Batch-safe: offsets are (3, 3) fractions, scaled by the per-frame length.
    tracked_len = np.linalg.norm(span, axis=-1, keepdims=True)  # (..., 1)
    tracked_frac = np.array([1.0 / 3.0, 2.0 / 3.0, 1.0])[..., None]
    straight = pelvis[..., None, :] + tracked_frac * span[..., None, :]
    return straight + offsets * tracked_len[..., None]


def anatomical_report(joints: np.ndarray) -> dict[str, object]:
    """Check the ordering a body must have, for tests and for pre-flight checks.

    ``joints`` is ``(24, 3)`` after :func:`mediapipe_to_smpl24`, i.e. y up. A scrambled
    mapping fails these; a correct one passes. This exists because every defect above
    produced geometry that *rendered* - it was only wrong, not broken - so a shape check
    is not enough and an ordering check is what catches it.
    """
    j = np.asarray(joints, dtype=np.float64)
    if j.shape != (N_SMPL_JOINTS, 3):
        raise ValueError(f"expected ({N_SMPL_JOINTS}, 3), got {j.shape}")
    y = j[:, 1]
    checks = {
        "head_above_pelvis": bool(y[15] > y[0]),
        "pelvis_above_knees": bool(y[0] > y[4] and y[0] > y[5]),
        "knees_above_ankles": bool(y[4] > y[7] and y[5] > y[8]),
        "shoulders_above_pelvis": bool(y[16] > y[0] and y[17] > y[0]),
        "spine_is_between": bool(y[0] < y[3] < y[6] < y[9]),
        # NOTE: "wrists below head" is deliberately NOT checked. Signers raise their
        # hands to face and above-head height as a matter of course, so that test
        # flags normal signing rather than a defect. The checks kept here are the
        # ones a scrambled mapping would violate: a body where the knees sit above
        # the hips, the spine runs backwards, or the head ends up at the wrists.
        # SMPL convention (verified against J_regressor @ v_template on the real
        # model): the subject's LEFT shoulder is at the LARGER x. So "left is left"
        # means the left-shoulder x exceeds the right-shoulder x. A flipped
        # inequality here would flag a correct, non-mirrored body as failing.
        "left_is_left": bool(j[16, 0] > j[17, 0]),
    }
    return {"checks": checks, "passed": all(checks.values()), "joints": j}


def assert_anatomical(joints: np.ndarray, *, label: str = "") -> None:
    """Raise unless the skeleton is ordered like a body."""
    rep = anatomical_report(joints)
    checks = rep.get("checks")
    if not rep.get("passed"):
        # `anatomical_report` is typed as dict[str, object]; narrow before use instead of
        # silencing the checker, because "checks" not being a mapping is a real failure mode.
        failed = (
            [k for k, v in checks.items() if not v]
            if isinstance(checks, dict)
            else ["<no checks mapping in report>"]
        )
        raise ValueError(
            f"{label + ': ' if label else ''}retargeted skeleton fails {failed}. The "
            "joint mapping or the y-flip is wrong; no geometry from this frame may be "
            "shown or used."
        )


def damp_depth_noise(joints: np.ndarray, *, depth_share: float = 0.25) -> np.ndarray:
    """Limit how much of each leg bone's length may lie along the depth (z) axis.

    **Why this exists.** MediaPipe recovers depth from a single 2-D image, and for the
    limbs it is close to a guess: it cannot tell a knee swung *forward* from a knee swung
    *sideways*, because both look identical in the image. The guesses are not merely noisy
    - they are too large. Measured on a real EmoSign clip, the raw hip->knee vector is
    ``[0.025, 0.176, -0.707]`` in image units: the depth is **four times** the vertical
    drop, for a joint at most one thigh-length away. After the first-pass damping the
    retarget input was still ``[0.032, -0.166, -0.256]`` - a thigh pointing further
    *forward* than *down*, which is a sitting stride, not a stance.

    **Why not just multiply z by a constant.** That was the first attempt and it is the
    wrong shape of correction. The error is *per bone*: a bone can be short with a huge
    spurious depth, or long with a small one, so scaling absolute z treats them the same.

    **Why the cap is measured against x and y only.** The obvious formulation - allow at
    most ``depth_share`` of the bone's *total* length along z - quietly defeats itself,
    because the bone length is inflated by the very depth being capped. A thigh reported
    as ``[0.033, -0.232, -0.795]`` has length 0.828, almost all of it the bad z, so
    ``0.25 * 0.828 = 0.207`` looks like a generous allowance while actually permitting the
    bone to stay depth-dominated. Instead the allowance is built from the **planar**
    length - ``hypot(x, y)``, both of which are image measurements and honest - so a bone
    that is genuinely long in the image may lean further out of plane, and a bone that is
    short may not. In practice this is a lean-angle limit: at ``depth_share = 0.25`` a leg
    bone may sit at most ~14 degrees out of the image plane.

    ``depth_share`` is that limit. Set 1.0 to disable.

    **The cap applies to the thigh and shin only - not the foot.** The foot bone runs
    *forward* from the ankle, so depth legitimately dominates it: measured on
    ``SMPLX_NEUTRAL`` the ankle->foot bone has a depth share of 1.58 on the left and 1.70
    on the right, against 0.025-0.079 for the thigh and shin. Capping the foot at the
    same 0.25 does not clean noise, it flattens a correct foot - it moved the model's own
    rest pose by 0.101 m, which is a 10 cm error injected into a pose that needed none.
    The foot is therefore excluded, and the pelvis-to-foot chain is capped at the ankle.

    The sign of z is kept, so a real forward step still reads as forward, just gentler.
    Applied **before** any rotation is derived from the skeleton, because the retargeter
    reads these positions: damping afterwards would correct the picture but not the frame
    it was drawn in.
    """
    j = np.asarray(joints, dtype=np.float64).copy()
    if depth_share >= 1.0:
        return j
    # Each entry: (child, parent). The root (pelvis) has no parent and is left alone -
    # it is the origin the rest is expressed against. (10, 7) and (11, 8), the feet, are
    # deliberately absent: see the docstring.
    chains = ((4, 1), (5, 2), (7, 4), (8, 5))
    # Solve for the allowed |z| given a planar length p: we want z <= depth_share * hypot(p, z),
    # which rearranges to |z| <= p * depth_share / sqrt(1 - depth_share^2).
    cap = depth_share / np.sqrt(1.0 - depth_share**2)
    for child, parent in chains:
        bone = j[..., child, :] - j[..., parent, :]
        planar = np.linalg.norm(bone[..., :2], axis=-1, keepdims=True)
        limit = cap * planar
        z = bone[..., 2:3]
        shrink = np.where(np.abs(z) > limit, limit / np.maximum(np.abs(z), 1e-12), 1.0)
        shrink = np.clip(shrink, 0.0, 1.0)
        j[..., child, 2] = j[..., parent, 2] + z[..., 0] * shrink[..., 0]
    return j


def normalise_to_rest_frame(
    joints: np.ndarray,
    *,
    rest: np.ndarray | None = None,
    target_span: float | None = None,
    leg_depth_damping: float = 0.25,
) -> np.ndarray:
    """Put tracked joints into a rest pose's frame and units.

    **This is the step the retargeter silently assumed was already done.**
    :func:`seam.avatar.synthesis.retarget_body` builds each joint's local rotation by
    comparing ``tracked[j] - tracked[parent]`` against ``rest[j] - rest[parent]``. That is
    a comparison of *directions*, so it looks scale-free - but only if the two skeletons
    are expressed in the same orientation and comparable proportions. The tracked joints
    come out of the landmark mapper in **image-pixel units**, with the limbs' depth (z) on
    a different scale from the rest pose's metres. Feeding those straight in asks the
    retargeter for enormous rotations, and the body renders lying on its back with its legs
    in the air - the "falling over" defect in the first M7 stimulus videos.

    ``rest`` is the pose to normalise *into*, and it must be the same one handed to
    :func:`retarget_body` - normally the joints of the model being skinned
    (:func:`seam.avatar.synthesis.model_rest_pose`). Defaulting to
    :func:`canonical_rest_pose` keeps the no-model path working, but it is a real
    mismatch when a model is loaded: that figure's torso is much longer (neck at
    ``y = 0.60`` against the model's ``0.108``), so tracking scaled to its height lands
    at the wrong proportions and the legs render too far apart.

    **What this does, in order:**

    1. Translate so the pelvis is at the origin.
    2. Scale uniformly so the body's height matches the rest pose's, which makes bone
       *lengths* comparable. One scale per clip (from the median frame), not per frame, so
       the avatar cannot pulse as tracking noise changes the apparent height.
    3. Rotate the whole skeleton so its torso axis (pelvis -> neck) points straight up,
       the way the rest pose's does. This removes camera tilt and does not invent a
       heading (rotation about the vertical axis is left alone).
    4. Cap how much of each leg bone's length lies along the depth (z) axis - see
       :func:`damp_depth_noise`. MediaPipe's recovered depth for the limbs is close to a
       guess and is far too large: the raw hip-to-knee depth was four times the vertical
       drop. Left alone, the legs splay into a wide straddle in every frame, because the
       retargeter faithfully reproduces a thigh that points forward instead of down.

    After this, ``retarget_body`` receives joints in the frame it was written for, and
    "no rotation" genuinely means "standing in the rest pose".

    The leg-depth capping of step 4 is applied **first**, before any rotation is derived,
    because the rotation is itself computed from these joint positions. Damping afterwards
    would correct the picture but not the frame it was drawn in.
    """
    j = np.asarray(joints, dtype=np.float64).copy()
    if j.shape[-2:] != (N_SMPL_JOINTS, 3):
        raise ValueError(f"expected (..., {N_SMPL_JOINTS}, 3), got {tuple(j.shape)}")

    # 0. damp the legs' tracked depth, before anything reads these positions.
    j = damp_depth_noise(j, depth_share=leg_depth_damping)

    # 1. pelvis to the origin
    j = j - j[..., 0:1, :]

    # 2. one uniform scale for the whole clip
    rest = np.asarray(rest if rest is not None else canonical_rest_pose(), dtype=np.float64)
    rest_span = float(rest[:, 1].max() - rest[:, 1].min())
    span = target_span if target_span is not None else rest_span
    flat = j.reshape(-1, N_SMPL_JOINTS, 3)
    if flat.shape[0] > 0:
        tracked = float(np.median(flat[:, :, 1].max(axis=1) - flat[:, :, 1].min(axis=1)))
    else:
        tracked = 0.0
    if tracked > 1e-9:
        j = j * (span / tracked)

    # 3. rotate each frame so the torso axis points up.
    #
    # The axis is pelvis -> neck, not pelvis -> spine3. spine3 is one of the joints
    # MediaPipe does not have, so the mapper *interpolates* it - and an interpolated joint
    # sitting near the pelvis makes the axis direction sensitive to small errors in the
    # two joints it is built from. The neck is derived from the shoulders, which are
    # measured directly and are far from the pelvis, so the axis it gives is stable. This
    # matters because the whole rotation is read off this one direction.
    rest_up = rest[NECK] - rest[0]
    rest_up = rest_up / np.linalg.norm(rest_up)
    out = j.copy()
    frames_flat = out.reshape(-1, N_SMPL_JOINTS, 3)
    for i in range(frames_flat.shape[0]):
        pelvis = frames_flat[i, 0]
        chest = frames_flat[i, NECK]
        up = chest - pelvis
        n = np.linalg.norm(up)
        if n < 1e-9:
            continue
        up = up / n
        axis = np.cross(up, rest_up)
        sin_t = float(np.linalg.norm(axis))
        cos_t = float(np.clip(np.dot(up, rest_up), -1.0, 1.0))
        if sin_t < 1e-9:
            if cos_t < 0:
                # Torso inverted: rotate 180 degrees about x, the only sensible upright
                # recovery when there is no defined side.
                axis = np.array([1.0, 0.0, 0.0])
                sin_t = 0.0
                cos_t = -1.0
            else:
                continue
        k = axis / max(sin_t, 1e-12)
        angle = float(np.arctan2(sin_t, cos_t))
        K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        frames_flat[i] = frames_flat[i] @ R.T

    # 4. the legs' depth was already damped in step 0; nothing more to do here.
    return out
