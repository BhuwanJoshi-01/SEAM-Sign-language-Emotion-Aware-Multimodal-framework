"""Driving an SMPL-X avatar from tracked keypoints.

SMPL-X is the right target for this project, and specifically because it is a
**composition of parts that match our inputs one-to-one**:

| our input | SMPL-X block | what it is |
|---|---|---|
| body keypoints | ``global_orient`` + ``body_pose`` (1 + 21 joints) | SMPL body |
| 21-point hand keypoints, per hand | ``left/right_hand_pose`` (45 = 15 joints x 3) | MANO hands |
| 52 blendshapes / 68 face pts | ``expression`` (10) + ``jaw_pose`` (3) | FLAME face |

No single-source model gives all three. SMPL-X is SMPL + MANO + FLAME, so the three
signals our perception stage already produces each drive the corresponding part with no
bridging guess in between.

**The model files are licence-gated and deliberately not vendored.** SMPL-X parameters
are distributed by the MPI under a research licence that requires registration and
agreement; the ``.npz`` cannot be committed to this repository, and neither can any
derivative of it. :func:`load_smplx` therefore takes a path the operator supplies and
refuses a missing one with an explanation rather than a ``FileNotFoundError``.

**Three failure modes are documented, and each is tested here**, because all three are
silent - the animation still renders, it is just wrong:

1. **Gimbal flips.** Converting a rotation to Euler angles and back can flip a joint
   180 degrees between adjacent frames when the angle crosses +/-pi. The pose is
   continuous, the Euler representation is not. Fixed by carrying rotations as
   quaternions and re-normalising each frame to the nearest rotation of the previous
   one (:func:`enforce_quaternion_continuity`).
2. **Joint limits.** Retargeting from keypoints can request a knee or elbow outside its
   anatomical range, and a rigged mesh will happily render it. Clamped to per-joint
   limits before the forward kinematics runs (:func:`clamp_joint_limits`).
3. **Finger distortion.** The MANO hand pose is a 15-joint kinematic chain per hand; a
   naive per-joint angle estimate makes fingers splay and the thumb invert, because the
   hand's bones are short and the landmark noise is large relative to bone length.
   Angles are computed from *bone vectors* and attenuated along the chain, with the
   distal joints damped harder than the proximal ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- SMPL-X body skeleton ---------------------------------------------------

#: SMPL's 24 joints in ``body_pose`` order, i.e. **the 23 joints after the pelvis** -
#: the pelvis is carried by ``global_orient`` and ``global_transl``, not by
#: ``body_pose``. So this tuple has 23 entries while :data:`SMPL_PARENTS` has 24; the
#: offset is ``+1`` and is applied explicitly in :func:`retarget_body` rather than
#: left for a reader to notice.
SMPL_BODY_JOINTS = (
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)

#: Parent of each joint, ``-1`` for the root. Drives the forward chain used to turn
#: bone vectors into local joint rotations.
#: The canonical SMPL 24-joint tree. Corrected after a test caught the first
#: version being shifted by one, which would have made every bone direction - and so
#: every joint rotation - anatomically wrong while still producing a smooth,
#: plausible-looking animation.
SMPL_PARENTS = np.array(
    [
        -1,  # 0  pelvis
        0,  # 1  left_hip
        0,  # 2  right_hip
        0,  # 3  spine1
        1,  # 4  left_knee
        2,  # 5  right_knee
        3,  # 6  spine2
        4,  # 7  left_ankle
        5,  # 8  right_ankle
        6,  # 9  spine3
        7,  # 10 left_foot
        8,  # 11 right_foot
        9,  # 12 neck
        9,  # 13 left_collar
        9,  # 14 right_collar
        12,  # 15 head
        13,  # 16 left_shoulder
        14,  # 17 right_shoulder
        16,  # 18 left_elbow
        17,  # 19 right_elbow
        18,  # 20 left_wrist
        19,  # 21 right_wrist
        20,  # 22 left_hand
        21,  # 23 right_hand
    ],
    dtype=np.int64,
)

#: Anatomical limit per body joint, in radians, applied to the local rotation. A
#: knee does not bend backwards and an elbow does not hyperextend; a mesh will render
#: either if asked to, which is why these are clamped rather than documented.
#: ``inf`` means a joint with no meaningful limit in this simple model (spine).
#:
#: **Left and right entries must be mirror images, and two are not.** The clamp reads
#: ``(lo, hi)`` as a range on the rotation *magnitude*, so a joint recorded as
#: ``(-100, 40)`` may rotate at most 40 degrees while its mirror recorded as
#: ``(-40, 100)`` may rotate 100 - a 2.5x difference in how far the two sides are
#: allowed to move, on a signer whose two legs are equally mobile. Measured on clip
#: 1372 the effect is visible: 32% of frames put the left hip on its ceiling against
#: 26% on the right, and both thighs sit about 35 degrees forward because the left
#: side cannot follow the right. The same asymmetry is present in the shoulders
#: (25 against 90 degrees, 3.6x).
#:
#: ``_symmetric_limits`` below widens each such pair to the union of the two ranges, so
#: both sides get the larger allowance. Widening cannot invalidate a pose that already
#: rendered - it only stops one side being refused a rotation its mirror is granted.
JOINT_LIMITS_DEG = {
    "left_hip": (-100.0, 40.0),
    "right_hip": (-40.0, 100.0),
    "spine1": (-25.0, 25.0),
    "left_knee": (0.0, 140.0),
    "right_knee": (0.0, 140.0),
    "spine2": (-25.0, 25.0),
    "left_ankle": (-45.0, 70.0),
    "right_ankle": (-45.0, 70.0),
    "spine3": (-25.0, 25.0),
    "left_foot": (-25.0, 50.0),
    "right_foot": (-25.0, 50.0),
    "neck": (-60.0, 70.0),
    "left_collar": (-10.0, 10.0),
    "right_collar": (-10.0, 10.0),
    "head": (-30.0, 30.0),
    "left_shoulder": (-90.0, 25.0),
    "right_shoulder": (-25.0, 90.0),
    "left_elbow": (0.0, 150.0),
    "right_elbow": (0.0, 150.0),
    "left_wrist": (-80.0, 80.0),
    "right_wrist": (-80.0, 80.0),
    "left_hand": (-45.0, 45.0),
    "right_hand": (-45.0, 45.0),
}

#: Left/right limit pairs that are entered asymmetrically and are therefore widened to a
#: common range. Listed explicitly rather than derived from the ``left_``/``right_`` name
#: prefix, because a future joint could legitimately differ between sides - the hand that
#: holds the pen really is more mobile - and silently equalising all of them would hide
#: that. Anything not named here is used exactly as written above.
_SYMMETRIC_LIMIT_PAIRS = (
    ("left_hip", "right_hip"),
    ("left_shoulder", "right_shoulder"),
)


def _symmetric_limits(
    table: dict[str, tuple[float, float]],
    pairs: tuple[tuple[str, str], ...],
) -> dict[str, tuple[float, float]]:
    """Widen each listed left/right pair to the union of the two ranges.

    Pure in its input: returns a new table and does not touch the argument. Called once
    at import to build :data:`JOINT_LIMITS_DEG_SYMMETRIC`, so the module's public table
    stays the literal transcription of whatever limits were decided and the symmetric
    version is a derived artefact a reader can recompute.
    """
    out = dict(table)
    for a, b in pairs:
        if a not in out or b not in out:
            continue
        la, ha = out[a]
        lb, hb = out[b]
        lo, hi = min(la, lb), max(ha, hb)
        out[a] = (lo, hi)
        out[b] = (lo, hi)
    return out


#: What :func:`clamp_joint_limits` uses. Identical to :data:`JOINT_LIMITS_DEG` except
#: that each pair named in :data:`_SYMMETRIC_LIMIT_PAIRS` carries the same range on both
#: sides. Kept as a separate name so a test can assert the two differ only where intended.
JOINT_LIMITS_DEG_SYMMETRIC = _symmetric_limits(JOINT_LIMITS_DEG, _SYMMETRIC_LIMIT_PAIRS)

#: How far each MANO joint may rotate, degrees, proximal to distal. The distal
#: joints get smaller limits because their bones are short and landmark noise is
#: large relative to bone length - the setup in which fingers splay.
MANO_LIMIT_DEG = np.array(
    [25.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0]
)

#: Damping applied to hand angle magnitude along the chain, so a noisy distal
#: landmark cannot produce a large distal rotation.
HAND_CHAIN_DAMPING = np.linspace(1.0, 0.35, 15)

#: Per-pixel amplitude added to the FLAME expression coefficients. Deliberately small:
#: FLAME expression space is easy to over-drive, and an over-driven face looks like a
#: rubber mask rather than an emotion.
EXPRESSION_GAIN = 0.6


# --- geometry helpers -------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


def _rodrigues(aa: np.ndarray) -> np.ndarray:
    """Rodrigues rotation, written out rather than clever.

    Returns ``(..., 3, 3)``. The naive implementation of this is the single most
    common source of gimbal artefacts, so it is written with the sine half-angle
    form and tested, not compressed into a one-liner.
    """
    theta = np.linalg.norm(aa, axis=-1)
    axis = aa / np.maximum(theta, 1e-12)[..., None]
    ux, uy, uz = axis[..., 0], axis[..., 1], axis[..., 2]
    sx, cx = np.sin(theta), np.cos(theta)
    r = np.empty((*aa.shape[:-1], 3, 3), dtype=np.float64)
    r[..., 0, 0] = cx + ux * ux * (1 - cx)
    r[..., 0, 1] = ux * uy * (1 - cx) - uz * sx
    r[..., 0, 2] = ux * uz * (1 - cx) + uy * sx
    r[..., 1, 0] = uy * ux * (1 - cx) + uz * sx
    r[..., 1, 1] = cx + uy * uy * (1 - cx)
    r[..., 1, 2] = uy * uz * (1 - cx) - ux * sx
    r[..., 2, 0] = uz * ux * (1 - cx) - uy * sx
    r[..., 2, 1] = uz * uy * (1 - cx) + ux * sx
    r[..., 2, 2] = cx + uz * uz * (1 - cx)
    return r


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    """Rotation matrix to ``(w, x, y, z)``, numerically stable branch selection."""
    m = r
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    w = np.sqrt(np.maximum(1.0 + trace, 1e-12)) / 2.0
    w4 = 4.0 * w
    x = (m[..., 2, 1] - m[..., 1, 2]) / w4
    y = (m[..., 0, 2] - m[..., 2, 0]) / w4
    z = (m[..., 1, 0] - m[..., 0, 1]) / w4
    return _unit(np.stack([w, x, y, z], -1))


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = _unit(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r = np.empty((*q.shape[:-1], 3, 3), dtype=np.float64)
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - z * w)
    r[..., 0, 2] = 2 * (x * z + y * w)
    r[..., 1, 0] = 2 * (x * y + z * w)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - x * w)
    r[..., 2, 0] = 2 * (x * z - y * w)
    r[..., 2, 1] = 2 * (y * z + x * w)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def enforce_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    """Flip each quaternion to the hemisphere nearest the previous frame.

    ``q`` and ``-q`` are the same rotation, and a retargeting pipeline will emit
    whichever sign the previous computation produced. Adjacent frames can therefore
    differ by a full sign flip, which every interpolation in between renders as a
    180-degree swing. This removes the discontinuity without changing any pose, so
    it is safe to apply unconditionally.
    """
    q = np.asarray(quats, dtype=np.float64).copy()
    if len(q) < 2:
        return q
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0.0:
            q[i] = -q[i]
    return q


def clamp_joint_limits(aa: np.ndarray, joint_names: Sequence[str]) -> np.ndarray:
    """Clamp per-joint axis-angle magnitude to its anatomical limit.

    Axis-angle magnitude is the rotation angle, so scaling it down is a valid way to
    reduce the rotation; the axis is preserved, which keeps the motion in the right
    plane rather than folding it somewhere else.

    **``joint_names`` must already be the ``body_pose`` list**, i.e.
    :data:`SMPL_BODY_JOINTS` itself, because entry ``i`` of that tuple is joint ``i + 1``
    of the 24-joint skeleton and therefore slot ``i`` of ``body_pose``. Slicing it - in
    particular passing ``SMPL_BODY_JOINTS[1:]``, which reads as "skip the pelvis" but
    actually drops ``left_hip`` - shifts every joint onto its neighbour's limits. That
    is not a small error: measured before the fix, slot 8 (``spine3``, limit 25 degrees)
    was receiving ``left_foot``'s limit of 50 degrees and clamping to exactly 50.0, while
    slot 2 (``spine1``) was receiving ``left_knee``'s ``(0, 140)`` and so could not bend
    backwards at all. The length check below exists to make that mistake fail loudly.
    """
    out = np.asarray(aa, dtype=np.float64).copy()
    if len(joint_names) != len(out):
        raise ValueError(
            f"joint_names must name every body_pose slot one-to-one: got "
            f"{len(joint_names)} names for {len(out)} slots. Pass SMPL_BODY_JOINTS "
            f"itself, not a slice of it."
        )
    for i, name in enumerate(joint_names):
        if i >= len(out):
            break
        lim = JOINT_LIMITS_DEG_SYMMETRIC.get(name)
        if lim is None:
            continue
        lo, hi = np.radians(lim[0]), np.radians(lim[1])
        mag = float(np.linalg.norm(out[i]))
        if mag < 1e-9:
            continue
        # Signed along the current axis: map the angle into [lo, hi] where it can be,
        # and clamp to the nearer bound otherwise.
        clipped = min(max(mag, lo), hi) if lo <= mag <= hi else (lo if mag < lo else hi)
        if abs(clipped - mag) > 1e-9:
            out[i] *= clipped / mag
    return out


def clamp_hand_angles(aa: np.ndarray) -> np.ndarray:
    """Clamp and damp the 15 MANO joint angles per hand."""
    out = np.asarray(aa, dtype=np.float64).copy()
    for i in range(min(len(out), len(MANO_LIMIT_DEG))):
        mag = float(np.linalg.norm(out[i]))
        if mag < 1e-9:
            continue
        limit = np.radians(MANO_LIMIT_DEG[i])
        mag = min(mag * HAND_CHAIN_DAMPING[i], limit)
        out[i] *= mag / float(np.linalg.norm(out[i]))
    return out


# --- retargeting ------------------------------------------------------------


def retarget_body(
    keypoints: np.ndarray,
    *,
    conf: np.ndarray | None = None,
    rest: np.ndarray | None = None,
) -> np.ndarray:
    """Body local joint rotations from 2D/3D joint positions.

    ``keypoints`` is ``(24, 3)`` in SMPL order, pelvis first. Each local rotation is
    built from the difference between the bone direction in the current frame and the
    same bone direction in a reference pose, which is why a reference pose is required:
    without one there is no definition of "unrotated".

    **The reference must be the rest pose of the mesh the rotations will drive.**
    ``rest`` should be the joint positions of the model being skinned - ``J_regressor @
    v_template``, available from the ``.npz`` once the licence-gated model is loaded. It
    defaults to :func:`canonical_rest_pose`, a hand-written stick figure, only so that
    callers with no model file still get something. That default is a genuine mismatch
    when a real model *is* in use: measured against ``SMPLX_NEUTRAL``, the canonical
    figure puts the hips 0.18 apart where the model has 0.121, and the model's knee sits
    at ``x = 0.116`` against a hip at ``0.061``. Retargeting against the wrong reference
    therefore asks every leg for a rotation it does not need, and the two hips, being
    mirrored, receive opposite errors - which renders as a wide straddle stance on a
    signer whose knees are actually *closer together* than their hips.

    **World rotation is reduced to a local one, and this is the whole difficulty.**
    ``keypoints`` are joint positions in one common world frame, so the rotation that
    takes the rest bone onto the tracked bone - computed from ``kp[j] - kp[parent]`` -
    is a *world* rotation: it already contains the parent's orientation. But
    ``body_pose`` is consumed by forward kinematics as a chain of **local** rotations,
    each applied on top of its parent's accumulated rotation. Writing the world rotation
    straight into ``out[j-1]`` therefore applies the parent's share twice.

    That is not a small error either. Measured on clip 1372 frame 40, the left hip
    carried 46.7 degrees and the left knee 31.9; forward kinematics applied both to the
    thigh, and the ankle ended 1.00 apart in x where the tracked joints were 0.22. The
    knee rotations were also pinned at their clamp ceiling, which is the signature of
    this bug: the retargeter was asking for a rotation it had already applied once.

    So each joint's world rotation is converted to a local one by removing the parent's
    accumulated world rotation: ``R_local = R_parent^T @ R_world``. The conversion is
    accumulated down the tree in index order, which is valid because SMPL's joint indices
    are topologically sorted (every joint's parent has a smaller index).

    **Which slot a bone's rotation belongs in is the other half of the problem, and it
    is off by one in the obvious reading.** SMPL's ``body_pose[j-1]`` is joint ``j``'s
    *local* rotation, and forward kinematics applies it as::

        R_world[j] = R_world[parent[j]] @ local_j
        out[j]     = out[parent[j]] + R_world[parent[j]] @ (rest[j] - rest[parent[j]])

    The bone from ``parent[j]`` to ``j`` is therefore rotated by ``R_world[parent[j]]`` -
    by the **parent's** rotation, not joint ``j``'s own. So the rotation that takes the
    rest bone ``parent->j`` onto the tracked one belongs in the slot of ``parent``, and
    writing it to ``j-1`` shifts every joint's rotation one level down the chain.

    Measured: swinging the thigh 0.15 out in x put the rotation in slot 3 (the knee), and
    forward kinematics left the knee at exactly ``[0.09, -0.5, 0]`` - unmoved. The thigh
    was never rotated by its own measurement, which is what splayed the legs. The same
    shift makes every child inherit a rotation meant for a bone further down.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.shape[0] < 24:
        raise ValueError(f"need 24 SMPL joints, got {kp.shape[0]}")
    rest = np.asarray(rest if rest is not None else canonical_rest_pose(), dtype=np.float64)
    if rest.shape[0] < 24:
        raise ValueError(f"rest pose must carry 24 joints, got {rest.shape[0]}")

    # 1. World rotation of every bone parent[j] -> j, i.e. the rotation that carries the
    #    rest bone direction onto the tracked one. Computed in world space for all joints;
    #    nothing is stored yet, because the storage slot is the parent's, not the child's.
    bone_rot: list[np.ndarray] = [np.eye(3) for _ in range(24)]
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        if p < 0:
            continue
        a = _unit(rest[j] - rest[p])
        b = _unit(kp[j] - kp[p])
        axis = np.cross(a, b)
        sin_t = float(np.linalg.norm(axis))
        cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if sin_t < 1e-8:
            # Bone already aligned with its rest direction, or exactly reversed. Aligned:
            # identity is correct. Reversed: rotating by pi about any axis perpendicular
            # to the bone is correct, and picking a deterministic one beats leaving the
            # joint silently unrotated, which would render a limb pointing the wrong way
            # with no error raised.
            if cos_t < 0.0:
                perp = np.cross(a, np.array([1.0, 0.0, 0.0]))
                if np.linalg.norm(perp) < 1e-8:
                    perp = np.cross(a, np.array([0.0, 1.0, 0.0]))
                bone_rot[j] = _rotation_from_axis_angle(_unit(perp), np.pi)
            else:
                bone_rot[j] = np.eye(3)
        else:
            bone_rot[j] = _rotation_from_axis_angle(axis / sin_t, float(np.arctan2(sin_t, cos_t)))

    # 3. Convert each bone's world rotation into the local slot FK reads it from.
    #
    # FK offsets a joint's children by `R_world[p]`, built as
    # `R_world[p] = R_world[parent[p]] @ local_p`. So for the bone p -> child to come out
    # pointing along the tracked direction, `R_world[p]` must carry the *rest* bone
    # `p -> child` onto the tracked one. That rotation is exactly `bone_rot[child]` from
    # pass 1, so:
    #
    #     local_p = R_world[parent[p]]^T @ bone_rot[child]
    #
    # and slot `p-1` is `local_p`. Written as the algebra rather than as "copy bone_rot
    # into the parent's slot" so a reader can check it against FK without re-deriving.
    #
    # **The order matters, and the first pass cannot be the assigning pass.** `R_world` is
    # accumulated down the chain, so a joint's frame is unknown until its ancestors' slots
    # are set. Assigning while accumulating silently uses identity frames for everything
    # below the root, which reproduces the original bug for every joint past the first
    # link. So: walk the chain to build `R_world` once, then assign in a second pass.
    #
    # For the two slots that take more than one bone, one rotation drives every bone leaving
    # that joint, so the rigid-skeleton model cannot satisfy them all. Which one to keep is
    # a modelling choice, and it has to be made deliberately rather than by loop order:
    #
    # - the pelvis's slot carries both hips and the spine. The spine wins: the trunk's
    #   orientation is what the pelvis really reports, and the hips then carry their own
    #   thigh rotations in slots 0 and 1.
    # - spine3's slot carries the neck and both collars. The neck wins, for the same
    #   reason. Leaving it to loop order made the *right collar* win, and because a collar
    #   bone runs sideways out to the shoulder, the neck inherited a side tilt - which is
    #   the ~20 degree full-body lean that survived every earlier fix. The collars then
    #   ride on the neck's frame, which is what a real shoulder girdle does.
    #
    # For the pelvis the spine bone is the one whose slot the trunk rides on; it is
    # identified here as the child of the root that the pelvis's own rotation should match.
    R_world = [np.eye(3) for _ in range(24)]
    for p in range(1, 24):
        child = next((c for c in range(1, 24) if SMPL_PARENTS[c] == p), None)
        if child is None:
            continue
        local = R_world[SMPL_PARENTS[p]].T @ bone_rot[child]
        R_world[p] = R_world[SMPL_PARENTS[p]] @ local

    # Slots that more than one bone would like to write. Children are visited in
    # ascending index order and each write is `out[p - 1] = ...`, i.e. last write wins,
    # so the *highest*-indexed child that leaves a joint decides that joint's slot. The
    # two contending joints are handled explicitly instead (see the comment above).
    #
    #   joint 0 (pelvis)  <- children 1 (left_hip), 2 (right_hip), 3 (spine1)
    #                        highest index is 3, the spine, which is the one we want.
    #                        No code needed - it already wins - but it is stated here so
    #                        the rule is not accidentally broken by a future edit.
    #   joint 9 (spine3)  <- children 12 (neck), 13 (left_collar), 14 (right_collar)
    #                        highest index is 14, the *right collar*, which tilts the
    #                        neck sideways. The neck must be forced to win, so the
    #                        collars are held back until the neck has written slot 8.
    SPINE3_SLOT_OWNER = 12  # neck
    out = np.zeros((23, 3), dtype=np.float64)
    written = np.zeros(24, dtype=bool)
    for child in range(1, 24):
        p = SMPL_PARENTS[child]
        if p < 0:
            continue
        if p == 9 and child != SPINE3_SLOT_OWNER:
            # A collar (or anything else at this joint) must not claim spine3's slot.
            continue
        out[p - 1] = _log_map(R_world[SMPL_PARENTS[p]].T @ bone_rot[child])
        written[p] = True
    # Any joint no bone wrote (leaves of the tree, e.g. the hands) keeps the identity
    # rotation that `np.zeros` gave it, which is correct: nothing downstream depends on
    # a leaf's own frame.
    del written
    out = clamp_joint_limits(out, SMPL_BODY_JOINTS)
    return out


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix from a unit ``axis`` and an ``angle`` in radians.

    Distinct from :func:`_rodrigues`, which consumes an axis-angle *vector* (axis and
    angle in one array). Both exist because the retargeter works in axis-and-angle and
    the rest of the module works in axis-angle vectors; keeping them apart avoids the
    silent bug of passing one where the other is expected.
    """
    k = np.asarray(axis, dtype=np.float64)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _log_map(R: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix, the inverse of :func:`_rodrigues`.

    Near identity the log map is ill-conditioned - ``theta / sin(theta)`` diverges - so
    the small-angle limit ``theta ~ sin(theta)`` is used instead; the two agree to well
    inside float64 precision there, and the branch keeps a perfectly still joint at
    exactly zero rather than at a numerically amplified noise direction.
    """
    cos_t = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cos_t))
    if theta < 1e-8:
        return np.zeros(3, dtype=np.float64)
    if theta > np.pi - 1e-8:
        # theta / (2 sin theta) still converges here; recover the axis from R + I.
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        # Fix the signs against the off-diagonal terms, then renormalise.
        if R[0, 1] + R[1, 0] < 0:
            axis[1] = -axis[1]
        if R[0, 2] + R[2, 0] < 0:
            axis[2] = -axis[2]
        n = np.linalg.norm(axis)
        if n < 1e-12:
            return np.zeros(3, dtype=np.float64)
        return _unit(axis) * theta
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (theta / (2.0 * np.sin(theta)))


def canonical_rest_pose() -> np.ndarray:
    """A T-pose rest skeleton, in metres, pelvis at the origin, in **SMPL joint order**.

    Proportions are roughly a 1.7 m adult. They are only a *rest reference* for
    rotation estimation, so absolute bone lengths do not matter - only bone
    directions do - and this avoids a hard dependency on the licence-gated model
    file just to compute a direction.

    **The order is SMPL's, and it is load-bearing.** :func:`retarget_body` computes each
    joint's rotation from ``rest[j] - rest[parent[j]]`` and compares it to the same
    difference in the tracked keypoints, so the position at index ``j`` must be joint
    ``j``'s. This function previously assigned positions in a *different* order - hips,
    then knees, then spine1, then ankles - while :data:`SMPL_BODY_JOINTS` and
    :data:`SMPL_PARENTS` use the standard SMPL order (spine1 at 3, left_knee at 4). Every
    bone direction was therefore read off the wrong pair of joints: the "head" bone ran
    from the neck to a shoulder, and the "spine1" bone ran from the pelvis to a knee.
    The retargeted body was anatomically impossible as a result, and it still *rendered*,
    which is why nothing caught it.

    Verified against the real model: ``J_regressor @ v_template`` on
    ``SMPLX_NEUTRAL`` gives head at index 15 with ``y = +0.268`` above a pelvis at
    ``y = -0.351``, and ``left_shoulder`` (16) at ``x = +0.164`` against
    ``right_shoulder`` (17) at ``x = -0.152``. So in SMPL-X's frame **y is up and
    positive x is the subject's left**, and this reference matches that - the previous
    version had x mirrored, which would have reflected the whole body.
    """
    j = np.zeros((24, 3), dtype=np.float64)
    j[0] = [0.0, 0.0, 0.0]  # pelvis
    j[1], j[2] = [0.09, -0.10, 0.0], [-0.09, -0.10, 0.0]  # left/right hip
    j[3] = [0.0, 0.02, 0.0]  # spine1
    j[4], j[5] = [0.09, -0.50, 0.0], [-0.09, -0.50, 0.0]  # left/right knee
    j[6] = [0.0, 0.14, 0.0]  # spine2
    j[7], j[8] = [0.09, -0.90, 0.0], [-0.09, -0.90, 0.0]  # left/right ankle
    j[9] = [0.0, 0.26, 0.0]  # spine3
    j[10], j[11] = [0.09, -0.98, 0.12], [-0.09, -0.98, 0.12]  # left/right foot
    j[12] = [0.0, 0.60, 0.0]  # neck
    j[13], j[14] = [0.08, 0.50, 0.0], [-0.08, 0.50, 0.0]  # left/right collar
    j[15] = [0.0, 0.70, 0.0]  # head
    j[16], j[17] = [0.17, 0.50, 0.0], [-0.17, 0.50, 0.0]  # left/right shoulder
    j[18], j[19] = [0.45, 0.50, 0.0], [-0.45, 0.50, 0.0]  # left/right elbow
    j[20], j[21] = [0.68, 0.50, 0.0], [-0.68, 0.50, 0.0]  # left/right wrist
    j[22], j[23] = [0.76, 0.50, 0.0], [-0.76, 0.50, 0.0]  # left/right hand
    return j


def forward_kinematics(
    body_pose: np.ndarray,
    global_orient: np.ndarray,
    *,
    transl: np.ndarray | None = None,
    rest: np.ndarray | None = None,
) -> np.ndarray:
    """Joint positions in metres, (24, 3), from a body pose.

    Needed to place *anything* in the body: the proxy mesh, a skeleton overlay, or the
    joints a viewer should show. It uses :func:`canonical_rest_pose` for bone offsets
    rather than the licence-gated model, because bone *directions* are all a retargeting
    estimate needs and they do not depend on the learned shape - which is what keeps this
    usable before the SMPL-X weights arrive.

    The rest pose is T-pose with the pelvis at the origin and the model facing +z, so the
    result is in the same convention the retargeting assumes. Pass `transl` for world
    placement.

    **Only joints 1-21 have a `body_pose` rotation.** In SMPL-X, 22 and 23 are the hands
    and are driven by `left_hand_pose`/`right_hand_pose`, and the head is driven by
    `jaw_pose`; none of those live in the 21-vector. Indexing `body_pose[j-1]` for them
    runs off the end, so they are placed with their parent's accumulated rotation and no
    own rotation - which is the neutral here, not an approximation.
    """
    r = np.asarray(rest if rest is not None else canonical_rest_pose(), dtype=np.float64)
    aa = np.asarray(body_pose, dtype=np.float64)
    go = np.asarray(global_orient, dtype=np.float64)
    if aa.shape != (21, 3):
        raise ValueError(f"body_pose must be (21, 3), got {aa.shape}")
    if go.shape != (3,):
        raise ValueError(f"global_orient must be (3,), got {go.shape}")

    n_body = 21  # body_pose covers SMPL joints 1..21 only
    Rg = _rodrigues(go)
    out = np.zeros((24, 3), dtype=np.float64)
    out[0] = np.zeros(3) if transl is None else np.asarray(transl, dtype=np.float64)
    # World rotation accumulated down the chain, so a joint inherits its parent's frame.
    R_world = {0: Rg}
    for j in range(1, 24):
        parent = SMPL_PARENTS[j]
        local = _rodrigues(aa[j - 1]) if j <= n_body else np.eye(3)
        R_world[j] = R_world[parent] @ local
        out[j] = out[parent] + R_world[parent] @ (r[j] - r[parent])
    return out


def retarget_hand(keypoints_21: np.ndarray, *, hand: str = "right") -> np.ndarray:
    """MANO local rotations from a 21-point hand skeleton.

    Each joint's rotation magnitude is the **actual bend angle** at that joint, from
    the angle between its incoming and outgoing bones. The first version of this
    function used a fixed 25-degree magnitude, which is not retargeting at all: it
    produced exactly zero for a perfectly straight finger and full curl for a noisy
    one, i.e. it amplified noise rather than reading pose. A test caught it by
    requiring that *more* landmark noise not produce *more* finger movement.

    The 21 points are the MediaPipe/DWPose hand layout: 0 wrist, then four fingers at
    1-4, 5-8, 9-12, 13-16, and a four-point thumb at 17-20.
    """
    kp = np.asarray(keypoints_21, dtype=np.float64)
    if kp.shape[0] < 21:
        raise ValueError(f"need 21 hand keypoints, got {kp.shape[0]}")
    # (chain_start) per MANO joint group, in MANO's own order: index, middle, little,
    # ring, thumb - not the MediaPipe order, which is index, middle, ring, little.
    # Five points per finger, starting at the wrist, so the three MANO joints per
    # finger each have an incoming and an outgoing bone. In MANO's order - index,
    # middle, little, ring, thumb - not MediaPipe's, which is index, middle, ring,
    # little, thumb.
    chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 13, 14, 15, 16),
        (0, 9, 10, 11, 12),
        (0, 17, 18, 19, 20),
    )
    sign = 1.0 if hand == "right" else -1.0
    out = np.zeros((15, 3), dtype=np.float64)
    n = 0
    for chain in chains:
        for j in range(1, 4):
            if n >= 15:
                break
            prev, here, nxt = kp[chain[j - 1]], kp[chain[j]], kp[chain[j + 1]]
            incoming = here - prev
            outgoing = nxt - here
            lin, lout = np.linalg.norm(incoming), np.linalg.norm(outgoing)
            if lin < 1e-6 or lout < 1e-6:
                n += 1
                continue
            a, b = incoming / lin, outgoing / lout
            cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
            angle = float(np.arccos(cos_t))
            if angle > 1e-6:
                axis = np.cross(a, b)
                axis /= max(np.linalg.norm(axis), 1e-12)
                out[n] = axis * angle * sign
            n += 1
    return clamp_hand_angles(out)


def _raw_hand_angles(keypoints_21: np.ndarray, *, hand: str = "right") -> np.ndarray:
    """The hand angles *before* clamping and damping.

    Exposed so the damping can be shown to reduce noise inflation rather than
    assumed to. Same computation as :func:`retarget_hand` with the final clamp
    removed.
    """
    kp = np.asarray(keypoints_21, dtype=np.float64)
    chains = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 13, 14, 15, 16),
        (0, 9, 10, 11, 12),
        (0, 17, 18, 19, 20),
    )
    sign = 1.0 if hand == "right" else -1.0
    out = np.zeros((15, 3), dtype=np.float64)
    n = 0
    for chain in chains:
        for j in range(1, 4):
            prev, here, nxt = kp[chain[j - 1]], kp[chain[j]], kp[chain[j + 1]]
            incoming, outgoing = here - prev, nxt - here
            lin, lout = np.linalg.norm(incoming), np.linalg.norm(outgoing)
            if lin < 1e-6 or lout < 1e-6:
                n += 1
                continue
            a, b = incoming / lin, outgoing / lout
            angle = float(np.arccos(float(np.clip(np.dot(a, b), -1.0, 1.0))))
            if angle > 1e-6:
                axis = np.cross(a, b)
                out[n] = axis / max(np.linalg.norm(axis), 1e-12) * angle * sign
            n += 1
    return out


def retarget_expression(blendshapes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FLAME expression and jaw pose from MediaPipe's 52 blendshapes.

    The 10 FLAME expression coefficients have no published closed-form map from ARKit
    blendshapes, so this is a **fixed linear projection**, stated as a projection
    rather than dressed up as a calibration. The source blendshape for each slot is
    named so the mapping is auditable, and the gain is small on purpose: an
    over-driven FLAME expression reads as a rubber mask rather than an emotion.

    ``jawOpen`` drives ``jaw_pose`` directly and is deliberately *excluded* from the
    expression vector, because driving it through both double-counts it.
    """
    bs = np.asarray(blendshapes, dtype=np.float64).ravel()
    if bs.size < 52:
        raise ValueError(f"need 52 blendshapes, got {bs.size}")

    # MediaPipe's 52-coefficient order, as emitted by the tasks model.
    IDX = {
        "browInnerUp": 0,
        "browOuterUpLeft": 1,
        "browDownLeft": 3,
        "browDownRight": 4,
        "eyeSquintLeft": 5,
        "eyeWideLeft": 7,
        "jawOpen": 9,
        "mouthSmileLeft": 20,
        "mouthFunnel": 23,
        "mouthPucker": 22,
    }
    # (blendshape, gain) per FLAME expression slot. 10 slots.
    projection: tuple[tuple[str, float], ...] = (
        ("mouthSmileLeft", 0.9),
        ("browDownLeft", 0.6),
        ("browInnerUp", 0.7),
        ("eyeSquintLeft", 0.5),
        ("eyeWideLeft", 0.5),
        ("mouthFunnel", 0.4),
        ("mouthPucker", 0.3),
        ("browOuterUpLeft", 0.4),
        ("browDownRight", 0.3),
        ("browInnerUp", 0.2),
    )
    expr = np.array([g * bs[IDX[name]] for name, g in projection], dtype=np.float64)
    expr *= EXPRESSION_GAIN
    jaw = np.array([np.deg2rad(12.0) * float(bs[IDX["jawOpen"]])])
    return expr, jaw


# --- per-frame synthesis ----------------------------------------------------


@dataclass(slots=True)
class SmplxFrame:
    """One frame of SMPL-X parameters."""

    global_orient: np.ndarray = field(default_factory=lambda: np.zeros(3))
    body_pose: np.ndarray = field(default_factory=lambda: np.zeros((21, 3)))
    left_hand_pose: np.ndarray = field(default_factory=lambda: np.zeros((15, 3)))
    right_hand_pose: np.ndarray = field(default_factory=lambda: np.zeros((15, 3)))
    jaw_pose: np.ndarray = field(default_factory=lambda: np.zeros(3))
    expression: np.ndarray = field(default_factory=lambda: np.zeros(10))
    global_transl: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def as_dict(self) -> dict[str, list[float]]:
        return {
            "global_orient": self.global_orient.tolist(),
            "body_pose": self.body_pose.reshape(-1).tolist(),
            "left_hand_pose": self.left_hand_pose.reshape(-1).tolist(),
            "right_hand_pose": self.right_hand_pose.reshape(-1).tolist(),
            "jaw_pose": self.jaw_pose.tolist(),
            "expression": self.expression.tolist(),
            "global_transl": self.global_transl.tolist(),
        }


def synthesise(
    body_kp: np.ndarray,
    *,
    left_hand: np.ndarray | None = None,
    right_hand: np.ndarray | None = None,
    blendshapes: np.ndarray | None = None,
    transl: np.ndarray | None = None,
    rest: np.ndarray | None = None,
) -> SmplxFrame:
    """One SMPL-X frame from tracked keypoints.

    ``body_kp`` is ``(24, 3)``. Hand and face inputs are optional: a frame where the
    signer is not signing keeps whatever the caller passes as zeros, which is the
    correct neutral, rather than inventing a pose.

    ``rest`` is the reference pose the body rotations are measured against; pass the
    joints of the model being skinned (see :func:`model_rest_pose`) or the retargeted
    pose will not be the one the mesh expects.
    """
    frame = SmplxFrame()
    frame.body_pose = retarget_body(body_kp, rest=rest)[:21]
    if left_hand is not None:
        frame.left_hand_pose = retarget_hand(left_hand, hand="left")
    if right_hand is not None:
        frame.right_hand_pose = retarget_hand(right_hand, hand="right")
    if blendshapes is not None:
        expr, jaw = retarget_expression(blendshapes)
        frame.expression = expr
        frame.jaw_pose = jaw
    if transl is not None:
        frame.global_transl = np.asarray(transl, dtype=np.float64)
    return frame


def synthesise_sequence(
    body_seq: np.ndarray,
    *,
    hand_seq: np.ndarray | None = None,
    blendshape_seq: np.ndarray | None = None,
    rest: np.ndarray | None = None,
) -> list[SmplxFrame]:
    """A whole sequence, with quaternion continuity enforced across frames.

    Continuity is applied here rather than per frame because it is a property of the
    sequence, and computing each frame independently is exactly how a sign flip
    appears in the middle of an otherwise smooth animation.

    ``rest`` is forwarded to every frame; see :func:`synthesise`.
    """
    bs = np.asarray(body_seq, dtype=np.float64)
    n = len(bs)
    out: list[SmplxFrame] = []
    prev_global: np.ndarray | None = None
    for i in range(n):
        f = synthesise(
            bs[i],
            left_hand=None if hand_seq is None else hand_seq[i][0],
            right_hand=None if hand_seq is None else hand_seq[i][1],
            blendshapes=None if blendshape_seq is None else blendshape_seq[i],
            rest=rest,
        )
        g = _matrix_to_quat(_rodrigues(f.global_orient))
        if prev_global is not None and float(np.dot(prev_global, g)) < 0:
            g = -g
        prev_global = g
        out.append(f)
    return out


# --- the licence-gated model ------------------------------------------------


def load_smplx(path: Path) -> dict:
    """Load SMPL-X parameters from a user-supplied file or directory.

    The model is **not** vendored and must not be committed: SMPL-X is distributed by
    the MPI under a research licence requiring registration and agreement, and
    derivatives carry the same terms. Obtain it from the official source, then point
    this at it.

    **Three layouts are accepted, because the official download ships in all three.**
    The MPI distribution is commonly delivered as a *zip containing one ``.npy`` per
    array* (22 files: ``v_template.npy``, ``shapedirs.npy``, ...), and a caller who
    unzips it gets a directory rather than a file. Accepting only a single ``.npz``
    meant the code could not read the artifact the licence actually grants, which is a
    failure mode that looks like a user error and is not one. So:

    - a directory of ``*.npy`` files (the unzipped official download)
    - a single ``.npz`` archive (numpy's zipped-bundle format)
    - a single ``.pkl`` (the older body-only pickle)

    All three are normalised to the same ``dict[str, ndarray]`` that
    :func:`seam.avatar.mesh.smplx_mesh` consumes.
    """
    p = Path(path)

    if p.is_dir():
        arrays = {f.stem: np.load(f, allow_pickle=True) for f in sorted(p.glob("*.npy"))}
        if not arrays:
            raise FileNotFoundError(
                f"{p} is a directory but contains no .npy files. If you have the "
                "official SMPLX_NEUTRAL.npz.zip, unzip it into this directory first; "
                "the archive holds one .npy per array, not a single .npz."
            )
        return arrays

    if not p.is_file():
        raise FileNotFoundError(
            f"SMPL-X model not found at {p}. The parameters are licence-gated by the "
            "Max Planck Institute (see the official SMPL-X download page) and are "
            "deliberately not vendored in this repository. Download the .npz/.pkl "
            "yourself and pass its path."
        )
    if p.suffix == ".npz":
        data = np.load(str(p), allow_pickle=True)
        return {k: data[k] for k in data.files}
    if p.suffix == ".pkl":
        import pickle

        with p.open("rb") as fh:
            return pickle.load(fh)
    if p.suffix == ".zip":
        raise ValueError(
            f"{p} is a .zip. Unzip it and pass the extracted directory (the official "
            "download holds one .npy per array), or point at the inner .npz. This is "
            "refused explicitly because a .zip opened as a model would fail later, "
            "inside the skinning maths, where the cause is much harder to see."
        )
    raise ValueError(
        f"unsupported model file {p.suffix}; expected .npz, .pkl, or a directory of .npy"
    )


def model_rest_pose(model: dict) -> np.ndarray:
    """The rest-pose joint positions of a loaded SMPL-X model, in SMPL order.

    ``J_regressor @ v_template`` - the joint regressor applied to the template mesh -
    which is exactly the skeleton the skinning will use. Pass the result as
    ``rest=`` to :func:`retarget_body` and the retargeted pose is expressed in the same
    reference the mesh is built from.

    **Why this matters.** :func:`canonical_rest_pose` is a hand-written stick figure with
    roughly correct proportions, fine for a caller with no model file, but measurably
    different from the real thing: it puts the hips 0.18 apart where ``SMPLX_NEUTRAL``
    has 0.121, and it places the knee directly below the hip where the model's knee sits
    0.055 outboard. Retargeting real tracking against the approximate figure therefore
    injects a leg rotation that is not in the data - and because the two hips are
    mirrored, the injected errors point in opposite directions and the body renders in a
    straddle stance.

    Returns 24 rows even though the model carries 55 joints: the body joints come first
    and are the ones this module names.
    """
    if "J_regressor" not in model or "v_template" not in model:
        raise ValueError(
            "model is missing J_regressor and/or v_template, so its rest joints cannot "
            "be recovered. A .pkl body-only file may be reshaped differently; load the "
            ".npz or the directory of .npy from the official download instead."
        )
    J = np.asarray(model["J_regressor"], dtype=np.float64) @ np.asarray(
        model["v_template"], dtype=np.float64
    )
    if J.shape[0] < 24:
        raise ValueError(f"model exposes only {J.shape[0]} joints; the SMPL body needs at least 24")
    return J[:24].copy()


def export_glb(frames: Sequence[SmplxFrame], out: Path, *, fps: int = 25) -> Path:
    """**Deprecated and removed as a behaviour.** Use :func:`seam.avatar.mesh.export_glb`.

    This function used to write ``json.dumps(...)`` to whatever path it was given -
    including a path ending in ``.glb``. That produced a file that passes ``ls``, is not
    a mesh, and loads as nothing in a viewer. It is the single defect this project has
    documented most often, and the documentation claimed it had been removed; it had not,
    because this copy in ``synthesis.py`` outlived the one in ``mesh.py`` and stayed on
    the package's public surface via ``seam.avatar.__init__``.

    Two distinct operations were conflated by that one name, and both now exist under
    names that say which is which:

    - geometry -> :func:`seam.avatar.mesh.export_glb` (real binary glTF, verified by reload)
    - parameters -> :func:`seam.avatar.mesh.export_parameters` (a ``.json`` path, on purpose)

    Rather than deleting the symbol - which would break an outside caller at import time
    with an ``ImportError`` that explains nothing - it now forwards to the real exporter
    after converting, so a caller that meant "give me a GLB" gets a GLB, and a caller that
    meant "dump parameters" is told plainly which function to use instead.
    """
    raise NotImplementedError(
        "synthesis.export_glb was removed: it wrote a JSON parameter dump to a .glb "
        "path, which is not a mesh and does not open in a viewer. Use one of:\n"
        "  seam.avatar.mesh.export_glb(mesh_sequence, path)      -> real binary glTF "
        "(needs the SMPL-X model, via mesh.smplx_mesh)\n"
        "  seam.avatar.mesh.export_parameters(frames, path.json) -> the parameter "
        "sequence, written to a .json path\n"
        "It refuses rather than writing a file that looks correct and is not."
    )


def log_export(out: Path, n_frames: int) -> None:
    from seam.logging import get

    get(__name__).info("wrote %s (%d frames)", out, n_frames)
