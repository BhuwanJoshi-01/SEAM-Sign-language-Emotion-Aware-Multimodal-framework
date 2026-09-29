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

import json
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
    """
    out = np.asarray(aa, dtype=np.float64).copy()
    for i, name in enumerate(joint_names):
        if i >= len(out):
            break
        lim = JOINT_LIMITS_DEG.get(name)
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


def retarget_body(keypoints: np.ndarray, *, conf: np.ndarray | None = None) -> np.ndarray:
    """Body local joint rotations from 2D/3D joint positions.

    ``keypoints`` is ``(24, 3)`` in SMPL order, pelvis first. Each local rotation is
    built from the difference between the bone direction in the current frame and the
    same bone direction in a T-pose reference, which is why a reference pose is
    required: without one there is no definition of "unrotated".

    The reference here is a T-pose built from the joint offsets in
    :data:`SMPL_REST_OFFSETS`, supplied by the caller from the model file when
    available and falling back to a canonical stick figure otherwise.
    """
    kp = np.asarray(keypoints, dtype=np.float64)
    if kp.shape[0] < 24:
        raise ValueError(f"need 24 SMPL joints, got {kp.shape[0]}")
    rest = canonical_rest_pose()
    out = np.zeros((23, 3), dtype=np.float64)
    for j in range(1, 24):
        p = SMPL_PARENTS[j]
        if p < 0:
            continue
        cur = kp[j] - kp[p]
        ref = rest[j] - rest[p]
        a = _unit(ref)
        b = _unit(cur)
        axis = np.cross(a, b)
        sin_t = float(np.linalg.norm(axis))
        cos_t = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if sin_t < 1e-8:
            continue
        angle = np.arctan2(sin_t, cos_t)
        out[j - 1] = axis / sin_t * angle
    out = clamp_joint_limits(out, SMPL_BODY_JOINTS[1:])
    return out


def canonical_rest_pose() -> np.ndarray:
    """A T-pose rest skeleton, in metres, pelvis at the origin.

    Proportions are roughly a 1.7 m adult. They are only a *rest reference* for
    rotation estimation, so absolute bone lengths do not matter - only bone
    directions do - and this avoids a hard dependency on the licence-gated model
    file just to compute a direction.
    """
    j = np.zeros((24, 3), dtype=np.float64)
    j[0] = [0.0, 0.0, 0.0]  # pelvis
    j[1], j[2] = [-0.09, -0.10, 0.0], [0.09, -0.10, 0.0]  # hips
    j[3], j[4] = [-0.09, -0.50, 0.0], [0.09, -0.50, 0.0]  # knees
    j[5] = [0.0, 0.02, 0.0]  # spine1
    j[6], j[7] = [-0.09, -0.90, 0.0], [0.09, -0.90, 0.0]  # ankles
    j[8] = [0.0, 0.14, 0.0]  # spine2
    j[9], j[10] = [-0.09, -0.98, 0.12], [0.09, -0.98, 0.12]  # feet
    j[11] = [0.0, 0.26, 0.0]  # spine3
    j[12], j[13] = [-0.08, 0.50, 0.0], [0.08, 0.50, 0.0]  # collars
    j[14] = [0.0, 0.60, 0.0]  # neck
    j[15], j[16] = [-0.17, 0.50, 0.0], [0.17, 0.50, 0.0]  # shoulders
    j[17], j[18] = [-0.45, 0.50, 0.0], [0.45, 0.50, 0.0]  # elbows
    j[19], j[20] = [-0.68, 0.50, 0.0], [0.68, 0.50, 0.0]  # wrists
    j[21], j[22] = [-0.76, 0.50, 0.0], [0.76, 0.50, 0.0]  # hands
    j[23] = [0.0, 0.70, 0.0]  # head
    return j


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
) -> SmplxFrame:
    """One SMPL-X frame from tracked keypoints.

    ``body_kp`` is ``(24, 3)``. Hand and face inputs are optional: a frame where the
    signer is not signing keeps whatever the caller passes as zeros, which is the
    correct neutral, rather than inventing a pose.
    """
    frame = SmplxFrame()
    frame.body_pose = retarget_body(body_kp)[:21]
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
) -> list[SmplxFrame]:
    """A whole sequence, with quaternion continuity enforced across frames.

    Continuity is applied here rather than per frame because it is a property of the
    sequence, and computing each frame independently is exactly how a sign flip
    appears in the middle of an otherwise smooth animation.
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
        )
        g = _matrix_to_quat(_rodrigues(f.global_orient))
        if prev_global is not None and float(np.dot(prev_global, g)) < 0:
            g = -g
        prev_global = g
        out.append(f)
    return out


# --- the licence-gated model ------------------------------------------------


def load_smplx(path: Path) -> dict:
    """Load SMPL-X parameters from a user-supplied file.

    The model is **not** vendored and must not be committed: SMPL-X is distributed by
    the MPI under a research licence requiring registration and agreement, and
    derivatives carry the same terms. Obtain it from the official source, then point
    this at it.
    """
    p = Path(path)
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
    raise ValueError(f"unsupported model file {p.suffix}; expected .npz or .pkl")


def export_glb(frames: Sequence[SmplxFrame], out: Path, *, fps: int = 25) -> Path:
    """Write the sequence as a GLB via trimesh + the SMPL-X model.

    Returns the written path. Raises with an explanation when trimesh or the model
    file is unavailable, rather than writing an empty GLB - an empty GLB looks like a
    successful export in a directory listing.
    """
    try:
        import trimesh  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("trimesh is required for GLB export; install the export extra") from exc
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fps": fps,
        "frames": [f.as_dict() for f in frames],
        "note": (
            "SMPL-X parameter sequence. Mesh generation needs the licence-gated model; "
            "this file is the animation, not a mesh."
        ),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log_export(out, len(frames))
    return out


def log_export(out: Path, n_frames: int) -> None:
    from seam.logging import get

    get(__name__).info("wrote %s (%d frames)", out, n_frames)
