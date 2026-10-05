"""Finger and wrist rotations for an SMPL-X hand, solved from 21 tracked hand points.

Why this exists
---------------
A whole-body regressor gives the avatar's hands almost nothing. Measured on the four demo
clips, SMPLer-X's 30 finger joints bend 12 to 16 degrees on average and move 5 to 8 degrees
over a clip: the hands stay flat and open while the signer forms handshapes. That is a known
property of whole-body regression (the hands are a few dozen pixels of a body-sized crop),
and the usual remedy is a dedicated hand model. This project already runs one - MediaPipe's
hand landmarker, which returns 21 points per hand in three dimensions - so the fingers are
solved from those points instead, and the body keeps the regressor's arms.

How
---
The palm is rigid, so three of its points (wrist, index knuckle, little-finger knuckle,
with the middle knuckle for direction) define a frame. The same frame is built on the
tracked hand and on the model's rest hand. Expressed in its own palm frame, each finger
bone of the tracked hand is a direction the model's bone has to reach, and each joint's
rotation is the smallest rotation that takes its rest bone there, solved down the chain
from knuckle to fingertip. Nothing here depends on where the camera is, how large the hand
appears, or which way the arm points: only on the hand's shape.

The wrist can be solved the same way, in world space, from where the palm frame points.

What is not recovered: rotation of a finger about its own length (twist). A bone's
direction does not carry it, and fingers barely have any.

Pure numpy. The tracking itself is in `seam.perception.hands3d`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: SMPL-X orders each hand's 15 joints by finger, three joints a finger, knuckle first.
FINGERS = ("index", "middle", "pinky", "ring", "thumb")

#: MediaPipe hand points for each finger: its three joints, then the fingertip.
MP_CHAIN: dict[str, tuple[int, int, int, int]] = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "pinky": (17, 18, 19, 20),
    "ring": (13, 14, 15, 16),
    "thumb": (1, 2, 3, 4),
}
MP_WRIST, MP_INDEX, MP_MIDDLE, MP_PINKY = 0, 5, 9, 17

#: SMPL-X joint indices.
WRIST_JOINT = {"left": 20, "right": 21}
ELBOW_JOINT = {"left": 18, "right": 19}
FIRST_FINGER_JOINT = {"left": 25, "right": 40}

#: SMPL-X mesh vertices at the fingertips (the model has no fingertip joints). These are
#: the published SMPL-X vertex ids; `hand_rest_from_model` checks each lies beyond its
#: finger's last joint and roughly along it, so a wrong id cannot pass silently.
TIP_VERTEX: dict[str, dict[str, int]] = {
    "left": {"index": 4933, "middle": 5058, "pinky": 5286, "ring": 5169, "thumb": 5361},
    "right": {"index": 7669, "middle": 7794, "pinky": 8022, "ring": 7905, "thumb": 8079},
}

#: No finger joint bends further than this; a larger solution is tracking noise.
MAX_JOINT_ANGLE = 2.0
#: The wrist is kept within this of the forearm; past it the palm frame is not believed.
MAX_WRIST_ANGLE = 2.2


@dataclass(frozen=True, slots=True)
class HandRest:
    """The model's hand in its rest pose: its palm frame and where each bone points."""

    side: str
    frame: np.ndarray  # (3, 3), columns along the hand, across the palm, out of the palm
    bones: np.ndarray  # (15, 3) unit vectors, finger joint to its child, SMPL-X order


def palm_frame(
    wrist: np.ndarray, index: np.ndarray, middle: np.ndarray, pinky: np.ndarray
) -> np.ndarray | None:
    """A right-handed frame fixed in the palm, or ``None`` when the points are degenerate.

    Built the same way on a tracked hand and on the model's hand, so what one frame calls
    "along the hand" the other does too. Which side of the hand the third axis leaves by
    differs between a left and a right hand, and does not matter: a hand is only ever
    compared with a model hand of the same side.
    """
    along = np.asarray(middle, dtype=np.float64) - np.asarray(wrist, dtype=np.float64)
    across = np.asarray(index, dtype=np.float64) - np.asarray(pinky, dtype=np.float64)
    n_along = float(np.linalg.norm(along))
    if n_along < 1e-9:
        return None
    u = along / n_along
    normal = np.cross(u, across)
    n_normal = float(np.linalg.norm(normal))
    if n_normal < 1e-9 * max(float(np.linalg.norm(across)), 1e-9) or n_normal < 1e-12:
        return None
    n = normal / n_normal
    return np.stack([u, np.cross(n, u), n], axis=1)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking direction ``a`` to direction ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    c = float(np.clip(a @ b, -1.0, 1.0))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Opposite directions: half a turn about any axis perpendicular to a.
        other = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        k = np.cross(a, other)
        k /= np.linalg.norm(k)
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def matrix_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to axis-angle, stable near zero and near half a turn."""
    R = np.asarray(R, dtype=np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(c))
    if theta < 1e-9:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Near half a turn the antisymmetric part vanishes; take the axis from R + I.
        m = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(m), 0.0, None))
        i = int(np.argmax(k))
        k = m[:, i] / max(k[i], 1e-12)
        return np.asarray(theta * k / max(float(np.linalg.norm(k)), 1e-12))
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.asarray(theta * w / (2.0 * np.sin(theta)))


def aa_to_matrix(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def _clamped(R: np.ndarray, limit: float) -> np.ndarray:
    aa = matrix_to_aa(R)
    angle = float(np.linalg.norm(aa))
    if angle <= limit:
        return aa
    return aa * (limit / angle)


def hand_rest(joints: np.ndarray, tips: dict[str, np.ndarray], side: str) -> HandRest:
    """Rest geometry of one hand from the model's rest joint positions and fingertips."""
    joints = np.asarray(joints, dtype=np.float64)
    first = FIRST_FINGER_JOINT[side]
    wrist = joints[WRIST_JOINT[side]]
    knuckle = {name: joints[first + 3 * i] for i, name in enumerate(FINGERS)}
    frame = palm_frame(wrist, knuckle["index"], knuckle["middle"], knuckle["pinky"])
    if frame is None:
        raise ValueError(f"the model's {side} palm is degenerate; wrong joint layout?")
    bones = np.zeros((15, 3))
    for i, name in enumerate(FINGERS):
        chain = [joints[first + 3 * i + k] for k in range(3)] + [np.asarray(tips[name])]
        for k in range(3):
            d = chain[k + 1] - chain[k]
            length = float(np.linalg.norm(d))
            if length < 1e-6:
                raise ValueError(f"{side} {name} bone {k} has no length in the rest pose")
            bones[3 * i + k] = d / length
    return HandRest(side=side, frame=frame, bones=bones)


def hand_rest_from_model(model: dict, betas: np.ndarray | None, side: str) -> HandRest:
    """:func:`hand_rest` for an SMPL-X model dict, with the fingertip vertices checked."""
    v = np.asarray(model["v_template"], dtype=np.float64)
    if betas is not None:
        shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
        b = np.zeros(shapedirs.shape[-1])
        given = np.asarray(betas, dtype=np.float64).reshape(-1)
        b[: min(len(given), len(b))] = given[: len(b)]
        v = v + shapedirs @ b
    joints = np.asarray(model["J_regressor"], dtype=np.float64) @ v
    first = FIRST_FINGER_JOINT[side]
    tips = {}
    for i, name in enumerate(FINGERS):
        tip = v[TIP_VERTEX[side][name]]
        last, before = joints[first + 3 * i + 2], joints[first + 3 * i + 1]
        beyond = tip - last
        along = (last - before) / np.linalg.norm(last - before)
        length = float(np.linalg.norm(beyond))
        if not (0.008 < length < 0.06 and float(beyond @ along) / length > 0.8):
            raise ValueError(
                f"vertex {TIP_VERTEX[side][name]} is not the {side} {name} fingertip of this "
                "model; the mesh topology differs from SMPL-X"
            )
        tips[name] = tip
    return hand_rest(joints, tips, side)


def solve_fingers(points: np.ndarray, rest: HandRest) -> np.ndarray | None:
    """(15, 3) axis-angle finger rotations for one tracked hand, or ``None`` if unusable.

    ``points`` is (21, 3) in any right-handed frame and any unit.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    seen = palm_frame(p[MP_WRIST], p[MP_INDEX], p[MP_MIDDLE], p[MP_PINKY])
    if seen is None:
        return None
    to_rest = rest.frame @ seen.T  # tracked hand, turned so its palm lies on the model's
    out = np.zeros((15, 3))
    for i, name in enumerate(FINGERS):
        chain = MP_CHAIN[name]
        carried = np.eye(3)  # rotation accumulated down this finger so far
        for k in range(3):
            d = p[chain[k + 1]] - p[chain[k]]
            length = float(np.linalg.norm(d))
            if length < 1e-9:
                continue
            target = carried.T @ (to_rest @ (d / length))
            aa = _clamped(rotation_between(rest.bones[3 * i + k], target), MAX_JOINT_ANGLE)
            out[3 * i + k] = aa
            carried = carried @ aa_to_matrix(aa)
    return out


def solve_wrist(points: np.ndarray, rest: HandRest, to_world: np.ndarray) -> np.ndarray | None:
    """The wrist joint's rotation in world space, (3, 3), from where the palm points.

    ``to_world`` rotates the tracker's frame (camera: x right, y down, z away) into the
    avatar's world. The result ``R`` satisfies ``R @ rest.frame == world palm frame``.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        return None
    seen = palm_frame(p[MP_WRIST], p[MP_INDEX], p[MP_MIDDLE], p[MP_PINKY])
    if seen is None:
        return None
    return np.asarray(to_world, dtype=np.float64) @ seen @ rest.frame.T


def global_rotations(pose: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """(J, 3, 3) world rotation of every joint from (J, 3) parent-relative axis-angle."""
    pose = np.asarray(pose, dtype=np.float64)
    out = np.zeros((len(pose), 3, 3))
    for j in range(len(pose)):
        local = aa_to_matrix(pose[j])
        parent = int(parents[j])
        out[j] = local if parent < 0 or j == 0 else out[parent] @ local
    return out


def wrist_local(
    wrist_world: np.ndarray, elbow_world: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Parent-relative wrist axis-angle, or ``fallback`` when the solution is implausible.

    A wrist does not bend much past a right angle. A solution beyond
    :data:`MAX_WRIST_ANGLE` means the palm frame was built from a hand seen edge-on, where
    its normal is poorly determined, and the regressor's wrist is kept instead.
    """
    aa = matrix_to_aa(np.asarray(elbow_world).T @ np.asarray(wrist_world))
    if float(np.linalg.norm(aa)) > MAX_WRIST_ANGLE:
        return np.asarray(fallback, dtype=np.float64).copy()
    return aa


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Axis-angle between two axis-angle rotations, a fraction ``t`` of the way."""
    Ra = aa_to_matrix(a)
    step = matrix_to_aa(Ra.T @ aa_to_matrix(b))
    return matrix_to_aa(Ra @ aa_to_matrix(step * t))


def fill_gaps(track: list[np.ndarray | None], max_gap: int) -> tuple[list[np.ndarray | None], int]:
    """Bridge short runs of missing frames by rotating from the frame before to the one after.

    Each element is (N, 3) axis-angle or ``None``. A gap of at most ``max_gap`` frames with
    a solved frame on both sides is interpolated on the rotation sphere, joint by joint; a
    gap at either end of the clip, or a longer one, is left as ``None`` for the caller to
    fill from another source. Returns the track and how many frames were bridged.
    """
    out = list(track)
    n = len(out)
    bridged = 0
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        before, after = (out[i - 1] if i > 0 else None), (out[j] if j < n else None)
        if before is not None and after is not None and (j - i) <= max_gap:
            for k in range(i, j):
                t = (k - i + 1) / (j - i + 1)
                out[k] = np.stack([_slerp(before[m], after[m], t) for m in range(len(before))])
                bridged += 1
        i = j
    return out, bridged


def mean_bend_deg(hand_pose: np.ndarray) -> float:
    """Mean rotation angle of a hand's joints, in degrees: how far from flat it is."""
    return float(np.degrees(np.linalg.norm(np.asarray(hand_pose).reshape(-1, 3), axis=1)).mean())
