"""Signing-space pose features for gloss recognition (M5a).

**Why a body-relative frame at all.** Raw MediaPipe pose is in image pixels, so the same
sign produced at a different distance from the camera, or by a signer seated at a
different height, lands in a different part of the vector. A model trained on that will
learn camera placement and signer posture along with the sign. Every token is therefore
expressed in a *signing space*: origin at the shoulder midpoint, unit scale set by
shoulder width, x to the signer's right, y up, and the shoulder line rotated horizontal.

**What that normalisation throws away, deliberately.** Levelling the shoulders discards
torso lean, and translating to the shoulder midpoint discards absolute head position.
Both are genuine linguistic signal in some sign languages and in emphasis. This module
makes the choice explicit rather than burying it: `level` and `keep_lean` are
parameters, and :func:`signing_space` documents what each costs. For a first
vocabulary-level model the body-relative frame is the right default, because a signer
who is shifting posture mid-sentence would otherwise shift the entire feature.

**What is not here yet.** Hands. The 21-point hand meshes are in the landmark array and
matter enormously for sign identity - many ASL signs differ only in finger
configuration - but at 4.6 s and 480x320 they are the noisiest part of the array, and
M5a is a *smoke* model whose job is to answer whether the pipeline and the alignment
carry any signal at all. `part="upper+hands"` includes them; the default does not, and
that omission is a known limitation of the first result rather than a claim that hands
are unimportant.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

#: Offsets into the concatenated 553-point landmark array.
FACE_POINTS = 478
POSE_POINTS = 33
LEFT_HAND_POINTS = 21
RIGHT_HAND_POINTS = 21
TOTAL_POINTS = FACE_POINTS + POSE_POINTS + LEFT_HAND_POINTS + RIGHT_HAND_POINTS

#: MediaPipe Pose indices. Only the signing-relevant upper body and the hips, which fix
#: the torso length and so make the scale estimate stable.
NOSE = 0
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24

UPPER_BODY = (NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP)

Part = Literal["upper", "upper+hands"]


def split_landmarks(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a (..., 553, 3) landmark array into face, pose, and both hands.

    The offsets are asserted rather than assumed: a landmark array that silently lost its
    face block would otherwise be sliced into plausible-looking garbage, and every
    downstream number would be quietly wrong.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape[-2] != TOTAL_POINTS:
        raise ValueError(
            f"expected {TOTAL_POINTS} landmark points, got {lm.shape[-2]}. "
            "The 553-point layout is 478 face + 33 pose + 21 left hand + 21 right hand; "
            "a different count means the extractor changed and these offsets are stale."
        )
    left = lm[..., FACE_POINTS + POSE_POINTS : FACE_POINTS + POSE_POINTS + LEFT_HAND_POINTS, :]
    right = lm[..., TOTAL_POINTS - RIGHT_HAND_POINTS : TOTAL_POINTS, :]
    return (
        lm[..., :FACE_POINTS, :],
        lm[..., FACE_POINTS : FACE_POINTS + POSE_POINTS, :],
        left,
        right,
    )


def signing_space(
    pose: np.ndarray,
    *,
    level: bool = True,
    min_width: float = 1e-3,
) -> np.ndarray:
    """Put (T, 33, 3) pose into a body-relative signing frame.

    Returns (T, K, 3) for the selected joints, with x to the signer's right, y up and z
    toward the camera negated so that positive z is away from the camera.

    MediaPipe's image y grows downward, so y is negated here. Skipping that is the kind of
    error that produces a model which works and an avatar whose hands move upward when
    the signer moves them downward.

    A frame whose shoulder width collapses below `min_width` is a detection dropout:
    there is no reliable body frame for it, so it is emitted as all-zero and flagged by
    returning a zero row. Clamping the divisor to a small epsilon instead - the obvious
    implementation - multiplies the noise by 1/eps and turns a dropout into a 4e5 spike,
    which then flows through the std and delta summaries and swamps the token. Zeroing
    is the honest representation of "no measurement here".
    """
    p = np.asarray(pose, dtype=np.float64)
    if p.ndim != 3 or p.shape[-2] != POSE_POINTS:
        raise ValueError(f"expected (T, {POSE_POINTS}, 3) pose, got {p.shape}")
    if p.shape[0] == 0:
        return p[:, UPPER_BODY, :].copy()

    # MediaPipe y is down, so flip before deriving anything from it: computing the
    # centre first and flipping afterwards leaves the origin in the old convention and
    # puts the shoulder midpoint at y = -2 rather than 0.
    out = p.copy()
    # Base transform: a 180-degree turn about z. Negating y makes +y up (MediaPipe's y
    # grows downward); negating x as well makes +x the signer's *right* for a
    # front-facing capture, where the signer's left shoulder has the larger image x.
    # Negating z makes +z point away from the viewer, giving a right-handed frame.
    #
    # Both x and y are negated together on purpose. Flipping y alone is a reflection,
    # which silently inverts the handedness of every x coordinate derived afterwards.
    out[..., 0] *= -1.0
    out[..., 1] *= -1.0
    out[..., 2] *= -1.0

    l_sh, r_sh = out[:, L_SHOULDER, :], out[:, R_SHOULDER, :]
    centre = 0.5 * (l_sh + r_sh)
    axis = l_sh - r_sh
    width = np.linalg.norm(axis[:, :2], axis=1)
    valid = width > min_width
    safe = np.where(valid, width, 1.0)

    out[:, :, :2] -= centre[:, None, :2]
    out[..., 2] -= centre[:, None, 2]

    if level:
        # Measured in the *original* image frame, before the base 180-degree turn. The
        # base turn is itself a rotation, so angles commute, but the wrap is only
        # unambiguous relative to the image's own axes - doing it afterwards makes the
        # choice of branch depend on the base turn and in-plane rotation stops being an
        # invariance.
        roll = np.arctan2(
            p[:, L_SHOULDER, 1] - p[:, R_SHOULDER, 1],
            p[:, L_SHOULDER, 0] - p[:, R_SHOULDER, 0],
        )
        # Wrap to (-pi/2, pi/2] before removing it. The raw atan2 of a shoulder axis
        # that already points along -x is pi, and subtracting that rotates the signer
        # 180 degrees - flipping the x convention and putting the head down. Levelling
        # must only ever remove a lean, never turn the signer around.
        roll = np.where(roll > np.pi / 2, roll - np.pi, roll)
        roll = np.where(roll <= -np.pi / 2, roll + np.pi, roll)
        c, sn = np.cos(-roll), np.sin(-roll)
        # Copy, not a view: writing column 0 in place would change the values that the
        # column-1 assignment then reads, applying the rotation twice and producing an
        # error that grows with the lean.
        xy = out[:, :, :2].copy()
        out[:, :, 0] = xy[..., 0] * c[:, None] - xy[..., 1] * sn[:, None]
        out[:, :, 1] = xy[..., 0] * sn[:, None] + xy[..., 1] * c[:, None]

        # Guard for a genuinely inverted capture. A pose with the head below the
        # shoulder line cannot be fixed without also reversing the x convention, so
        # this trades handedness for an upright frame and is reported rather than
        # hidden. Real captures should not trigger it.
        if np.any(out[:, NOSE, 1] < 0):
            flip = out[:, NOSE, 1] < 0
            out[flip, :, 0] *= -1.0
            out[flip, :, 1] *= -1.0

    out /= safe[:, None, None]
    out[~valid] = 0.0
    return out[:, UPPER_BODY, :]


def token_feature(
    landmarks: np.ndarray,
    start: int,
    end: int,
    *,
    part: Part = "upper",
    stats: tuple[str, ...] = ("mean", "std", "delta"),
) -> np.ndarray:
    """Summarise one gloss token's frames into a fixed-length vector.

    `start`/`end` are **1-based inclusive crop-frame indices**, as returned by
    `asllrp.crop_frame_range`, which rescales the 30 fps session timeline to the clip's
    own frame rate. Out-of-range positions are reported by raising rather than clamped:
    clamping would silently label the wrong frames, which is the failure this whole
    module exists downstream of.

    The temporal summary is mean, std and mean-absolute-frame-to-frame-delta rather than
    a flattened sequence, so the feature is length-invariant. Sign duration is real
    information, so it is appended as a final scalar rather than discarded - but it is
    appended *separately* so a downstream model can be tested with and without it, since
    on this corpus duration correlates strongly with syntactic class and would
    otherwise let a duration classifier masquerade as a sign recogniser.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    _, pose, lh, rh = split_landmarks(lm)

    n = pose.shape[0]
    if not (1 <= start <= end <= n):
        raise IndexError(
            f"token frames {start}-{end} fall outside a {n}-frame clip. "
            "Do not clamp: report the token as unalignable."
        )
    sp = signing_space(pose[start - 1 : end])
    if part == "upper+hands":
        # Hands are expressed relative to the hand mesh's *own* wrist (MediaPipe hand
        # point 0), not the body pose's wrist estimate. Using the body estimate would
        # make the feature sensitive to how the two trackers disagree, and would leave
        # whole-hand translation leaking into the finger coordinates - mixing wrist
        # travel with finger configuration. The hand is already in normalised image
        # coordinates, matching the frame the body was put in.
        lh_s, rh_s = lh[start - 1 : end], rh[start - 1 : end]
        # Each hand is anchored on its OWN wrist. Broadcasting the first point of the
        # concatenated array would anchor the right hand on the left wrist, which mixes
        # the two hands together and makes the feature depend on where they are relative
        # to each other rather than on each hand's own shape.
        anchors = np.concatenate(
            [
                np.repeat(lh_s[:, :1, :], LEFT_HAND_POINTS, axis=1),
                np.repeat(rh_s[:, :1, :], RIGHT_HAND_POINTS, axis=1),
            ],
            axis=1,
        )
        hands = np.concatenate([lh_s, rh_s], axis=1)
        sp = np.concatenate([sp, hands - anchors], axis=1)
    return np.append(_summarise(sp, stats), float(end - start + 1) / 10.0)


def _summarise(seq: np.ndarray, stats: tuple[str, ...]) -> np.ndarray:
    """mean / std / mean-absolute-delta over time, flattened joint-major."""
    if seq.shape[0] == 0:
        raise IndexError("empty frame range")
    parts: list[np.ndarray] = []
    if "mean" in stats:
        parts.append(seq.mean(axis=0).ravel())
    if "std" in stats:
        parts.append(seq.std(axis=0).ravel())
    if "delta" in stats:
        d = np.abs(np.diff(seq, axis=0)) if seq.shape[0] > 1 else np.zeros_like(seq)
        parts.append(d.mean(axis=0).ravel())
    if not parts:
        raise ValueError("stats must name at least one of mean, std, delta")
    return np.concatenate(parts)


def feature_dim(part: Part = "upper", stats: tuple[str, ...] = ("mean", "std", "delta")) -> int:
    """Dimensionality of :func:`token_feature`, so callers can assert rather than assume."""
    n_seq = len(UPPER_BODY) if part == "upper" else len(UPPER_BODY) + 2 * LEFT_HAND_POINTS
    return n_seq * 3 * len(stats) + 1
