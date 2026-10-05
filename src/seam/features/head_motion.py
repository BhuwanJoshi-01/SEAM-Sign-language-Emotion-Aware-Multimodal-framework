"""Head motion from MediaPipe's facial transformation matrix, named for what the head does.

Why this module exists
----------------------
`seam.features.markers._euler_from_matrix` returns ``(roll, pitch, yaw)`` using the
aerospace convention, in which the body's forward axis is x. A face's forward axis in
MediaPipe's canonical model is z, with y up. The decomposition is correct and the *names*
are not: measured on the 200 EmoSign clips against the face landmarks themselves,

==========================  ===============  ================  ==============
landmark geometry           that module's    that module's     that module's
                            ``roll``         ``pitch``         ``yaw``
==========================  ===============  ================  ==============
nose moves sideways (turn)  -0.03            **0.90**          0.58
nose moves up/down (nod)    **0.82**         0.03              -0.19
eye line rotates (tilt)     0.08             -0.29             **-0.95**
==========================  ===============  ================  ==============

(median per-clip correlation). `markers.signals` reads ``yaw`` for `head_shake` and
``pitch`` for `head_nod`, so its head shake measures a head *tilt* and its head nod
measures a head *turn*. Both scored 0.50 against human annotation, which was read as "head
movements cannot be seen from this input". They can: on the right axis the same detector
family scores well above chance (`scripts/validate_markers.py`).

The functions here read the face's own forward and right vectors, which is also what the
live page (`docs/index.html`, ``headAngles``) does, so the browser and the offline
validation now measure the same quantity.

`markers.signals` is deliberately left as it is in this change: nine cited artifacts were
computed with it and the staleness guard would, correctly, mark every one of them stale.
The defect and the artifacts it touches are recorded in `paper/EXPERIMENT_LOG.md`.
"""

from __future__ import annotations

import numpy as np

#: Column order of :func:`head_angles`.
TURN, NOD, TILT = 0, 1, 2
AXES = ("turn", "nod", "tilt")

#: What the live page ran before any label was looked at: `Oscillation(5, 1300)`.
BROWSER_MIN_DEG = 5.0
BROWSER_WINDOW_S = 1.3
#: Reversals inside the window at which the page's level reaches 1.0.
BROWSER_FULL_SCALE = 3.0


def head_angles(rotation: np.ndarray) -> np.ndarray:
    """(T, 3) degrees: turn (left/right, the shake axis), nod (up/down), tilt (ear to shoulder).

    ``rotation`` is (T, 4, 4) or (T, 3, 3), canonical face space to camera space. The
    third column is where the face points and the first is its right-hand direction, so
    the three angles are read off those two vectors and no Euler order is involved.
    """
    if rotation is None or len(rotation) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    r = np.asarray(rotation, dtype=np.float64)[:, :3, :3]
    turn = np.arctan2(r[:, 0, 2], r[:, 2, 2])
    nod = -np.arcsin(np.clip(r[:, 1, 2], -1.0, 1.0))
    tilt = np.arctan2(r[:, 1, 0], r[:, 0, 0])
    return np.degrees(np.stack([turn, nod, tilt], axis=1))


def reversal_score(
    angle_deg: np.ndarray,
    fps: float,
    *,
    min_deg: float = BROWSER_MIN_DEG,
    window_s: float = BROWSER_WINDOW_S,
) -> np.ndarray:
    """Per-frame oscillation level: the live page's detector, ported line for line.

    The angle has to travel at least ``min_deg`` one way and then at least ``min_deg``
    back to count one reversal; the score is the number of reversals in the last
    ``window_s`` seconds over :data:`BROWSER_FULL_SCALE`, capped at 1.2. It is causal - a
    frame is scored from the past only - because that is all a live page has.
    """
    x = np.asarray(angle_deg, dtype=np.float64)
    out = np.zeros(len(x), dtype=np.float64)
    if len(x) == 0 or fps <= 0:
        return out
    window_ms = 1000.0 * window_s
    reversals: list[float] = []
    direction = 0.0
    extreme = float(x[0])
    for i in range(1, len(x)):
        t = 1000.0 * i / fps
        d = float(x[i]) - extreme
        if direction == 0.0:
            if abs(d) >= min_deg:
                direction = float(np.sign(d))
                extreme = float(x[i])
        elif np.sign(d) == direction:
            extreme = float(x[i])
        elif abs(d) >= min_deg:
            reversals.append(t)
            direction = -direction
            extreme = float(x[i])
        reversals = [r for r in reversals if t - r <= window_ms]
        out[i] = min(len(reversals) / BROWSER_FULL_SCALE, 1.2)
    return out


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """Centred moving average with the ends held, so the output is as long as the input."""
    if k <= 1 or len(x) == 0:
        return np.asarray(x, dtype=np.float64)
    padded = np.pad(np.asarray(x, dtype=np.float64), (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def band_energy(
    angle_deg: np.ndarray,
    fps: float,
    *,
    window_s: float = 0.8,
    fast_s: float = 0.12,
    slow_s: float = 0.7,
) -> np.ndarray:
    """Per-frame RMS, in degrees, of the part of the angle that goes back and forth.

    The angle is smoothed over ``fast_s`` to drop single-frame tracking noise, and its own
    moving average over ``slow_s`` is subtracted to drop a slow turn or a held posture.
    What is left is motion with a period between roughly those two, which is where a shake
    or a nod lives; its rolling RMS over ``window_s`` is the score. Unlike
    :func:`reversal_score` this has no amplitude threshold, so a small conversational
    shake scores in proportion to its size instead of not at all.
    """
    x = np.asarray(angle_deg, dtype=np.float64)
    if len(x) < 3 or fps <= 0:
        return np.zeros(len(x), dtype=np.float64)
    fast = _moving_average(x, max(round(fast_s * fps), 1))
    slow = _moving_average(x, max(round(slow_s * fps) | 1, 3))
    band = fast - slow
    return np.sqrt(_moving_average(band**2, max(round(window_s * fps) | 1, 3)))


def live_band_rms(
    angle_deg: np.ndarray,
    fps: float,
    *,
    fast_s: float = 0.08,
    slow_s: float = 0.5,
    hold_s: float = 0.6,
) -> np.ndarray:
    """:func:`band_energy` as a live page can compute it: from the past only.

    Three exponential averages, each advanced by the time since the previous frame, so the
    result does not depend on the frame rate: a fast one to drop tracking noise, a slow
    one that follows posture, and one over the squared difference of the two. The square
    root of the last is the RMS, in degrees, of the back-and-forth part of the angle.
    `docs/index.html` (``BandEnergy``) is this function line for line.
    """
    x = np.asarray(angle_deg, dtype=np.float64)
    out = np.zeros(len(x), dtype=np.float64)
    if len(x) == 0 or fps <= 0:
        return out
    dt = 1.0 / fps
    a_fast, a_slow, a_hold = (1.0 - np.exp(-dt / tau) for tau in (fast_s, slow_s, hold_s))
    fast = slow = float(x[0])
    energy = 0.0
    for i in range(1, len(x)):
        fast += a_fast * (float(x[i]) - fast)
        slow += a_slow * (float(x[i]) - slow)
        band = fast - slow
        energy += a_hold * (band * band - energy)
        out[i] = np.sqrt(energy)
    return out


def live_oscillation(
    angle_deg: np.ndarray,
    other_deg: np.ndarray,
    fps: float,
    *,
    contrast: float = 0.5,
    slow_s: float = 0.5,
    hold_s: float = 0.6,
) -> np.ndarray:
    """Degrees RMS of back-and-forth motion on one axis, less a share of the other axis.

    A shake is a turn that oscillates while the nod axis stays comparatively still, and a
    nod is the reverse. Subtracting ``contrast`` times the other axis's energy keeps a
    whole-head wobble, which moves both, from reading as either.
    """
    own = live_band_rms(angle_deg, fps, slow_s=slow_s, hold_s=hold_s)
    other = live_band_rms(other_deg, fps, slow_s=slow_s, hold_s=hold_s)
    return np.maximum(own - contrast * other, 0.0)


def brow_geometry(face_landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Brow position measured on the 478 face landmarks, as a check on the blendshapes.

    Returns the gap between the inner brow ends and the height of the inner, middle and
    outer brow above the eye, all in units of the distance between the outer eye corners
    so that distance from the camera cancels. Heights are taken along the face's own
    chin-to-forehead direction, so a tilted head does not read as a moved brow.
    """
    lm = np.asarray(face_landmarks, dtype=np.float64)[:, :, :2]
    eye_span = np.linalg.norm(lm[:, 263] - lm[:, 33], axis=1) + 1e-6
    up = lm[:, 10] - lm[:, 152]
    up = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-9)

    def height(brow: int, eye: int) -> np.ndarray:
        return np.asarray(((lm[:, brow] - lm[:, eye]) * up).sum(axis=1) / eye_span)

    return {
        "inner_gap": np.linalg.norm(lm[:, 107] - lm[:, 336], axis=1) / eye_span,
        "inner_height": 0.5 * (height(107, 133) + height(336, 362)),
        "mid_height": 0.5 * (height(105, 159) + height(334, 386)),
        "outer_height": 0.5 * (height(70, 33) + height(300, 263)),
    }
