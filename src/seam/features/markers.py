"""Grammatical non-manual markers, as rules over the blendshape basis.

This is the object the confound audit turns on. In ASL the non-manual channel is
grammatically obligatory: raised brows mark a yes/no question, a furrowed brow
marks a wh-question, a head shake marks negation, mouth morphemes modify
predicates. The same channel carries affect. A facial-emotion model trained on
non-signers has no way to tell the two apart, and the documented result is that
it reads grammar as negative affect.

So the markers have to be *measurable* before they can be controlled for. These
are hand-set rules over the 52 ARKit coefficients plus the head rotation, and
every threshold is stated here rather than buried in a config, because a marker
detector whose thresholds cannot be read is a marker detector whose results
cannot be interpreted.

**Provenance.** These are heuristic (`pseudo`) labels, not the ASLLRP SignStream
non-manual annotations, which are not in the mirror. M3 Track A's disclosure
applies to everything derived from this module: every downstream table marks it
as a pseudo-label. Track B (BU access) replaces these with real annotations and
re-runs.

**Availability, measured on the EmoSign 200.** Nine coefficients carry exactly
zero variance on this corpus and cannot support a rule:
``_neutral``, ``cheekPuff``, ``cheekSquintLeft/Right``, ``jawForward``,
``jawRight``, ``mouthFrownRight``, ``noseSneerLeft/Right``. That kills the
canonical facial-negation rule (nose wrinkle plus tongue out) outright - there is
no nose wrinkle to detect. The negation marker is therefore carried by the head
shake, which is how negation is marked in ASL anyway, and the limitation is
stated rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from seam.perception.tasks_api import BLENDSHAPE_INDEX

# ---------------------------------------------------------------------------
# Coefficient groupings
# ---------------------------------------------------------------------------

#: Raised brows: the yes/no question marker, and topic prominence.
BROW_RAISE = ("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
#: Lowered brows: the wh-question furrow, and also anger.
BROW_FURROW = ("browDownLeft", "browDownRight")
#: Widened eyes, which co-occur with the yes/no question and with surprise.
EYE_WIDE = ("eyeWideLeft", "eyeWideRight")
#: The pursed / rounded mouth shape. In ASL a wh-question is conventionally
#: marked with furrowed brows *and* pursed lips; a lowered mouth plus rounded
#: lips also covers the mouthing of rounded vowels.
MOUTH_MORPHEME = ("mouthPucker", "mouthFunnel", "mouthShrugLower", "mouthShrugUpper")
#: Open mouth: mouthing of vowels, and breath.
MOUTH_OPEN = ("jawOpen", "mouthClose")
#: Smile. Affect-leaning, used here as a control marker: a marker that should
#: *not* produce a negative shift.
MOUTH_POSITIVE = ("mouthSmileLeft", "mouthSmileRight")

#: Coefficients with zero measured variance on the EmoSign 200. Any rule naming
#: one of these is silently inert, so they are listed here for the audit to
#: assert against.
DEAD_COEFFICIENTS = (
    "_neutral",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "jawForward",
    "jawRight",
    "mouthFrownRight",
    "noseSneerLeft",
    "noseSneerRight",
)

#: Marker names, in a fixed order so matrices and tables line up.
MARKERS = (
    "brow_raise",  # yes/no question, topic marking
    "brow_furrow",  # wh-question
    "mouth_morpheme",  # wh-question co-articulation, mouthing
    "head_shake",  # negation
    "head_nod",  # affirmation, emphasis
    "mouth_positive",  # control: should not read as negative
)

_IDX = BLENDSHAPE_INDEX


def _mean(bs: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Mean of the named coefficients over the leading (time) axis."""
    cols = [_IDX[n] for n in names]
    return bs[:, cols].mean(axis=1)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MarkerThresholds:
    """Detection thresholds for each marker.

    Defaults are set from the measured EmoSign distributions: a marker fires when
    its score exceeds ``baseline + k * spread``, where baseline and spread are
    computed per clip. That makes the detector adaptive to a signer's habitual
    expression level, which a fixed absolute threshold cannot be - a signer who
    holds a mild furrow throughout would otherwise be "marked" on every frame
    and the audit would compare noise.
    """

    #: How many robust spreads above the clip's own median a score must reach.
    k: float = 1.5
    #: Floor on the robust spread, so a near-constant clip cannot produce a
    #: vanishing denominator and fire on numerical noise.
    min_spread: float = 0.01
    #: Minimum fraction of frames in a window that must fire for the window to
    #: count as marker-bearing.
    min_window_fraction: float = 0.25
    #: Head rotation must exceed this many radians to count as a shake/nod.
    #: ~20 degrees: enough to exclude tracking jitter, small enough to catch a
    #: real negation beat.
    head_angle: float = 0.35
    #: A shake needs at least this many direction reversals in the yaw signal.
    min_reversals: int = 2

    def scaled(self, factor: float) -> MarkerThresholds:
        return MarkerThresholds(
            k=self.k * factor,
            min_spread=self.min_spread,
            min_window_fraction=self.min_window_fraction,
            head_angle=self.head_angle,
            min_reversals=self.min_reversals,
        )


DEFAULT_THRESHOLDS = MarkerThresholds()


# ---------------------------------------------------------------------------
# Scores
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class MarkerSignals:
    """Raw, per-frame marker evidence before thresholding."""

    brow_raise: np.ndarray
    brow_furrow: np.ndarray
    mouth_morpheme: np.ndarray
    head_shake: np.ndarray
    head_nod: np.ndarray
    mouth_positive: np.ndarray

    def as_matrix(self) -> np.ndarray:
        """(T, 6) in ``MARKERS`` order."""
        return np.stack([getattr(self, m) for m in MARKERS], axis=1).astype(np.float32)

    def as_dict(self) -> dict[str, np.ndarray]:
        return {m: getattr(self, m) for m in MARKERS}


def _euler_from_matrix(rot: np.ndarray) -> np.ndarray:
    """Extract (roll, pitch, yaw) in radians from (T, 4, 4) transforms.

    MediaPipe's facial transformation matrix maps canonical face space to camera
    space, so the rotation block is orthonormal and the extraction below is the
    standard decomposition. The straight-up case, where the pitch solution is
    singular, is resolved with ``np.where`` rather than an ``if``: the input is a
    per-frame array, and a scalar branch on it raises rather than degrading.
    """
    if rot is None or len(rot) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    r = np.asarray(rot, dtype=np.float64)[:, :3, :3]

    pitch = np.arcsin(np.clip(-r[:, 2, 0], -1.0, 1.0))
    cos_pitch = np.cos(pitch)
    singular = np.abs(cos_pitch) < 1e-6

    roll_normal = np.arctan2(r[:, 2, 1], r[:, 2, 2])
    yaw_normal = np.arctan2(r[:, 2, 0], r[:, 1, 0])
    yaw_singular = np.arctan2(-r[:, 0, 1], r[:, 1, 1])

    roll = np.where(singular, 0.0, roll_normal)
    yaw = np.where(singular, yaw_singular, yaw_normal)
    return np.stack([roll, pitch, yaw], axis=1)


def _oscillation(signal: np.ndarray, fps: float, thresholds: MarkerThresholds) -> np.ndarray:
    """Per-frame evidence of a directional oscillation in ``signal``.

    Fires where the signal has moved away from its own running baseline and is
    heading back, i.e. at the turning points of a shake or a nod. The magnitude
    is the distance from baseline, so a large-amplitude shake scores higher than
    a small one.
    """
    if len(signal) < 3 or fps <= 0:
        return np.zeros(len(signal), dtype=np.float64)

    baseline = float(np.median(signal))
    centred = signal - baseline
    derivative = np.gradient(centred) * fps

    # A turning point: derivative sign flip while displaced from baseline.
    turning = np.zeros(len(signal), dtype=bool)
    for i in range(1, len(signal) - 1):
        if (
            np.sign(derivative[i - 1]) != np.sign(derivative[i + 1])
            and np.sign(derivative[i - 1]) != 0
        ):
            turning[i] = True

    magnitude = np.abs(centred)
    # Threshold at ``head_angle`` itself, as the field documents (~20 degrees).
    # The previous code used ``head_angle / 2.0``, so the effective threshold was
    # 10 degrees - half the stated one - which is well inside the tracking jitter
    # of a 256x256 face. Measured consequence: head_shake fired on 98% of
    # EmoSign clips and head_nod on 65%, so neither carried information.
    active = turning & (magnitude > thresholds.head_angle)

    # Require ``min_reversals`` *large-amplitude* turning points, not merely
    # ``min_reversals`` sign changes somewhere in the clip. The old test was
    # clip-global: one qualifying sign change anywhere licensed every noisy
    # turning point in the clip, so a single wobble anywhere turned the whole
    # signal into "a shake". An oscillation is several large deflections, so the
    # count has to be of the same kind of event the detector fires on.
    if int(active.sum()) < thresholds.min_reversals:
        return np.zeros(len(signal), dtype=np.float64)
    return np.where(active, magnitude, 0.0)


def signals(
    blendshapes: np.ndarray,
    rotation: np.ndarray | None = None,
    fps: float = 25.0,
    thresholds: MarkerThresholds = DEFAULT_THRESHOLDS,
) -> MarkerSignals:
    """Compute per-frame marker evidence.

    Blendshape scores are already normalized to the face, so no further
    normalization is applied - only the adaptive thresholding in
    :func:`fire`, which happens per window rather than per clip so that a marker
    in one window is not diluted over a whole clip.
    """
    bs = np.asarray(blendshapes, dtype=np.float64)
    if bs.ndim != 2 or bs.shape[1] < 52:
        raise ValueError(f"expected (T, 52) blendshapes, got {bs.shape}")

    euler = _euler_from_matrix(rotation) if rotation is not None else np.zeros((len(bs), 3))
    if len(euler) != len(bs):
        euler = np.zeros((len(bs), 3))

    return MarkerSignals(
        brow_raise=_mean(bs, BROW_RAISE),
        brow_furrow=_mean(bs, BROW_FURROW),
        mouth_morpheme=_mean(bs, MOUTH_MORPHEME),
        head_shake=_oscillation(euler[:, 2], fps, thresholds),
        head_nod=_oscillation(euler[:, 1], fps, thresholds),
        mouth_positive=_mean(bs, MOUTH_POSITIVE),
    )


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------


def _robust_spread(x: np.ndarray, floor: float) -> float:
    """Median absolute deviation, scaled to be comparable to a standard deviation."""
    if len(x) < 2:
        return floor
    mad = float(np.median(np.abs(x - np.median(x)))) * 1.4826
    return max(mad, floor)


def fire(
    sig: MarkerSignals,
    thresholds: MarkerThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, np.ndarray]:
    """Threshold each marker against its own clip-relative baseline.

    The rule is ``score > median + k * MAD``. It is deliberately self-calibrating
    rather than an absolute cut: signers differ enormously in habitual brow
    position, and a fixed threshold would label a whole signer as
    marker-bearing, at which point the audit would be measuring the signer.
    """
    out: dict[str, np.ndarray] = {}
    for name in MARKERS:
        x = getattr(sig, name)
        if name in ("head_shake", "head_nod"):
            # Already an absolute-evidence signal: zeros mean "no oscillation".
            out[name] = (x > 0).astype(bool)
            continue
        baseline = float(np.median(x))
        spread = _robust_spread(x, thresholds.min_spread)
        out[name] = x > baseline + thresholds.k * spread
    return out


def window_fired(
    sig: MarkerSignals,
    edges: np.ndarray,
    window: int,
    thresholds: MarkerThresholds = DEFAULT_THRESHOLDS,
) -> np.ndarray:
    """Per-window marker flags.

    Returns ``(n_windows, len(MARKERS))`` boolean. A window counts as
    marker-bearing for a given marker when the marker fires in at least
    ``min_window_fraction`` of its frames, which tolerates the single-frame
    dropouts a tracker produces without letting a 1-frame spike mark a window.
    """
    fired = fire(sig, thresholds)
    out = np.zeros((len(edges), len(MARKERS)), dtype=bool)
    for i, start in enumerate(edges):
        sl = slice(int(start), int(start) + window)
        for j, name in enumerate(MARKERS):
            flag = fired[name][sl]
            if len(flag) and flag.mean() >= thresholds.min_window_fraction:
                out[i, j] = True
    return out


def describe() -> dict[str, object]:
    """Machine-readable description of the marker set, for the paper's appendix."""
    return {
        "provenance": "heuristic (pseudo-labels); ASLLRP SignStream XML not available",
        "markers": list(MARKERS),
        "coefficients": {
            "brow_raise": list(BROW_RAISE),
            "brow_furrow": list(BROW_FURROW),
            "eye_wide": list(EYE_WIDE),
            "mouth_morpheme": list(MOUTH_MORPHEME),
            "mouth_open": list(MOUTH_OPEN),
            "mouth_positive": list(MOUTH_POSITIVE),
        },
        "thresholds": {
            "k": DEFAULT_THRESHOLDS.k,
            "min_spread": DEFAULT_THRESHOLDS.min_spread,
            "min_window_fraction": DEFAULT_THRESHOLDS.min_window_fraction,
            "head_angle_rad": DEFAULT_THRESHOLDS.head_angle,
            "min_reversals": DEFAULT_THRESHOLDS.min_reversals,
            "rule": "score > median + k * MAD, computed per clip",
        },
        "dead_coefficients": list(DEAD_COEFFICIENTS),
        "unavailable_markers": {
            "facial_negation": (
                "the canonical nose-wrinkle + tongue-out negation display is not "
                "detectable here: noseSneer*, cheekPuff and jawForward carry zero "
                "variance on the EmoSign corpus. Negation is carried by the head "
                "shake alone."
            )
        },
    }
