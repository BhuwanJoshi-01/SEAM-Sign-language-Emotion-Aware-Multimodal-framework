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

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seam.features.head_motion import (
    BROWSER_FULL_SCALE,
    NOD,
    TILT,
    TURN,
    head_angles,
    live_band_rms,
    reversal_score,
)
from seam.perception.tasks_api import BLENDSHAPE_INDEX

#: The three averaging times of the head detector, in seconds: the member of the declared
#: grid that `scripts/validate_markers.py` chose. The live page uses the same three.
HEAD_FAST_S, HEAD_SLOW_S, HEAD_HOLD_S = 0.08, 0.4, 0.4

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

#: What M3 measured about each marker **on the EmoSign 200**, recorded per marker so it
#: cannot be re-guessed at display time.
#:
#: Two properties, both measured, and they are not the same property:
#:
#: ``prevalence``
#:     Fraction of corpus clips on which the clip-level presence criterion fires. A
#:     marker present on ~90% of clips cannot discriminate *between* clips.
#: ``zero_fraction``
#:     Fraction of corpus clips on which the marker is identically zero, which makes it
#:     **blind**: no conclusion is available from it, and its non-separation from a
#:     label is not evidence about the label.
#:
#: ``state`` is the label the project uses for each, and only three words are allowed:
#:
#: ``blind``
#:     Cannot discriminate at all. Both head markers. Report as unavailable, never as a
#:     null.
#: ``degenerate``
#:     Informative but near-constant across clips, so any agreement statistic sits near
#:     the base rate. Three of six, including ``mouth_positive`` - which is supposed to be
#:     the control, and a control present on 90% of clips is not a control.
#: ``usable``
#:     Fires on a minority of clips and can separate them. ``brow_furrow`` only.
#:
#: **These are properties of a corpus, not of any window.** A live window's own zero
#: fraction is a different measurement and must never be substituted for these. Doing so
#: is the bug this table exists to prevent: the demo originally decided blindness in the
#: browser by testing ``zero_fraction > 0.6``, a threshold nobody chose and nobody
#: justified, which made a generated test signal report the corpus's blindness back to
#: the reader as though the instrument had measured it.
#:
#: ``head_shake`` was **re-measured** and its recorded value was wrong. See
#: ``paper/EXPERIMENT_LOG.md`` for the entry that withdrew the negation/head-shake
#: correlation; the short version is that ``_euler_from_matrix`` extracted yaw from the
#: wrong matrix entries, so the channel ``head_shake`` reads was identically zero for a
#: pure yaw. The recorded 0.785 was an artifact of that and the correct figure is 0.100.
#: Every other row is unchanged by the fix, which is the check that it moved the rotation
#: path and nothing else.
#:
#: Source: ``paper/EXPERIMENT_LOG.md`` entries 27 (prevalence) and 61 (blind), plus the
#: re-measurements after the yaw fix and after the head-axis fix. The numbers live in
#: ``marker_census.json`` beside this file, not in this module: a measurement recorded in
#: the code that produced it changes that code's fingerprint, and every artifact computed
#: with it would then read as stale for having been measured.
MEASURED_DEGENERACY: dict[str, dict[str, object]] = json.loads(
    Path(__file__).with_name("marker_census.json").read_text(encoding="utf-8")
)["markers"]

#: A marker is blind when no conclusion is available from it. Named as a predicate rather
#: than re-tested at each call site so the threshold lives with the measurement.
BLIND_STATE = "blind"


def is_blind(name: str) -> bool:
    """Whether M3 measured this marker as unable to discriminate.

    Reads :data:`MEASURED_DEGENERACY`, never a live signal: a window with no energy in a
    channel says the window is quiet, not that the instrument is blind.
    """
    row = MEASURED_DEGENERACY.get(name)
    return row is not None and row["state"] == BLIND_STATE


_IDX = BLENDSHAPE_INDEX


def _mean(bs: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """Mean of the named coefficients over the leading (time) axis."""
    cols = [_IDX[n] for n in names]
    return bs[:, cols].mean(axis=1)


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ClipCriteria:
    """What counts as a marker being *present in a clip*.

    This is a different instrument from :func:`fire`, and the distinction is the
    whole point. ``fire`` answers "is this frame elevated?", which is the right
    question for locating a marker in time. Asking "does this clip contain the
    marker?" by OR-ing per-frame flags over 85 frames is not the same question,
    and the answer it produced was degenerate: 4 of 6 markers came out present on
    79-95% of clips, including the ``mouth_positive`` control at 0.91. A marker
    present on nearly every clip cannot discriminate between clips, and its
    agreement with anything sits at the base rate - which looks like a careful null
    and is actually a broken measurement.

    So a clip-level label requires all three of:

    * **duration** - the longest *contiguous* run of elevated frames, in ms. ASL
      non-manual markers are held, not flickered; a single-frame spike is tracker
      noise, and so is a 30 ms blip.
    * **peak** - the largest excursion relative to the clip's own dynamic range, so
      the bar scales with how expressive that signer is on that clip. Without this,
      a flat clip is trivially "marked" by any absolute threshold, and a
      heavy-tailed signal is trivially "unmarked" by a MAD-based one.
    * **coverage** - the fraction of frames elevated, so a clip with 40 scattered
      two-frame spikes does not pass on duration alone.

    Defaults were chosen to satisfy a control criterion fixed in advance — see
    :func:`calibrate` — and not to move any agreement statistic.
    """

    #: Longest contiguous elevated run required, in milliseconds.
    min_run_ms: float = 200.0
    #: Peak excursion required, as a fraction of the clip's own p95-p50 range
    #: (head markers: radians). ``1.0`` means "reaches the top of this clip's own
    #: range for this channel".
    min_peak_mads: float = 0.8
    #: Minimum fraction of the clip that must be elevated.
    min_coverage: float = 0.04
    #: Per-frame bar, as a fraction of the clip's own dynamic range, above which a
    #: blendshape frame counts as elevated.
    #:
    #: This is deliberately *not* ``MarkerThresholds.k``. That is 1.5 in units of
    #: MAD, a different scale from the clip-range units :func:`clip_evidence`
    #: returns, and because the evidence is bounded near 1.0 the two cannot be
    #: compared - a 1.5 bar is unreachable, so every blendshape marker reported
    #: zero coverage and zero duration and the whole clip-level view silently
    #: reported "absent" for four of six markers. Scale is not a detail here: a bar
    #: in the wrong units is an instrument that always reads zero, and zero looks
    #: like a finding.
    min_frame_evidence: float = 0.5

    def describe(self) -> dict[str, float]:
        return {
            "min_run_ms": self.min_run_ms,
            "min_peak_mads": self.min_peak_mads,
            "min_coverage": self.min_coverage,
            "min_frame_evidence": self.min_frame_evidence,
        }


DEFAULT_CLIP_CRITERIA = ClipCriteria()


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
    #: Back-and-forth energy of the head turn, in degrees RMS, above which a shake counts.
    #: Not a guess: `scripts/validate_markers.py` sets it where frames the annotators did
    #: not mark exceed it a quarter of the time (``shipped_in_page``), and a test holds
    #: this number to that artifact. The earlier rule asked for 20 degrees of displacement,
    #: which a fluent signer's head shake, a few degrees wide, never reaches.
    shake_rms_deg: float = 2.44
    #: The same for the nod axis.
    nod_rms_deg: float = 3.01
    #: The angle must also have gone there and back: this many reversals ...
    min_reversals: int = 2
    #: ... of at least this many degrees each, inside the detector's window. One quick
    #: turn of the head has energy and no reversal, and is not a shake.
    swing_deg: float = 2.0

    def scaled(self, factor: float) -> MarkerThresholds:
        return MarkerThresholds(
            k=self.k * factor,
            min_spread=self.min_spread,
            min_window_fraction=self.min_window_fraction,
            shake_rms_deg=self.shake_rms_deg,
            nod_rms_deg=self.nod_rms_deg,
            min_reversals=self.min_reversals,
            swing_deg=self.swing_deg,
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
    """(roll, pitch, yaw) in radians from (T, 4, 4) transforms, named for a head.

    * ``roll`` - the head tilting, ear towards shoulder;
    * ``pitch`` - the head nodding, chin up or down;
    * ``yaw`` - the head turning left or right, which is the axis a head shake moves.

    MediaPipe's facial transformation matrix maps canonical face space to camera space.
    In that space the face looks along z with y up, so the three angles are read from the
    direction the face points (the third column) and its right-hand direction (the first),
    by `seam.features.head_motion.head_angles`.

    Two earlier versions of this function were wrong in ways no test of theirs could see.
    The first read yaw from a pair of entries that are not a rotation-matrix pair for it,
    and returned 0.0 for any pure yaw. The second was a correct ZYX Euler decomposition
    with the aerospace names, where the forward axis is x: its "yaw" was a head tilt and
    its "pitch" a head turn, so `head_shake` measured tilting and `head_nod` measured
    turning. Both were tested with rotations built in the same convention as the code.
    The check that settles it involves no rotation matrix at all: the nose moves sideways
    across the face landmarks when ``yaw`` here changes, and the line through the eyes
    rotates when ``roll`` does (`scripts/validate_markers.py`, ``axis_check``).
    """
    if rot is None or len(rot) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    angles = np.radians(head_angles(np.asarray(rot, dtype=np.float64)))
    return np.stack([angles[:, TILT], angles[:, NOD], angles[:, TURN]], axis=1)


def _oscillation(
    angle: np.ndarray, fps: float, on_deg: float, thresholds: MarkerThresholds
) -> np.ndarray:
    """Per-frame evidence that a head angle (radians) is going back and forth.

    The detector that was scored against human annotation, not a rule of thumb: the RMS of
    the oscillating part of the angle (`head_motion.live_band_rms`) must reach ``on_deg``,
    and the angle must have reversed ``min_reversals`` times by ``swing_deg`` inside the
    window (`head_motion.reversal_score`). Where both hold the evidence is that RMS, in
    radians; elsewhere it is exactly zero, which is what :func:`fire` reads as "no
    oscillation". It is causal, so the same code can run on a live stream.

    What this replaces asked for turning points displaced 20 degrees from the clip's
    median. On the 200 EmoSign clips that fires on almost nothing a linguist marked.
    """
    if len(angle) < 3 or fps <= 0:
        return np.zeros(len(angle), dtype=np.float64)
    deg = np.degrees(np.asarray(angle, dtype=np.float64))
    energy = live_band_rms(deg, fps, fast_s=HEAD_FAST_S, slow_s=HEAD_SLOW_S, hold_s=HEAD_HOLD_S)
    swing = reversal_score(deg, fps, min_deg=thresholds.swing_deg)
    swinging = swing >= thresholds.min_reversals / BROWSER_FULL_SCALE - 1e-9
    return np.where((energy >= on_deg) & swinging, np.radians(energy), 0.0)


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
        head_shake=_oscillation(euler[:, 2], fps, thresholds.shake_rms_deg, thresholds),
        head_nod=_oscillation(euler[:, 1], fps, thresholds.nod_rms_deg, thresholds),
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


# ---------------------------------------------------------------------------
# Clip-level presence
# ---------------------------------------------------------------------------


def clip_evidence(
    sig: MarkerSignals, thresholds: MarkerThresholds = DEFAULT_THRESHOLDS
) -> dict[str, np.ndarray]:
    """Per-frame excursion above the clip's own baseline, in MADs.

    Blendshape markers are normalised by the clip's own **dynamic range**,
    ``p95 - p50``, rather than by MAD. MAD is the wrong scale for these signals:
    a blendshape coefficient spends most frames near its floor and peaks
    occasionally, so the MAD is dominated by the flat part and the ratio
    ``(x - median) / MAD`` explodes at the peak. Measured over the 200 EmoSign
    clips, MAD normalisation put ``mouth_positive`` at a median peak of **27
    "MADs"** while ``head_nod`` sat at 0 — a 27:0 spread in which no single
    threshold can be both meaningful and shared. Normalising by the observed
    range instead asks the scale-free question that actually matters: *does this
    frame approach the top of this clip's own range on this channel?*

    Head markers are already absolute-evidence signals in radians and stay in
    radians, so the peak threshold is in different units per marker family, which
    is why :func:`clip_presence` reports the unit.

    **Known limitation.** The baseline is the clip's own median, so a marker
    occupying more than about half the clip drags the baseline up with it and its
    measured range collapses. Measured on a 20-frame marker: at 20/40 frames the
    baseline is 0.50 and the reported mean is 0.500, at 20/80 frames the baseline
    is 0.0 and the mean is 0.250 - the same marker reads as twice as strong
    because the clip is shorter. For ASL non-manual markers, which are minority
    events within an utterance, this does not bite; for a clip-long facial
    expression it would, and the fix would be a baseline taken from a
    signer- or corpus-level distribution rather than from the clip.
    """
    out: dict[str, np.ndarray] = {}
    for name in MARKERS:
        x = np.asarray(getattr(sig, name), dtype=np.float64)
        if name in ("head_shake", "head_nod"):
            out[name] = np.maximum(x, 0.0)
            continue
        if not len(x):
            out[name] = np.zeros(0, dtype=np.float64)
            continue
        base = float(np.median(x))
        scale = float(np.percentile(x, 95)) - base
        if scale <= 0:
            # A flat clip has no range to be expressive within; nothing is
            # elevated. Returning zeros rather than a floored ratio avoids
            # manufacturing evidence out of numerical noise.
            out[name] = np.zeros(len(x), dtype=np.float64)
            continue
        out[name] = np.clip((x - base) / scale, 0.0, None)
    return out


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest contiguous True run."""
    if not mask.size:
        return 0
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def clip_presence(
    sig: MarkerSignals,
    fps: float,
    *,
    thresholds: MarkerThresholds = DEFAULT_THRESHOLDS,
    criteria: ClipCriteria = DEFAULT_CLIP_CRITERIA,
) -> dict[str, dict[str, float | bool | str]]:
    """Per-marker clip-level presence, with the evidence that produced it.

    Returns, for each marker, the three measured quantities and the verdict. The
    quantities are returned even when the verdict is False, so a prevalence figure
    can be diagnosed instead of merely observed.
    """
    ev = clip_evidence(sig, thresholds)
    need = criteria.min_run_ms * fps / 1000.0
    out: dict[str, dict[str, float | bool | str]] = {}
    for name in MARKERS:
        x = ev[name]
        # Per-frame bar, in the units clip_evidence actually returns. Blendshapes: a
        # fraction of the clip's own range. Head markers: the detector has already
        # applied its own threshold and returns exactly zero below it, so any evidence at
        # all is an active frame and the peak criterion has nothing left to add. (The
        # earlier bar here was 0.35 rad per frame and a 0.8 rad peak - 46 degrees - which
        # no head shake reaches, so the clip-level view could not see one.)
        head = name in ("head_shake", "head_nod")
        bar = 0.0 if head else criteria.min_frame_evidence
        mask = x > bar
        run = _longest_run(mask)
        peak = float(x.max()) if x.size else 0.0
        coverage = float(mask.mean()) if x.size else 0.0
        peak_ok = peak > 0.0 if head else peak >= criteria.min_peak_mads
        present = bool(run >= need and peak_ok and coverage >= criteria.min_coverage)
        out[name] = {
            "present": present,
            "run_frames": float(run),
            "run_ms": float(run * 1000.0 / fps) if fps > 0 else 0.0,
            "peak": peak,
            "coverage": coverage,
            "peak_unit": "rad" if name in ("head_shake", "head_nod") else "clip_range",
        }
    return out


def clip_magnitude(
    sig: MarkerSignals,
    fps: float,
    *,
    thresholds: MarkerThresholds = DEFAULT_THRESHOLDS,
    reduce: str = "mean",
) -> dict[str, float]:
    """A continuous per-clip marker magnitude, 0 = absent, larger = more.

    Binary clip-level presence turned out to be the wrong representation for these
    signals, for a reason worth stating plainly: ASL non-manual markers are
    *graded and frequent*, not rare events. A brow raise marks topics, contrast and
    yes/no questions, so it is present in most signing rather than in a minority of
    clips, and the control ``mouth_positive`` is active in essentially every clip
    because mouths move while people sign. A prevalence screen therefore cannot
    separate "frequent marker" from "useless marker" — the two look identical, and
    the screen rejected both.

    A magnitude does not have that failure mode. "How strongly does this clip raise
    its brows?" is a well-posed question with a sensible null, and it can be tested
    against a syntactic label by permutation, which needs no prevalence assumption
    at all. So the magnitude is the primary statistic and the binary flag is
    retained only for reporting.

    ``reduce="mean"`` (the default) averages over frames and is therefore
    **duration-free**. ``reduce="total"`` integrates instead, and is a trap: every
    syntactic label correlates with clip length (measured on the 200 clips:
    interrogative 5.77s vs 4.48s, r_pb = +0.68; negation r_pb = +0.54), because
    questions and negated statements are simply longer utterances. An integral
    therefore inherits that confound mechanically, and the first version of this
    analysis reported interrogative/brow_raise at r_pb = +0.414 on the integral and
    **-0.045** on the mean - the entire association was clip length. The mean is
    the default so that mistake cannot be made silently, and the association test
    additionally controls for duration.
    """
    ev = clip_evidence(sig, thresholds)
    out: dict[str, float] = {}
    for name in MARKERS:
        x = ev[name]
        if not x.size:
            out[name] = 0.0
            continue
        if reduce == "total":
            # Integral of the excursion. Confounded by clip length; see the
            # docstring. Kept available for a within-clip duration analysis, not
            # for cross-clip comparison.
            out[name] = float(np.sum(x) / max(fps, 1e-6))
        elif reduce == "mean":
            out[name] = float(np.mean(x)) if x.size else 0.0
        else:
            raise ValueError(f"reduce must be 'mean' or 'total', got {reduce!r}")
    return out


def clip_prevalence(
    records: Sequence[tuple[MarkerSignals, float]],
    *,
    thresholds: MarkerThresholds = DEFAULT_THRESHOLDS,
    criteria: ClipCriteria = DEFAULT_CLIP_CRITERIA,
) -> dict[str, float]:
    """Marker prevalence over ``(signals, fps)`` pairs under these criteria."""
    if not records:
        return dict.fromkeys(MARKERS, float("nan"))
    counts: dict[str, int] = dict.fromkeys(MARKERS, 0)
    for sig, fps in records:
        pres = clip_presence(sig, fps, thresholds=thresholds, criteria=criteria)
        for name, r in pres.items():
            if r["present"]:
                counts[name] += 1
    n = len(records)
    return {m: counts[m] / n for m in MARKERS}


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
            "shake_rms_deg": DEFAULT_THRESHOLDS.shake_rms_deg,
            "nod_rms_deg": DEFAULT_THRESHOLDS.nod_rms_deg,
            "swing_deg": DEFAULT_THRESHOLDS.swing_deg,
            "min_reversals": DEFAULT_THRESHOLDS.min_reversals,
            "rule": "score > median + k * MAD, computed per clip",
        },
        "dead_coefficients": list(DEAD_COEFFICIENTS),
        "measured_degeneracy": {k: dict(v) for k, v in MEASURED_DEGENERACY.items()},
        "degeneracy_source": (
            "EmoSign 200 (M3): paper/EXPERIMENT_LOG.md entries 27 and 61. Corpus "
            "properties, not per-window measurements."
        ),
        "unavailable_markers": {
            "facial_negation": (
                "the canonical nose-wrinkle + tongue-out negation display is not "
                "detectable here: noseSneer*, cheekPuff and jawForward carry zero "
                "variance on the EmoSign corpus. Negation is carried by the head "
                "shake alone."
            )
        },
    }
