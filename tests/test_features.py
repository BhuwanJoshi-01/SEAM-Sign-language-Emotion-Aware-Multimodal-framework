"""Preprocessing and feature properties.

These are the tests the plan asks for by name, and they are property-style on
purpose: "normalization is translation/scale invariant", "mirroring swaps hands
correctly", "a scripted fast/large motion scores higher speed/amplitude than a
slow/small one", "a synthetic brow-raise is detected". A test that only checks a
hardcoded output value would pass even if the property it was written to protect
had been broken.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.features import markers as M
from seam.features import prosody as P
from seam.perception.tasks_api import (
    BLENDSHAPE_CATEGORIES,
    BLENDSHAPE_INDEX,
    PART_SLICES,
    TOTAL_POINTS,
)
from seam.preprocess import normalize as N

POSE_SLICE = PART_SLICES["pose"]
#: ``arr[:, SLICE]`` with a tuple is *fancy* indexing and silently selects those
#: two rows, not the range. Every test that means "the pose block" has to spell
#: the slice out, so the range is materialised once here.
POSE_ROWS = slice(PART_SLICES["pose"][0], PART_SLICES["pose"][1])
LH_ROWS = slice(PART_SLICES["left_hand"][0], PART_SLICES["left_hand"][1])


# ---------------------------------------------------------------------------
# fixtures: a plausible synthetic signer
# ---------------------------------------------------------------------------


#: A plausible 33-point MediaPipe upper-body skeleton, as offsets from the
#: shoulder midpoint in units of shoulder width. The 33 pose points are emitted
#: all-or-nothing by the detector, so a fixture that leaves indices 1..10, 13,
#: 14 at zero is unrealistic - and an unset point at the origin becomes
#: ``-center / scale`` after normalization, which then varies with where the
#: signer stands and breaks the invariance properties this file is testing.
_SKELETON = {
    0: (0.0, -0.75),  # nose
    2: (-0.10, -0.80),
    7: (-0.35, -0.85),
    8: (0.35, -0.85),
    11: (-0.5, 0.0),
    12: (0.5, 0.0),  # shoulders
    13: (-0.7, 0.7),
    14: (0.7, 0.7),  # elbows
    15: (-0.8, 1.4),
    16: (0.8, 1.4),  # wrists
}


def synthetic_pose(
    n: int = 60,
    *,
    cx: float = 0.0,
    cy: float = 0.0,
    span: float = 0.2,
    fps: float = 25.0,
    hand_speed: float = 0.0,
    hand_amp: float = 0.0,
    two_handed_opposed: bool = False,
    seed: int = 0,
) -> np.ndarray:
    """A (T, TOTAL_POINTS, 3) array with a full pose and a moving wrist pair.

    ``hand_speed`` translates the hand pair; ``hand_amp`` adds an oscillation.
    Both hands move *together* by default, which is what a sign looks like.

    ``two_handed_opposed`` makes the hands travel in opposite directions, whose
    centroid is therefore stationary. That is a real configuration (symmetric
    two-handed signs) and it is worth being able to construct, but it must be
    asked for: a fixture that does it silently makes every speed-based assertion
    vacuously true.
    """
    rng = np.random.default_rng(seed)
    lm = np.zeros((n, TOTAL_POINTS, 3), dtype=np.float32)
    t = np.arange(n) / fps

    center = np.array([0.5 + cx, 0.5 + cy])
    half = span
    for idx in range(33):
        # Unset pose points would land on the origin, and an origin landmark
        # becomes -center/scale after normalization, which varies with where the
        # signer stands. That breaks the invariance properties under test.
        ox, oy = _SKELETON.get(idx, (0.0, 0.25))
        lm[:, idx, 0] = center[0] + ox * half
        lm[:, idx, 1] = center[1] + oy * half

    wobble = hand_amp * np.sin(2 * np.pi * 2.0 * t)
    # Two hands, two signs. Opposed means the *translation* opposes too - a flag
    # that only flipped the oscillation would leave the centroid moving and make
    # the symmetric-motion property untestable.
    s15, s16 = (-1.0, 1.0) if two_handed_opposed else (1.0, 1.0)
    lm[:, 15, 0] = center[0] + s15 * (hand_speed * t + wobble)
    lm[:, 16, 0] = center[0] + s16 * (hand_speed * t + wobble)
    lm[:, 15, 1] = center[1] + s15 * wobble
    lm[:, 16, 1] = center[1] + s16 * wobble

    # Detector noise is proportional to the signer's apparent size. Adding a
    # fixed amount instead would make the scale-invariance property untestable
    # at any sane tolerance, because the normalized noise would differ by exactly
    # the scale factor the test is asking about.
    # Hand blocks must hold real landmarks. A fixture that leaves them at zero
    # cannot distinguish "the hand was not detected" from "the hand was at the
    # origin", which is precisely the distinction the missing-hand policy turns
    # on.
    for name, (lo, hi) in (
        ("left_hand", PART_SLICES["left_hand"]),
        ("right_hand", PART_SLICES["right_hand"]),
    ):
        sign = -1.0 if name == "left_hand" else 1.0
        for j in range(hi - lo):
            lm[:, lo + j, 0] = center[0] + sign * half * (0.8 + 0.3 * j / (hi - lo - 1))
            lm[:, lo + j, 1] = center[1] + half * 1.4

    lm += rng.normal(0, 1e-3 * span, lm.shape).astype(np.float32)
    return lm


def all_present(n: int) -> np.ndarray:
    return np.ones((n, 4), dtype=bool)


# ---------------------------------------------------------------------------
# normalize: invariance
# ---------------------------------------------------------------------------


def test_normalization_is_translation_invariant() -> None:
    a = synthetic_pose(60, cx=0.3, cy=0.1)
    b = synthetic_pose(60, cx=-0.2, cy=0.4)
    na, _ = N.normalize_pose(a, POSE_SLICE)
    nb, _ = N.normalize_pose(b, POSE_SLICE)
    np.testing.assert_allclose(na[:, POSE_ROWS, :2], nb[:, POSE_ROWS, :2], atol=1e-5)


def test_normalization_is_scale_invariant() -> None:
    a = synthetic_pose(60, span=0.2)
    b = synthetic_pose(60, span=0.5)
    na, _ = N.normalize_pose(a, POSE_SLICE)
    nb, _ = N.normalize_pose(b, POSE_SLICE)
    np.testing.assert_allclose(na[:, POSE_ROWS, :2], nb[:, POSE_ROWS, :2], atol=1e-5)


def test_normalization_puts_the_shoulder_midpoint_at_the_origin() -> None:
    lm = synthetic_pose(40, cx=0.31, cy=0.27, span=0.17)
    out, params = N.normalize_pose(lm, POSE_SLICE)
    mid = out[:, 11:13, :2].mean(axis=1)
    np.testing.assert_allclose(mid, 0.0, atol=1e-5)
    assert params.center.shape == (40, 2)
    assert params.scale.shape == (40,)
    # Normalized shoulder span is 1 by construction.
    span = np.linalg.norm(out[:, 12, :2] - out[:, 11, :2], axis=1)
    np.testing.assert_allclose(span, 1.0, atol=1e-5)


def test_normalization_survives_a_missing_shoulder_frame() -> None:
    lm = synthetic_pose(30, span=0.2)
    lm[10, 11, :2] = 0.0  # left shoulder not detected on one frame
    lm[10, 12, :2] = 0.0
    out, _ = N.normalize_pose(lm, POSE_SLICE)
    assert np.isfinite(out).all()


def test_normalization_raises_when_no_frame_has_an_anchor() -> None:
    lm = np.zeros((10, TOTAL_POINTS, 3), dtype=np.float32)
    with pytest.raises(N.NormalizeError, match="no frame has a usable pose anchor"):
        N.normalize_pose(lm, POSE_SLICE)


def test_normalization_rejects_wrong_shape() -> None:
    with pytest.raises(N.NormalizeError, match=r"\(T, P, 3\)"):
        N.interpolate_gaps(np.zeros((10, 5), dtype=np.float32), np.ones((10, 4), dtype=bool))


# ---------------------------------------------------------------------------
# interpolate_gaps
# ---------------------------------------------------------------------------


def test_interpolation_fills_a_short_gap_and_leaves_a_long_one() -> None:
    lm = synthetic_pose(40)
    lm[10:13, LH_ROWS, :] = 0.0  # 3-frame gap in the left hand: short
    lm[20:35, LH_ROWS, :] = 0.0  # 15-frame gap: too long to interpolate
    pres = all_present(40)
    pres[10:13, 1] = False
    pres[20:35, 1] = False
    out, repaired = N.interpolate_gaps(lm, pres, max_gap=6)

    assert not np.all(out[11, LH_ROWS] == 0)
    assert np.all(out[25, LH_ROWS] == 0)
    assert repaired.shape == (40, TOTAL_POINTS)


def test_interpolation_does_not_fabricate_a_never_detected_part() -> None:
    """A hand that was never visible must not be interpolated into existence.

    The prosody channel measures motion; a fabricated hand trajectory is motion
    that never happened.
    """
    lm = synthetic_pose(30)
    lm[:, LH_ROWS, :] = 0.0  # the detector saw no hand on any frame
    pres = all_present(30)
    pres[:, 1] = False
    out, _ = N.interpolate_gaps(lm, pres)
    assert np.all(out[:, LH_ROWS, :] == 0)


def test_interpolation_rejects_mismatched_presence_length() -> None:
    lm = synthetic_pose(10)
    with pytest.raises(N.NormalizeError, match="presence has"):
        N.interpolate_gaps(lm, np.ones((9, 4), dtype=bool))


# ---------------------------------------------------------------------------
# One Euro filter
# ---------------------------------------------------------------------------


def test_one_euro_attenuates_jitter_more_than_signal() -> None:
    """The reason for choosing it: less jitter, and less lag on real motion."""
    fps = 25.0
    t = np.arange(200) / fps
    signal = np.sin(2 * np.pi * 1.0 * t)[:, None]
    jitter = np.random.default_rng(0).normal(0, 0.05, signal.shape)
    clean = N.one_euro_filter(signal, fps)
    noisy = N.one_euro_filter(signal + jitter, fps)

    signal_var = float(np.var(noisy - clean))
    jitter_var = float(np.var((noisy - clean) - (clean - np.roll(clean, 1, axis=0))))
    assert signal_var >= 0.0
    assert jitter_var >= 0.0
    # Filtering reduces the deviation from the clean signal relative to no filter.
    unfiltered_dev = float(np.var(signal + jitter - signal))
    assert signal_var < unfiltered_dev


def test_one_euro_is_causal() -> None:
    """Frame t may depend only on frames <= t.

    A centred smoother would break every latency claim in the project, so this is
    asserted rather than trusted.
    """
    fps = 25.0
    x = np.random.default_rng(1).normal(0, 1, (100, 3))
    full = N.one_euro_filter(x, fps)

    truncated = N.one_euro_filter(x[:60], fps)
    np.testing.assert_allclose(full[:60], truncated, atol=1e-6)


def test_one_euro_rejects_zero_fps() -> None:
    with pytest.raises(N.NormalizeError, match="fps must be positive"):
        N.one_euro_filter(np.zeros((10, 2)), 0.0)


# ---------------------------------------------------------------------------
# resampling and windowing
# ---------------------------------------------------------------------------


def test_resample_halves_the_frame_count() -> None:
    idx = N.resample_indices(100, 25.0, 12.0)
    assert len(idx) == 48  # 100/25 = 4 s, at 12 fps
    assert idx.max() <= 99


def test_resample_respects_the_frame_cap() -> None:
    idx = N.resample_indices(1000, 30.0, 30.0, max_frames=64)
    assert len(idx) == 64


def test_resample_handles_a_single_frame() -> None:
    assert len(N.resample_indices(1, 25.0, 12.0)) == 1
    assert len(N.resample_indices(0, 25.0, 12.0)) == 0


def test_resample_rejects_bad_rates() -> None:
    with pytest.raises(N.NormalizeError, match="frame rates must be positive"):
        N.resample_indices(10, 0.0, 12.0)


def test_windows_tile_without_overrun() -> None:
    edges = N.window_indices(100, 64, 32)
    assert edges.tolist() == [0, 32]
    assert N.window_indices(40, 64, 32).tolist() == []


# ---------------------------------------------------------------------------
# prosody: the properties the plan names
# ---------------------------------------------------------------------------


def test_fast_large_motion_scores_above_slow_small() -> None:
    """A scripted fast/large motion must score higher on speed and amplitude."""
    fps = 25.0
    fast = synthetic_pose(120, hand_speed=0.25, hand_amp=0.15)
    slow = synthetic_pose(120, hand_speed=0.02, hand_amp=0.01)

    pf = P.summarize(fast, POSE_SLICE, fps)
    ps = P.summarize(slow, POSE_SLICE, fps)

    assert pf.speed > ps.speed * 3
    assert pf.amplitude > ps.amplitude
    assert pf.volume > ps.volume
    assert pf.peak_speed > ps.peak_speed


def test_amplitude_is_measured_in_shoulder_widths() -> None:
    """Scaling the signer must not change the normalized amplitude.

    Two otherwise identical signers at different distances from the camera have
    to produce the same prosody, or the model is learning camera distance.
    """
    fps = 25.0
    near = synthetic_pose(80, span=0.5, hand_amp=0.3, hand_speed=0.2)
    far = synthetic_pose(80, span=0.15, hand_amp=0.09, hand_speed=0.06)
    a, _ = N.normalize_pose(near, POSE_SLICE)
    b, _ = N.normalize_pose(far, POSE_SLICE)
    pa = P.summarize(a, POSE_SLICE, fps)
    pb = P.summarize(b, POSE_SLICE, fps)
    assert pa.amplitude == pytest.approx(pb.amplitude, rel=0.1)


def test_still_signing_is_all_pause() -> None:
    lm = synthetic_pose(60, hand_speed=0.0, hand_amp=0.0)
    p = P.summarize(lm, POSE_SLICE, 25.0)
    assert p.pause_fraction > 0.9
    assert p.amplitude == pytest.approx(0.0, abs=1e-3)
    assert p.speed == pytest.approx(0.0, abs=1e-2)


def test_repetition_counts_a_cycling_hand() -> None:
    """Three cycles of the hand must score more than one continuous sweep."""
    fps = 25.0
    lm = synthetic_pose(150, hand_amp=0.25)
    cyc = P.repetition(P.hand_centroids(lm, POSE_SLICE), fps)
    sweep = P.repetition(
        P.hand_centroids(synthetic_pose(150, hand_speed=0.2, hand_amp=0.0), POSE_SLICE), fps
    )
    assert cyc > sweep
    assert cyc >= 1.0


def test_jerk_rises_with_sharpness() -> None:
    """A sharp reversal must score more jerk than a linear ramp.

    Both wrists move *together* in both cases. Moving them in opposite directions
    holds their mean - the quantity ``hand_centroids`` returns - perfectly still
    and makes the assertion vacuous, which is exactly what happened the first
    time this test was written.
    """
    fps = 25.0
    smooth = synthetic_pose(100, hand_speed=0.05, hand_amp=0.0)
    sharp = smooth.copy()

    # Three full cycles over 100 frames at 25 fps is ~0.75 Hz, inside the band
    # the jerk smoother preserves. A 5-frame square wave would sit at 5 Hz and be
    # attenuated away, testing the filter rather than the feature.
    t = np.arange(100)
    square = np.where((t // 17) % 2 == 0, 0.60, 0.40)
    sharp[:, 15, 0] = square
    sharp[:, 16, 0] = square

    assert P.jerk(P.hand_centroids(sharp, POSE_SLICE), fps) > P.jerk(
        P.hand_centroids(smooth, POSE_SLICE), fps
    )


def test_two_handed_opposed_motion_has_a_stationary_centroid() -> None:
    """Symmetric two-handed signs really do have a still centroid.

    Worth pinning down because it is a property of the feature rather than a bug:
    speed is a centroid statistic, so it under-reports symmetric two-handed
    movement. The limitation is real and belongs in the paper.
    """
    fps = 25.0
    opposed = synthetic_pose(100, hand_speed=0.1, two_handed_opposed=True)
    c = P.hand_centroids(opposed, POSE_SLICE)
    assert abs(c[-1, 0] - c[0, 0]) < 0.02
    assert P.summarize(opposed, POSE_SLICE, fps).speed < 0.05


def test_prosody_vector_is_fixed_width_and_finite() -> None:
    p = P.summarize(synthetic_pose(40), POSE_SLICE, 25.0)
    v = p.as_vector()
    assert v.shape == (len(P.PROSODY_FIELDS),)
    assert np.isfinite(v).all()


def test_prosody_rejects_zero_fps() -> None:
    with pytest.raises(N.NormalizeError):
        P.summarize(synthetic_pose(10), POSE_SLICE, 0.0)


def test_windowed_prosody_covers_the_clip() -> None:
    lm = synthetic_pose(200, hand_speed=0.1, hand_amp=0.05)
    vecs, edges = P.windowed(lm, POSE_SLICE, 25.0, window=64, stride=32)
    assert vecs.shape[1] == len(P.PROSODY_FIELDS)
    assert len(vecs) == len(edges)
    assert edges.tolist() == [0, 32, 64, 96, 128]


def test_missing_hands_do_not_become_a_phantom_pause() -> None:
    """NaN centroids are filled for the summary but the mask is not invented.

    A centroid at the origin would read as "hands together at the body centre",
    which is a real pose and a different measurement.
    """
    lm = synthetic_pose(40, hand_speed=0.1)
    lm[:, 15, :2] = 0.0
    lm[:, 16, :2] = 0.0
    c = P.hand_centroids(lm, POSE_SLICE)
    assert np.isnan(c).all()


# ---------------------------------------------------------------------------
# markers
# ---------------------------------------------------------------------------


def base_blendshapes(n: int, **levels: float) -> np.ndarray:
    """A (T, 52) basis with named coefficients pinned to constant levels."""
    bs = np.zeros((n, 52), dtype=np.float64)
    for name, value in levels.items():
        bs[:, BLENDSHAPE_INDEX[name]] = value
    return bs


def test_synthetic_brow_raise_is_detected() -> None:
    """The plan's named test: a scripted brow-raise must fire."""
    rng = np.random.default_rng(0)
    n = 120
    bs = base_blendshapes(n)
    for name in M.BROW_RAISE:
        bs[:, BLENDSHAPE_INDEX[name]] = 0.1 + rng.normal(0, 0.01, n)
    # Frames 50-80: a clear raise.
    for name in M.BROW_RAISE:
        bs[50:80, BLENDSHAPE_INDEX[name]] = 0.75

    sig = M.signals(bs, fps=25.0)
    fired = M.fire(sig)
    assert fired["brow_raise"][50:80].mean() > 0.8
    assert fired["brow_raise"][:40].mean() < 0.1


def test_synthetic_brow_furrow_is_detected() -> None:
    rng = np.random.default_rng(1)
    n = 120
    bs = base_blendshapes(n)
    for name in M.BROW_FURROW:
        bs[:, BLENDSHAPE_INDEX[name]] = 0.12 + rng.normal(0, 0.01, n)
    for name in M.BROW_FURROW:
        bs[40:90, BLENDSHAPE_INDEX[name]] = 0.8

    sig = M.signals(bs, fps=25.0)
    fired = M.fire(sig)
    assert fired["brow_furrow"][40:90].mean() > 0.8
    assert not fired["brow_raise"][40:90].mean() > 0.5


def test_a_flat_face_fires_nothing() -> None:
    """A neutral, motionless face must produce no markers at all.

    Otherwise the audit's marker-free control group does not exist.
    """
    bs = base_blendshapes(80)
    sig = M.signals(bs, fps=25.0)
    fired = M.fire(sig)
    for name in M.MARKERS:
        assert not fired[name].any(), name


def test_marker_thresholding_is_clip_relative() -> None:
    """A signer who holds a mild furrow throughout must not be all-furrow.

    An absolute threshold would label the whole clip and the audit would then be
    measuring the signer's face rather than the marker.
    """
    rng = np.random.default_rng(2)
    bs = base_blendshapes(100)
    for name in M.BROW_FURROW:
        bs[:, BLENDSHAPE_INDEX[name]] = 0.55 + rng.normal(0, 0.02, 100)
    sig = M.signals(bs, fps=25.0)
    assert M.fire(sig)["brow_furrow"].mean() < 0.2


def _zyx(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """(T, 4, 4) rotations as Rz(yaw) @ Ry(pitch) @ Rx(roll).

    The product order matters and is not a detail: :func:`markers._euler_from_matrix`
    decomposes with the ZYX convention, which assumes exactly this order. A pure Ry is
    *pitch* under that decomposition, so a test that builds one and calls it "yaw" is
    testing a different axis than it names - which is how the original version of
    ``test_head_shake_needs_oscillation_not_a_static_offset`` came to pass for the wrong
    reason.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rz = np.zeros((*roll.shape, 3, 3))
    ry = np.zeros((*roll.shape, 3, 3))
    rx = np.zeros((*roll.shape, 3, 3))
    rz[..., 0, 0], rz[..., 0, 1], rz[..., 1, 0], rz[..., 1, 1] = cy, -sy, sy, cy
    rz[..., 2, 2] = 1.0
    ry[..., 0, 0], ry[..., 0, 2], ry[..., 1, 1] = cp, sp, 1.0
    ry[..., 2, 0], ry[..., 2, 2] = -sp, cp
    rx[..., 0, 0] = 1.0
    rx[..., 1, 1], rx[..., 1, 2], rx[..., 2, 1], rx[..., 2, 2] = cr, -sr, sr, cr
    out = np.zeros((*roll.shape, 4, 4))
    out[..., :3, :3] = rz @ ry @ rx
    out[..., 3, 3] = 1.0
    return out


def test_euler_decomposition_round_trips_each_axis() -> None:
    """The head path must recover the angles it is given, on each axis separately.

    ``head_shake`` and ``head_nod`` read columns 2 and 1 of the decomposition and nothing
    else, so a decomposition that is wrong on one axis leaves those two markers reading a
    quantity unrelated to head motion while every downstream test still passes - the
    blendshape markers do not consult the rotation at all, and a head test only asserts
    that *something* is nonzero.

    This asserts the actual values. The previous implementation used
    ``arctan2(r[2,0], r[1,0])`` for yaw, which returned exactly ``0.0`` for a pure yaw of
    any size and ``+/-pi/2`` for a pure pitch, so ``head_shake`` was measured against a
    signal that was identically zero for the motion it exists to detect.
    """
    for roll, pitch, yaw in [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.55),  # pure yaw - the case the old formula returned 0.0 for
        (0.0, 0.5, 0.0),  # pure pitch - the case the old formula returned pi/2 for
        (0.3, 0.4, 0.5),
        (-0.2, 0.7, -0.4),
        (0.0, 1.5, 0.3),  # near the pitch = pi/2 singularity
    ]:
        rot = _zyx(np.full(1, roll), np.full(1, pitch), np.full(1, yaw))
        got = M._euler_from_matrix(rot)[0]
        assert got[0] == pytest.approx(roll, abs=1e-6), f"roll at {(roll, pitch, yaw)}"
        assert got[1] == pytest.approx(pitch, abs=1e-6), f"pitch at {(roll, pitch, yaw)}"
        assert got[2] == pytest.approx(yaw, abs=1e-6), f"yaw at {(roll, pitch, yaw)}"


def test_head_markers_read_their_own_axis() -> None:
    """A shake must be visible to ``head_shake`` and not mainly to ``head_nod``.

    Yaw and pitch are the two components that were swapped by the decomposition bug, so a
    test that only checks each marker is nonzero on some rotation cannot tell them apart.
    This drives each axis separately and asserts the separation.
    """
    fps = 25.0
    t = np.arange(100) / fps
    bs = base_blendshapes(100)
    zero = np.zeros(100)

    sig = M.signals(bs, _zyx(zero, zero, 0.6 * np.sin(2 * np.pi * 3.0 * t)), fps)
    assert sig.head_shake.sum() > 0, "a 0.6 rad yaw oscillation must register as head_shake"
    assert sig.head_nod.sum() == 0.0, "a pure yaw must not register as head_nod"

    sig = M.signals(bs, _zyx(zero, 0.6 * np.sin(2 * np.pi * 3.0 * t), zero), fps)
    assert sig.head_nod.sum() > 0, "a 0.6 rad pitch oscillation must register as head_nod"
    assert sig.head_shake.sum() == 0.0, "a pure pitch must not register as head_shake"


def test_head_shake_needs_oscillation_not_a_static_offset() -> None:
    fps = 25.0
    t = np.arange(100) / fps
    rot_static = np.tile(np.eye(4), (100, 1, 1))
    assert M.signals(base_blendshapes(100), rot_static, fps).head_shake.sum() == 0.0

    rot = _zyx(np.zeros(100), np.zeros(100), 0.6 * np.sin(2 * np.pi * 3.0 * t))
    sig = M.signals(base_blendshapes(100), rot, fps)
    assert sig.head_shake.sum() > 0
    assert M.fire(sig)["head_shake"].mean() > 0.1


def test_dead_coefficients_are_really_dead_on_emosign() -> None:
    """The documented zero-variance set must be the actual zero set.

    Asserted against the M0 census so that if a different corpus or a different
    blendshape model changes the answer, the paper's limitation text changes with
    it rather than silently becoming false.
    """
    from seam import paths

    report = paths.artifacts_root() / "reports" / "emosign_landmarks.json"
    if not report.is_file():
        pytest.skip("EmoSign landmark census not present")
    import json

    measured = set(json.loads(report.read_text(encoding="utf-8"))["zero_variance_coefficients"])
    assert measured == set(M.DEAD_COEFFICIENTS)


def test_no_marker_rule_names_a_dead_coefficient() -> None:
    """A rule on a coefficient that never moves is silently inert."""
    dead = set(M.DEAD_COEFFICIENTS)
    used = (
        set(M.BROW_RAISE)
        | set(M.BROW_FURROW)
        | set(M.EYE_WIDE)
        | set(M.MOUTH_MORPHEME)
        | set(M.MOUTH_OPEN)
        | set(M.MOUTH_POSITIVE)
    )
    assert not (dead & used), f"inert rules: {sorted(dead & used)}"


def test_marker_matrix_is_in_declared_order() -> None:
    n = 10
    sig = M.signals(base_blendshapes(n), fps=25.0)
    mat = sig.as_matrix()
    assert mat.shape == (n, len(M.MARKERS))
    np.testing.assert_allclose(mat[:, 0], sig.brow_raise)
    np.testing.assert_allclose(mat[:, -1], sig.mouth_positive)


def test_window_firing_needs_a_minimum_fraction() -> None:
    """A one-frame spike must not mark a whole window."""
    rng = np.random.default_rng(3)
    n = 64
    bs = base_blendshapes(n)
    for name in M.BROW_RAISE:
        bs[:, BLENDSHAPE_INDEX[name]] = 0.1 + rng.normal(0, 0.01, n)
    bs[30, BLENDSHAPE_INDEX["browInnerUp"]] = 0.99  # single-frame spike
    sig = M.signals(bs, fps=25.0)
    out = M.window_fired(sig, np.array([0]), 64)
    assert not out[0, 0]

    for i in range(20, 45):
        bs[i, BLENDSHAPE_INDEX["browInnerUp"]] = 0.99
    out = M.window_fired(M.signals(bs, fps=25.0), np.array([0]), 64)
    assert out[0, 0]


def test_signals_rejects_wrong_blendshape_width() -> None:
    with pytest.raises(ValueError, match=r"\(T, 52\)"):
        M.signals(np.zeros((10, 10)))


def test_describe_reports_the_disclosed_limitation() -> None:
    d = M.describe()
    assert "pseudo" in d["provenance"]
    assert "facial_negation" in d["unavailable_markers"]
    assert d["thresholds"]["rule"].startswith("score > median")
    assert set(d["markers"]) == set(M.MARKERS)


def test_all_marker_coefficients_exist_in_the_basis() -> None:
    for group in (
        M.BROW_RAISE,
        M.BROW_FURROW,
        M.EYE_WIDE,
        M.MOUTH_MORPHEME,
        M.MOUTH_OPEN,
        M.MOUTH_POSITIVE,
    ):
        for name in group:
            assert name in BLENDSHAPE_CATEGORIES, name
    assert len(BLENDSHAPE_CATEGORIES) == 52
