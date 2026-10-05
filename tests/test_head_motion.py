"""Head angles named for what the head does, and the detectors that read them.

`seam.features.markers` once read a head tilt for `head_shake` and a head turn for
`head_nod`. That went unnoticed because every test of it used angles built with the same
convention it was written in. The tests here build a rotation from the physical motion - the head
turning about the vertical axis, nodding about the ear-to-ear axis, tilting about the
nose - and ask which reading moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.features import head_motion as HM
from seam.features import markers as VM

FPS = 24.0


def _about(axis: str, deg: np.ndarray) -> np.ndarray:
    """(T, 4, 4) rotations of a canonical face (x right, y up, z out of the face)."""
    a = np.radians(np.asarray(deg, dtype=np.float64))
    c, s = np.cos(a), np.sin(a)
    rot = np.tile(np.eye(4), (len(a), 1, 1))
    if axis == "vertical":  # a turn: the shake axis
        rot[:, 0, 0], rot[:, 0, 2], rot[:, 2, 0], rot[:, 2, 2] = c, s, -s, c
    elif axis == "ear_to_ear":  # a nod
        rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = c, -s, s, c
    elif axis == "nose":  # a tilt
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1] = c, -s, s, c
    else:
        raise ValueError(axis)
    return rot


def _wave(amplitude: float, hz: float = 2.5, seconds: float = 3.0) -> np.ndarray:
    t = np.arange(int(seconds * FPS)) / FPS
    return amplitude * np.sin(2 * np.pi * hz * t)


@pytest.mark.parametrize(
    ("axis", "column"),
    [("vertical", HM.TURN), ("ear_to_ear", HM.NOD), ("nose", HM.TILT)],
)
def test_each_physical_motion_moves_its_own_column_and_no_other(axis: str, column: int) -> None:
    deg = np.linspace(-25, 25, 21)
    angles = HM.head_angles(_about(axis, deg))
    np.testing.assert_allclose(np.abs(angles[:, column]), np.abs(deg), atol=1e-6)
    others = [k for k in range(3) if k != column]
    assert np.abs(angles[:, others]).max() < 1e-6


def test_an_empty_clip_has_no_angles() -> None:
    assert HM.head_angles(np.zeros((0, 4, 4))).shape == (0, 3)


def test_the_offline_markers_read_the_motion_they_are_named_for() -> None:
    """A turn is a shake, a nod is a nod, and a tilt is neither.

    For a day this test was a strict expected failure: `markers.signals` read a head tilt
    for head_shake and a head turn for head_nod, and it was left alone because nine cited
    artifacts had been computed with it. It was corrected and all of them were re-run
    together (EXPERIMENT_LOG, 2026-10-05, "one hand and the wrong axis, corrected").
    """
    turn = VM.signals(np.zeros((72, 52)), _about("vertical", _wave(30.0)), fps=FPS)
    nod = VM.signals(np.zeros((72, 52)), _about("ear_to_ear", _wave(30.0)), fps=FPS)
    tilt = VM.signals(np.zeros((72, 52)), _about("nose", _wave(30.0)), fps=FPS)
    assert np.asarray(turn.head_shake).max() > 0 and np.asarray(turn.head_nod).max() == 0
    assert np.asarray(nod.head_nod).max() > 0 and np.asarray(nod.head_shake).max() == 0
    assert np.asarray(tilt.head_shake).max() == 0 and np.asarray(tilt.head_nod).max() == 0


def test_one_turn_of_the_head_is_not_a_shake() -> None:
    single = np.concatenate([np.zeros(12), np.linspace(0, 30, 12), np.full(48, 30.0)])
    assert HM.reversal_score(single, FPS).max() == 0.0


def test_reversals_count_only_swings_of_real_amplitude() -> None:
    assert HM.reversal_score(_wave(1.0), FPS, min_deg=5.0).max() == 0.0
    big = HM.reversal_score(_wave(12.0), FPS, min_deg=5.0)
    assert big.max() >= 2 / HM.BROWSER_FULL_SCALE
    assert big[:3].max() == 0.0, "the detector is causal: nothing can fire before it has moved"


def test_reversals_are_forgotten_after_the_window() -> None:
    x = np.concatenate([_wave(12.0, seconds=1.5), np.zeros(int(3 * FPS))])
    score = HM.reversal_score(x, FPS, window_s=1.3)
    assert score[: int(1.5 * FPS)].max() > 0
    assert score[-1] == 0.0


def test_band_energy_is_the_rms_of_the_oscillation_and_ignores_posture() -> None:
    held = np.full(72, 20.0)
    drift = np.linspace(0, 20, 240)
    assert HM.band_energy(held, FPS).max() < 1e-9
    assert HM.band_energy(drift, FPS)[30:-30].max() < 0.5
    shake = HM.band_energy(20.0 + _wave(6.0, seconds=4.0), FPS)
    # A 6 degree sine has an RMS of 4.2; the two averages attenuate a 2.5 Hz wave a little.
    assert 2.5 < float(np.median(shake[24:-24])) < 4.5


def test_band_energy_has_no_amplitude_threshold() -> None:
    small = HM.band_energy(_wave(1.5, seconds=4.0), FPS)
    large = HM.band_energy(_wave(6.0, seconds=4.0), FPS)
    ratio = np.median(large[24:-24]) / np.median(small[24:-24])
    assert ratio == pytest.approx(4.0, rel=0.05)


def test_the_live_score_uses_the_past_only() -> None:
    x = _wave(8.0, seconds=4.0)
    full = HM.live_band_rms(x, FPS)
    cut = HM.live_band_rms(x[:40], FPS)
    np.testing.assert_allclose(full[:40], cut, atol=1e-12)


def test_the_live_score_does_not_depend_on_the_frame_rate() -> None:
    """The page runs at whatever rate the machine manages; the reading must not change."""
    out = {}
    for fps in (15.0, 30.0, 60.0):
        t = np.arange(int(4 * fps)) / fps
        x = 8.0 * np.sin(2 * np.pi * 2.0 * t)
        out[fps] = float(np.median(HM.live_band_rms(x, fps)[int(2 * fps) :]))
    assert out[15.0] == pytest.approx(out[60.0], rel=0.12)
    assert out[30.0] == pytest.approx(out[60.0], rel=0.06)


def test_the_live_score_rises_with_a_shake_and_falls_after_it() -> None:
    x = np.concatenate([np.zeros(48), _wave(8.0, seconds=2.0), np.zeros(72)])
    score = HM.live_band_rms(x, FPS)
    assert score[:48].max() == 0.0
    assert score[60:96].max() > 3.0
    assert score[-1] < 0.3


def test_contrast_takes_a_share_of_the_other_axis_off() -> None:
    turn, nod = _wave(8.0, seconds=4.0), _wave(8.0, seconds=4.0)
    alone = HM.live_oscillation(turn, np.zeros_like(turn), FPS, contrast=1.0)
    wobble = HM.live_oscillation(turn, nod, FPS, contrast=1.0)
    assert alone[48:].min() > 2.0
    assert wobble.max() < 1e-9, "equal motion on both axes is neither a shake nor a nod"
    assert (HM.live_oscillation(turn, nod, FPS, contrast=0.0) == HM.live_band_rms(turn, FPS)).all()


def _face(brow_lift: float = 0.0, scale: float = 1.0, tilt_deg: float = 0.0) -> np.ndarray:
    """One frame of the landmarks `brow_geometry` reads, on an upright or tilted face."""
    pts = np.zeros((478, 2))
    eyes = {33: (-0.30, 0.0), 263: (0.30, 0.0), 133: (-0.10, 0.0), 362: (0.10, 0.0)}
    eyes |= {159: (-0.20, -0.02), 386: (0.20, -0.02)}
    brows = {107: (-0.07, -0.12), 336: (0.07, -0.12), 105: (-0.20, -0.14), 334: (0.20, -0.14)}
    brows |= {70: (-0.32, -0.10), 300: (0.32, -0.10)}
    for k, (x, y) in eyes.items():
        pts[k] = (x, y)
    for k, (x, y) in brows.items():
        pts[k] = (x, y - brow_lift)  # image y grows downwards, so up is negative
    pts[10], pts[152] = (0.0, -0.5), (0.0, 0.6)
    a = np.radians(tilt_deg)
    spin = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (pts @ spin.T * scale + 0.5)[None, :, :]


def test_a_raised_brow_measures_higher_and_distance_from_the_camera_cancels() -> None:
    rest = HM.brow_geometry(_face())
    raised = HM.brow_geometry(_face(brow_lift=0.05))
    near = HM.brow_geometry(_face(scale=2.0))
    for key in ("inner_height", "mid_height", "outer_height"):
        assert raised[key][0] > rest[key][0] > 0
        assert near[key][0] == pytest.approx(rest[key][0], rel=1e-6)
    assert near["inner_gap"][0] == pytest.approx(rest["inner_gap"][0], rel=1e-6)


def test_a_tilted_head_does_not_read_as_a_moved_brow() -> None:
    rest = HM.brow_geometry(_face())
    tilted = HM.brow_geometry(_face(tilt_deg=20.0))
    for key, value in rest.items():
        assert tilted[key][0] == pytest.approx(value[0], abs=1e-6), key
