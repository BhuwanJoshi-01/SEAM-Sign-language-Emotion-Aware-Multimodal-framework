"""The finger solver must put each bone where the tracked hand's bone is.

Everything here runs on a made-up hand, so it needs neither the licensed body model nor
MediaPipe: a "model" hand is laid out flat, a pose is applied to it by forward kinematics,
its joints are read off as the 21 points a tracker would report, and the solver has to get
the bones back - whatever way the hand is facing, however big it looks, and for a left hand
as well as a right.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.avatar import hand_ik as H


def _rest_hand(side: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Rest joints (55, 3) and fingertips of a flat hand, shaped like SMPL-X's layout."""
    sign = 1.0 if side == "left" else -1.0  # SMPL-X arms point along +x (left) and -x (right)
    joints = np.zeros((55, 3))
    wrist = np.array([sign * 0.67, 0.03, -0.06])
    joints[H.WRIST_JOINT[side]] = wrist
    spread = {"index": 0.02, "middle": 0.0, "ring": -0.02, "pinky": -0.04, "thumb": 0.045}
    reach = {"index": 0.10, "middle": 0.11, "ring": 0.10, "pinky": 0.085, "thumb": 0.04}
    lengths = {"index": (0.032, 0.022, 0.024), "middle": (0.034, 0.024, 0.026)}
    lengths |= {"ring": (0.031, 0.023, 0.024), "pinky": (0.024, 0.018, 0.02)}
    lengths |= {"thumb": (0.03, 0.026, 0.03)}
    tips = {}
    for i, name in enumerate(H.FINGERS):
        along = np.array([sign, 0.0, 0.0])
        if name == "thumb":
            along = np.array([sign * 0.7, -0.1, 0.7])
            along /= np.linalg.norm(along)
        pos = wrist + np.array([sign * reach[name], 0.0, spread[name]])
        for k in range(3):
            joints[H.FIRST_FINGER_JOINT[side] + 3 * i + k] = pos
            pos = pos + along * lengths[name][k]
        tips[name] = pos
    return joints, tips


def _posed_points(
    joints: np.ndarray, tips: dict[str, np.ndarray], side: str, pose: np.ndarray
) -> np.ndarray:
    """The 21 points a tracker would see for this hand with these finger rotations."""
    first = H.FIRST_FINGER_JOINT[side]
    pts = np.zeros((21, 3))
    pts[H.MP_WRIST] = joints[H.WRIST_JOINT[side]]
    for i, name in enumerate(H.FINGERS):
        rest = [joints[first + 3 * i + k] for k in range(3)] + [tips[name]]
        carried = np.eye(3)
        pos = rest[0]
        pts[H.MP_CHAIN[name][0]] = pos
        for k in range(3):
            carried = carried @ H.aa_to_matrix(pose[3 * i + k])
            pos = pos + carried @ (rest[k + 1] - rest[k])
            pts[H.MP_CHAIN[name][k + 1]] = pos
    return pts


def _bones(points: np.ndarray) -> np.ndarray:
    out = []
    for name in H.FINGERS:
        c = H.MP_CHAIN[name]
        out += [points[c[k + 1]] - points[c[k]] for k in range(3)]
    b = np.array(out)
    return b / np.linalg.norm(b, axis=1, keepdims=True)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _curl(side: str, rng: np.random.Generator) -> np.ndarray:
    """A plausible handshape: each finger bent about an axis across the palm."""
    pose = np.zeros((15, 3))
    for i, name in enumerate(H.FINGERS):
        axis = np.array([0.0, 0.0, 1.0]) if name != "thumb" else np.array([0.0, 1.0, 0.0])
        for k in range(3):
            pose[3 * i + k] = axis * rng.uniform(0.1, 1.2) * (1.0 if side == "left" else -1.0)
    return pose


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_flat_hand_solves_to_no_rotation(side: str) -> None:
    joints, tips = _rest_hand(side)
    rest = H.hand_rest(joints, tips, side)
    solved = H.solve_fingers(_posed_points(joints, tips, side, np.zeros((15, 3))), rest)
    assert solved is not None
    np.testing.assert_allclose(solved, 0.0, atol=1e-7)


@pytest.mark.parametrize("side", ["left", "right"])
def test_solved_fingers_reproduce_every_bone_of_the_tracked_hand(side: str) -> None:
    """Whichever way the hand faces and however large it looks."""
    rng = np.random.default_rng(3)
    joints, tips = _rest_hand(side)
    rest = H.hand_rest(joints, tips, side)
    for _ in range(12):
        pose = _curl(side, rng)
        seen = _posed_points(joints, tips, side, pose)
        turn, scale, shift = _random_rotation(rng), rng.uniform(0.2, 40.0), rng.normal(size=3)
        tracked = (seen @ turn.T) * scale + shift
        solved = H.solve_fingers(tracked, rest)
        assert solved is not None
        rebuilt = _posed_points(joints, tips, side, solved)
        np.testing.assert_allclose(_bones(rebuilt), _bones(seen), atol=1e-6)


def test_a_fist_bends_far_more_than_an_open_hand() -> None:
    """The number the demo reports, on the two handshapes it has to tell apart."""
    joints, tips = _rest_hand("right")
    rest = H.hand_rest(joints, tips, "right")
    fist = np.zeros((15, 3))
    for i in range(4):  # four fingers curled about the axis across the palm
        fist[3 * i : 3 * i + 3] = np.array([0.0, 0.0, -1.3])
    solved = H.solve_fingers(_posed_points(joints, tips, "right", fist), rest)
    assert solved is not None
    assert H.mean_bend_deg(solved) > 50.0
    flat = H.solve_fingers(_posed_points(joints, tips, "right", np.zeros((15, 3))), rest)
    assert flat is not None and H.mean_bend_deg(flat) < 1e-4


def test_unusable_points_give_no_solution_instead_of_a_guess() -> None:
    joints, tips = _rest_hand("left")
    rest = H.hand_rest(joints, tips, "left")
    assert H.solve_fingers(np.zeros((21, 3)), rest) is None, "every point at one place"
    assert H.solve_fingers(np.full((21, 3), np.nan), rest) is None
    assert H.solve_fingers(np.zeros((20, 3)), rest) is None
    line = np.outer(np.arange(21), [1.0, 0.0, 0.0])  # a palm with no width
    assert H.solve_fingers(line, rest) is None


def test_a_noisy_joint_is_clamped_not_believed() -> None:
    joints, tips = _rest_hand("left")
    rest = H.hand_rest(joints, tips, "left")
    pts = _posed_points(joints, tips, "left", np.zeros((15, 3)))
    c = H.MP_CHAIN["index"]
    pts[c[1]] = pts[c[0]] - (pts[c[1]] - pts[c[0]])  # the first index bone folded back on itself
    solved = H.solve_fingers(pts, rest)
    assert solved is not None
    assert np.linalg.norm(solved[0]) == pytest.approx(H.MAX_JOINT_ANGLE)


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_wrist_is_recovered_in_world_space(side: str) -> None:
    """Turn the whole hand; the solved wrist rotation is that turn."""
    rng = np.random.default_rng(5)
    joints, tips = _rest_hand(side)
    rest = H.hand_rest(joints, tips, side)
    seen = _posed_points(joints, tips, side, _curl(side, rng))
    to_world = np.diag([1.0, -1.0, -1.0])  # camera (y down, z away) to a y-up world
    for _ in range(6):
        wrist = _random_rotation(rng)
        in_world = seen @ wrist.T
        in_camera = in_world @ to_world  # to_world is its own inverse
        solved = H.solve_wrist(in_camera, rest, to_world)
        assert solved is not None
        np.testing.assert_allclose(solved, wrist, atol=1e-6)


def test_an_implausible_wrist_falls_back_to_the_regressor() -> None:
    elbow = np.eye(3)
    fallback = np.array([0.1, 0.2, 0.3])
    half_turn = H.aa_to_matrix(np.array([0.0, 3.0, 0.0]))
    np.testing.assert_allclose(H.wrist_local(half_turn, elbow, fallback), fallback)
    small = H.aa_to_matrix(np.array([0.0, 0.6, 0.0]))
    np.testing.assert_allclose(H.wrist_local(small, elbow, fallback), [0.0, 0.6, 0.0], atol=1e-9)


def test_global_rotations_compose_down_the_chain() -> None:
    parents = np.array([-1, 0, 1])
    pose = np.array([[0.0, 0.0, 0.5], [0.0, 0.0, 0.25], [0.0, 0.0, -0.75]])
    world = H.global_rotations(pose, parents)
    np.testing.assert_allclose(world[1], H.aa_to_matrix([0.0, 0.0, 0.75]), atol=1e-12)
    np.testing.assert_allclose(world[2], np.eye(3), atol=1e-12)


def test_short_gaps_are_bridged_on_the_sphere_and_long_ones_are_left() -> None:
    a = np.zeros((15, 3))
    b = np.tile([0.0, 0.0, 0.9], (15, 1))
    track: list[np.ndarray | None] = [a, None, None, b, None, None, None, None, None, None, a, None]
    filled, bridged = H.fill_gaps(track, max_gap=3)
    assert bridged == 2
    np.testing.assert_allclose(filled[1][0], [0.0, 0.0, 0.3], atol=1e-9)
    np.testing.assert_allclose(filled[2][0], [0.0, 0.0, 0.6], atol=1e-9)
    assert all(filled[k] is None for k in range(4, 10)), "a long gap is not invented"
    assert filled[11] is None, "nor is the end of the clip"


def test_rotation_between_handles_opposite_directions() -> None:
    a = np.array([1.0, 0.0, 0.0])
    R = H.rotation_between(a, -a)
    np.testing.assert_allclose(R @ a, -a, atol=1e-9)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
    np.testing.assert_allclose(H.aa_to_matrix(H.matrix_to_aa(R)) @ a, -a, atol=1e-6)


def test_the_real_models_fingertips_are_where_the_ids_say() -> None:
    """With the licensed model present, the published fingertip vertices are checked."""
    from pathlib import Path

    path = Path(
        "/mnt/Volume2/SignLanguagge/NSL Data/Sapien_Pipeline/models/smplx/SMPLX_NEUTRAL.npz"
    )
    if not path.is_file():
        pytest.skip("the licence-gated SMPL-X model is not on this machine")
    model = dict(np.load(path, allow_pickle=True))
    for side in ("left", "right"):
        rest = H.hand_rest_from_model(model, None, side)
        np.testing.assert_allclose(np.linalg.norm(rest.bones, axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(rest.frame @ rest.frame.T, np.eye(3), atol=1e-9)
        assert np.linalg.det(rest.frame) == pytest.approx(1.0)
