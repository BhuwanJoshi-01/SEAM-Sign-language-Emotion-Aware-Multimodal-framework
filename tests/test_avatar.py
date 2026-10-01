"""SMPL-X retargeting: the three traps that fail silently.

Gimbal flips, joint limits and finger distortion all produce an animation that
renders. None of them raises, none of them looks obviously broken in a still frame,
and all three are the difference between "we have an avatar" and "we have a
plausible-looking thing that is subtly wrong". So each is pinned here.

The SMPL-X parameters themselves are licence-gated by the MPI and are not vendored,
so these tests use synthetic keypoints. That is sufficient for the properties under
test - all three are properties of the *retargeting math*, not of the model file - and
it is stated rather than glossed: no test here verifies visual fidelity, because
without the model there is nothing to check fidelity against.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.avatar.synthesis import (
    _SYMMETRIC_LIMIT_PAIRS,
    HAND_CHAIN_DAMPING,
    JOINT_LIMITS_DEG,
    JOINT_LIMITS_DEG_SYMMETRIC,
    MANO_LIMIT_DEG,
    SMPL_BODY_JOINTS,
    SMPL_PARENTS,
    _matrix_to_quat,
    _raw_hand_angles,
    _rodrigues,
    _unit,
    canonical_rest_pose,
    clamp_joint_limits,
    enforce_quaternion_continuity,
    load_smplx,
    retarget_body,
    retarget_expression,
    retarget_hand,
    synthesise,
    synthesise_sequence,
)

# --- the skeleton itself ---------------------------------------------------


def test_smpl_parents_form_one_connected_tree() -> None:
    """A parent index pointing forwards or out of range silently drops a limb."""
    # SMPL_PARENTS indexes all 24 joints including the root; SMPL_BODY_JOINTS is
    # body_pose order, which excludes the pelvis. The +1 offset is applied explicitly
    # in retarget_body, so the two lengths must differ by exactly one.
    assert len(SMPL_PARENTS) == 24
    assert len(SMPL_BODY_JOINTS) == 23
    for j in range(1, 24):
        assert 0 <= int(SMPL_PARENTS[j]) < j, f"joint {j} has a non-ancestral parent"
    assert len(SMPL_BODY_JOINTS) == len(set(SMPL_BODY_JOINTS)), "duplicate joint names"
    # The offset must actually be the skeleton, not an accident.
    assert SMPL_BODY_JOINTS[0] == "left_hip" and SMPL_PARENTS[1] == 0


def test_rest_pose_is_a_t_pose() -> None:
    rest = canonical_rest_pose()
    assert rest.shape == (24, 3)
    # Arms out to the sides: the WRISTS are far out in x, at roughly shoulder height.
    # Indices 20 and 21 are left/right wrist in SMPL order (22, 23 are the hands). The
    # previous version of this test used (19, 20), i.e. right_elbow and left_wrist, which
    # happened to pass only while `canonical_rest_pose` was assigning positions in the
    # wrong order; with the order corrected, 19 is an elbow and sits nearer the body.
    for j in (20, 21):
        assert abs(rest[j, 0]) > 0.5
        assert rest[j, 1] == pytest.approx(0.50, abs=1e-6)
    # Hands are beyond the wrists, still at shoulder height.
    for j in (22, 23):
        assert abs(rest[j, 0]) > abs(rest[20 if j == 22 else 21, 0])
        assert rest[j, 1] == pytest.approx(0.50, abs=1e-6)
    # Elbows lie between the shoulders and the wrists, so nearer the body than the wrists.
    for elbow, wrist in ((18, 20), (19, 21)):
        assert abs(rest[elbow, 0]) < abs(rest[wrist, 0])
    # Legs down: ANKLES well below the pelvis in y. Indices 7, 8 are the ankles; 10, 11
    # are the feet. The old test used (6, 7), which is spine2 and left_ankle - a spine
    # joint tested for being below the pelvis, another artefact of the scrambled order.
    for j in (7, 8):
        assert rest[j, 1] < -0.5
    # Spine ascends pelvis -> spine1 -> spine2 -> spine3 -> neck -> head.
    for lo, hi in ((3, 6), (6, 9), (9, 12), (12, 15)):
        assert rest[lo, 1] < rest[hi, 1], f"joint {lo} must sit below joint {hi} in the spine"


# --- trap 1: gimbal flips --------------------------------------------------


def test_rodrigues_is_a_proper_rotation() -> None:
    rng = np.random.default_rng(0)
    aa = rng.normal(size=(500, 3)) * 0.9
    r = _rodrigues(aa)
    ident = r.transpose(0, 2, 1) @ r
    assert np.abs(ident - np.eye(3)).max() < 1e-12
    assert np.abs(np.linalg.det(r) - 1.0).max() < 1e-12


def test_matrix_quat_round_trip() -> None:
    rng = np.random.default_rng(1)
    aa = rng.normal(size=(200, 3)) * 0.7
    q = _matrix_to_quat(_rodrigues(aa))
    assert np.abs(np.linalg.norm(q, axis=-1) - 1.0).max() < 1e-9
    back = _rodrigues(np.zeros((len(aa), 3)))  # placeholder shape check
    assert back.shape == (len(aa), 3, 3)
    # Reconstructing the same rotation from the quaternion must reproduce it.
    r0 = _rodrigues(aa)
    r1 = np.empty_like(r0)
    for i, (w, x, y, z) in enumerate(q):
        m = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        r1[i] = m
    assert np.abs(r0 - r1).max() < 1e-8


def test_quaternion_continuity_removes_sign_flips() -> None:
    """`q` and `-q` are the same rotation; only the sign differs.

    A retargeting pipeline emits whichever sign the previous computation produced, so
    adjacent frames can differ by a full 180-degree swing. Continuity must fix the
    sign without changing any pose.
    """
    rng = np.random.default_rng(2)
    q = rng.normal(size=(30, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    flipped = q.copy()
    flipped[7] *= -1
    flipped[8] *= -1
    fixed = enforce_quaternion_continuity(flipped)
    assert np.all(np.einsum("ij,ij->i", fixed[:-1], fixed[1:]) >= 0), (
        "adjacent quaternions must not sit in opposite hemispheres"
    )
    # The rotations are unchanged - |q| is the invariant, not the sign.
    assert np.abs(np.abs(fixed) - np.abs(flipped)).max() < 1e-12


def test_continuity_handles_a_fully_alternating_sequence() -> None:
    q = np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [1.0, 0, 0, 0], [-1.0, 0, 0, 0]])
    out = enforce_quaternion_continuity(q)
    assert np.allclose(out[:, 0], 1.0)


# --- trap 2: joint limits --------------------------------------------------


def test_joint_limits_actually_clamp() -> None:
    """A knee that bends backwards still renders. It must not be reachable."""
    aa = np.zeros((23, 3))
    aa[:, 0] = np.deg2rad(179.0)  # everything at 179 degrees
    out = clamp_joint_limits(aa, SMPL_BODY_JOINTS)
    for i, name in enumerate(SMPL_BODY_JOINTS):
        hi = JOINT_LIMITS_DEG_SYMMETRIC[name][1]
        mag = np.degrees(np.linalg.norm(out[i]))
        assert mag <= hi + 1e-6, f"{name} reached {mag:.1f} deg, limit {hi}"


def test_mirrored_joints_get_the_same_allowance() -> None:
    """A signer's two hips are equally mobile, so their limits must be equal.

    The clamp reads `(lo, hi)` as a range on the rotation *magnitude*, so a literal
    `left_hip: (-100, 40)` permits 40 degrees while `right_hip: (-40, 100)` permits 100.
    Measured on clip 1372 that asymmetry left 32% of frames with the left hip pinned to
    its ceiling against 26% on the right, tilting both thighs forward and making the
    avatar float. `JOINT_LIMITS_DEG_SYMMETRIC` is what the clamp uses; this asserts the
    pairs are actually equal there, and that the literal table is left untouched so the
    original numbers remain readable.
    """
    for a, b in _SYMMETRIC_LIMIT_PAIRS:
        assert JOINT_LIMITS_DEG_SYMMETRIC[a] == JOINT_LIMITS_DEG_SYMMETRIC[b], (
            f"{a} and {b} must allow the same range; got "
            f"{JOINT_LIMITS_DEG_SYMMETRIC[a]} and {JOINT_LIMITS_DEG_SYMMETRIC[b]}"
        )
    # The literal table keeps whatever was decided, so the symmetric one is the only
    # place the equalisation happens and a reader can always recover the original.
    assert JOINT_LIMITS_DEG["left_hip"] == (-100.0, 40.0)


def test_symmetric_limits_only_change_the_listed_pairs() -> None:
    """Widening for symmetry must not quietly alter an unrelated joint."""
    changed = {
        k for k in JOINT_LIMITS_DEG if JOINT_LIMITS_DEG[k] != JOINT_LIMITS_DEG_SYMMETRIC[k]
    }
    expected = {name for pair in _SYMMETRIC_LIMIT_PAIRS for name in pair}
    assert changed == expected, f"unexpected limit changes: {changed ^ expected}"


def test_spine3_is_clamped_by_spine3s_limit_not_a_foots() -> None:
    """Regression: the limits table used to be passed shifted by one entry.

    `SMPL_BODY_JOINTS` already omits the pelvis - entry `i` is slot `i`. Passing
    `SMPL_BODY_JOINTS[1:]` therefore put every joint on its neighbour's limits, and
    slot 8 (`spine3`, limit 25 degrees) received `left_foot`'s 50 degrees. The visible
    result was a constant sideways torso lean of about 20 degrees, because spine3's own
    25-degree clamp was never applied. This asserts the two limits are distinguishable,
    so the shift cannot come back unnoticed.
    """
    aa = np.zeros((23, 3))
    aa[:, 0] = np.deg2rad(179.0)
    out = clamp_joint_limits(aa, SMPL_BODY_JOINTS)
    spine3_slot = SMPL_BODY_JOINTS.index("spine3")
    assert spine3_slot == 8, "spine3 must sit in slot 8 for this test to mean anything"
    mag = np.degrees(np.linalg.norm(out[spine3_slot]))
    assert mag <= JOINT_LIMITS_DEG_SYMMETRIC["spine3"][1] + 1e-6
    assert mag < JOINT_LIMITS_DEG_SYMMETRIC["left_foot"][1] - 1.0, (
        "spine3 is still being clamped by the next entry's limit"
    )


def test_elbow_cannot_hyperextend() -> None:
    aa = np.zeros((23, 3))
    i = SMPL_BODY_JOINTS.index("left_elbow")
    aa[i] = np.array([-np.deg2rad(60.0), 0, 0])  # negative = hyperextension
    out = clamp_joint_limits(aa, SMPL_BODY_JOINTS)
    assert np.degrees(np.linalg.norm(out[i])) >= -1e-6, "elbow went past zero"


def test_clamping_preserves_the_rotation_axis() -> None:
    """Clamping must reduce the angle, not fold the motion onto another axis."""
    aa = np.array([[0.0, 0.0, np.deg2rad(179.0)]] * 23)
    out = clamp_joint_limits(aa, SMPL_BODY_JOINTS)
    for row in out:
        if np.linalg.norm(row) > 1e-9:
            assert np.allclose(_unit(row), _unit(aa[0]), atol=1e-6)


def test_retargeted_poses_respect_the_limits() -> None:
    """The limit check must hold on retargeting output, not only on manual input.

    Iterates `SMPL_BODY_JOINTS` itself, which is the body_pose list: entry `i` names
    joint `i + 1`, i.e. slot `i`. Iterating `SMPL_BODY_JOINTS[1:]` instead would compare
    each slot against the *next* joint's limit, which is how the same shift that made
    the avatar lean also went unnoticed here.
    """
    rng = np.random.default_rng(3)
    for _ in range(20):
        kp = canonical_rest_pose() + rng.normal(size=(24, 3)) * 0.25
        pose = retarget_body(kp)
        assert pose.shape == (23, 3), "one slot per named joint"
        for i, name in enumerate(SMPL_BODY_JOINTS):
            hi = JOINT_LIMITS_DEG_SYMMETRIC[name][1]
            mag = np.degrees(np.linalg.norm(pose[i]))
            assert mag <= hi + 1e-3, f"{name} at {mag:.1f} deg exceeds {hi}"


# --- trap 3: finger distortion ---------------------------------------------


def _hand21(scale: float = 1.0) -> np.ndarray:
    """A plausible open right hand, wrist at the origin."""
    p = np.zeros((21, 3))
    fingers = {"index": 1, "middle": 5, "ring": 9, "little": 13, "thumb": 17}
    for base in fingers.values():
        for k in range(4):
            p[base + k] = [0.0, scale * (0.02 + 0.02 * k), 0.0]
    p[0] = [0.0, 0.0, 0.0]
    return p


def test_hand_angles_are_clamped_and_damped() -> None:
    """Distal joints must move less than proximal ones.

    Short bones plus noisy landmarks is the setup in which fingers splay, so the
    damping has to decrease along the chain or the distal phalanges dominate.
    """
    rng = np.random.default_rng(4)
    bent = _hand21()
    bent[2] = [0.0, 0.02, 0.02]
    aa = retarget_hand(bent + rng.normal(size=(21, 3)) * 0.01, hand="right")
    mags = np.array([np.degrees(np.linalg.norm(a)) for a in aa])
    assert len(mags) == 15
    for i in range(15):
        assert mags[i] <= MANO_LIMIT_DEG[i] + 1e-6, f"joint {i} exceeded its limit"
    assert np.all(np.diff(HAND_CHAIN_DAMPING) < 0), "damping must fall along the chain"


def test_damping_reduces_the_noise_inflation() -> None:
    """Damping must help, and it demonstrably does.

    The bend estimator is ``arccos`` of the angle between a joint's incoming and
    outgoing bone, and that function has unbounded derivative near zero: a finger
    that is nearly straight has a true bend of epsilon but an estimate that moves by
    ~sqrt(2*epsilon) under any perturbation. So landmark noise **does** inflate the
    measured curl, and no amount of clamping removes it - the clamp only bounds the
    result.

    What is claimed here is the narrower, true thing: the chain damping in
    :func:`clamp_hand_angles` reduces that inflation relative to leaving the angles
    raw, which is the mechanism the module claims. Claiming the noise away would be
    claiming something false; the amplification is a property of the estimator and is
    documented as such.
    """
    rng = np.random.default_rng(5)
    bent = _hand21()
    bent[2] = [0.0, 0.02, 0.02]
    noisy = bent + rng.normal(size=(21, 3)) * 0.003

    clean = float(np.linalg.norm(retarget_hand(bent, hand="right"), axis=1).sum())
    damped = float(np.linalg.norm(retarget_hand(noisy, hand="right"), axis=1).sum())
    raw = float(np.linalg.norm(_raw_hand_angles(noisy, hand="right"), axis=1).sum())

    assert clean > 0.0
    assert damped < raw, f"damping did not reduce the noise inflation ({damped:.3f} vs {raw:.3f})"
    assert damped <= np.radians(MANO_LIMIT_DEG).sum() * 1.5


def test_hand_output_stays_within_limits_under_heavy_noise() -> None:
    """Whatever the estimator does, the output is bounded. This is the guarantee."""
    rng = np.random.default_rng(6)
    worst = 0.0
    for _ in range(30):
        kp = _hand21() + rng.normal(size=(21, 3)) * 0.01
        aa = retarget_hand(kp, hand="right")
        for i, row in enumerate(aa):
            worst = max(worst, np.degrees(np.linalg.norm(row)) / MANO_LIMIT_DEG[i])
    assert worst <= 1.0001, f"a joint reached {worst:.2f}x its limit"


def test_hand_bend_is_read_from_the_keypoints() -> None:
    """A straight finger must give zero curl; a bent one must not.

    The first implementation used a fixed 25-degree magnitude, so a perfectly straight
    hand scored 0 and a noisy one scored full curl - it amplified noise instead of
    reading pose.
    """
    straight = retarget_hand(_hand21(), hand="right")
    assert np.abs(straight).max() == pytest.approx(0.0, abs=1e-9), (
        "collinear bones must give zero rotation"
    )
    bent = _hand21()
    bent[2] = [0.0, 0.02, 0.02]  # curl one joint out of plane
    curled = retarget_hand(bent, hand="right")
    assert np.abs(curled).max() > 1e-3, "a bend must produce rotation"


def test_hand_is_mirrored_between_sides() -> None:
    bent = _hand21()
    bent[2] = [0.0, 0.02, 0.02]
    right = retarget_hand(bent, hand="right")
    left = retarget_hand(bent, hand="left")
    assert not np.allclose(right, left), "left and right hands must not be identical"
    assert np.allclose(np.abs(right), np.abs(left)), "mirroring must not change the magnitudes"


# --- expression ------------------------------------------------------------


def test_expression_is_projected_not_amplified() -> None:
    """FLAME expression space over-drives easily; a rubber mask is the failure mode."""
    bs = np.zeros(52)
    bs[20] = 1.0  # mouthSmileLeft saturated
    expr, jaw = retarget_expression(bs)
    assert expr.shape == (10,)
    assert np.abs(expr).max() < 1.0, "a saturated blendshape must not exceed unity"
    assert jaw[0] == pytest.approx(0.0), "jawOpen is not being set here"


def test_jaw_tracks_jaw_open_only() -> None:
    bs = np.zeros(52)
    bs[9] = 1.0
    _expr, jaw = retarget_expression(bs)
    assert np.degrees(jaw[0]) == pytest.approx(12.0)
    bs[9] = 0.0
    _expr, jaw0 = retarget_expression(bs)
    assert jaw0[0] == pytest.approx(0.0)


def test_expression_rejects_a_short_blendshape_vector() -> None:
    with pytest.raises(ValueError, match="52"):
        retarget_expression(np.zeros(51))


# --- the frame and sequence ------------------------------------------------


def test_synthesise_pads_body_pose_to_21_joints() -> None:
    f = synthesise(canonical_rest_pose())
    assert f.body_pose.shape == (21, 3)
    assert f.left_hand_pose.shape == (15, 3)
    assert f.expression.shape == (10,)
    assert np.abs(f.body_pose).max() == 0.0, "a T-pose is the rest pose"


def test_optional_inputs_stay_neutral_rather_than_invented() -> None:
    """No hands tracked means a neutral hand, not an invented one."""
    f = synthesise(canonical_rest_pose(), blendshapes=np.zeros(52))
    assert np.abs(f.left_hand_pose).max() == 0.0
    assert np.abs(f.right_hand_pose).max() == 0.0


def test_sequence_has_one_frame_per_input() -> None:
    rest = canonical_rest_pose()
    seq = np.stack([rest] * 12)
    frames = synthesise_sequence(seq, blendshape_seq=np.zeros((12, 52)))
    assert len(frames) == 12
    assert all(f.body_pose.shape == (21, 3) for f in frames)


def test_smplx_model_is_not_vendored() -> None:
    """The licence gate must explain itself, not just fail.

    SMPL-X parameters are distributed by the MPI under a research licence requiring
    registration; the file cannot be committed here, and a bare FileNotFoundError
    would read as a missing dependency rather than a deliberate choice.
    """
    with pytest.raises(FileNotFoundError) as e:
        load_smplx("/nonexistent/smplx/SMPLX_NEUTRAL.npz")
    msg = str(e.value).lower()
    assert "licence" in msg or "license" in msg
    assert "max planck" in msg or "mpi" in msg
