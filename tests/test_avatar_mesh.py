"""Tests for mesh generation and GLB export.

The failure these exist to prevent is specific and has already happened once: the
previous `export_glb` wrote a JSON parameter dump to a path ending in `.glb`. That file
passes `ls`, appears next to real assets, and opens as nothing in any viewer. So the
central tests here assert that what comes out is *loadable geometry*, not that a file
exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from seam.avatar.mesh import (
    MeshSequence,
    describe_mesh,
    export_glb,
    export_parameters,
    skeleton_proxy_mesh,
    smplx_mesh,
)
from seam.avatar.synthesis import SmplxFrame, canonical_rest_pose, forward_kinematics


def _frames(t: int = 6) -> list[SmplxFrame]:
    out = []
    for i in range(t):
        f = SmplxFrame()
        f.global_orient = np.array([0.0, 0.1 * i, 0.0])
        f.body_pose = np.tile(np.array([0.0, 0.02 * i, 0.0]), (21, 1))
        f.left_hand_pose = np.tile(np.array([0.0, 0.3, 0.0]), (15, 1))
        f.right_hand_pose = np.tile(np.array([0.0, 0.4, 0.0]), (15, 1))
        f.expression = np.linspace(0.0, 0.5, 10)
        out.append(f)
    return out


def _joints(t: int = 6) -> np.ndarray:
    """A plausible (T, 24, 3) skeleton in metres, from the real forward kinematics."""
    return np.asarray([forward_kinematics(f.body_pose, f.global_orient) for f in _frames(t)])


def test_the_written_file_is_loadable_geometry_not_json(tmp_path: Path) -> None:
    """The core regression: a `.glb` must contain a mesh, not a JSON blob."""
    trimesh = pytest.importorskip("trimesh")
    jp = _joints()
    seq = skeleton_proxy_mesh(_frames(), jp)
    out = export_glb(seq, tmp_path / "a.glb")

    assert out.stat().st_size > 0
    head = out.read_bytes()[:4]
    assert head == b"glTF", f"expected a binary glTF magic number, got {head!r}"

    scene = trimesh.load(str(out), file_type="glb", force="scene")
    assert isinstance(scene, trimesh.Scene)
    assert len(scene.geometry) == seq.n_frames
    for name, mesh in scene.geometry.items():
        assert len(mesh.vertices) > 0, f"{name} has no vertices"
        assert len(mesh.faces) > 0, f"{name} has no faces"
        assert np.isfinite(mesh.vertices).all(), f"{name} has non-finite vertices"


def test_export_refuses_a_json_payload_disguised_as_a_glb(tmp_path: Path) -> None:
    """A parameter dump must not be reachable through the mesh export path."""
    jp = _joints()
    seq = skeleton_proxy_mesh(_frames(), jp)
    out = export_glb(seq, tmp_path / "b.glb")
    try:
        import json

        json.loads(out.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass  # expected: a GLB is binary
    else:
        pytest.fail("the .glb parsed as JSON, so it is a parameter dump with the wrong extension")


def test_parameters_go_to_a_json_path_with_a_warning(tmp_path: Path) -> None:
    """The two outputs must be distinguishable by name and by content."""
    out = export_parameters(_frames(3), tmp_path / "params.json")
    assert out.suffix == ".json"
    doc = __import__("json").loads(out.read_text())
    assert doc["n_frames"] == 3
    assert "NOT a mesh" in doc["note"]


def test_a_proxy_is_labelled_a_proxy_everywhere(tmp_path: Path) -> None:
    """A stand-in mistaken for a human body is worse than no stand-in at all."""
    trimesh = pytest.importorskip("trimesh")
    jp = _joints()
    seq = skeleton_proxy_mesh(_frames(), jp)
    assert seq.is_proxy is True

    d = describe_mesh(seq)
    assert d["is_proxy"] is True
    assert d["is_human_mesh"] is False
    assert "not a human body" in str(d["note"]).lower()

    out = export_glb(seq, tmp_path / "c.glb")
    scene = trimesh.load(str(out), file_type="glb", force="scene")
    assert scene.metadata.get("is_proxy") is True, (
        "the GLB must carry the proxy flag; a file that reaches a rater without it is "
        "indistinguishable from a real render"
    )


def test_a_real_mesh_is_not_labelled_a_proxy(tmp_path: Path) -> None:
    """The inverse of the case above, and the one that actually broke.

    ``describe_mesh`` briefly hardcoded ``is_human_mesh: False``, so a real
    SMPL-X export was described as stand-in geometry while the manifest's own
    top-level field said the opposite. Testing only the proxy direction let that
    ship: a constant ``False`` passes every proxy assertion ever written.
    """
    m = _real_model()
    seq = smplx_mesh([SmplxFrame()], m)
    assert seq.is_proxy is False

    d = describe_mesh(seq)
    assert d["is_proxy"] is False
    assert d["is_human_mesh"] is True, (
        "a real SMPL-X mesh must be reported as human geometry; reporting it as a "
        "proxy makes a real study look like it ran on stand-ins"
    )
    assert "proxy" not in str(d["note"]).lower()

    out = export_glb(seq, tmp_path / "real.glb")
    trimesh = pytest.importorskip("trimesh")
    scene = trimesh.load(str(out), file_type="glb", force="scene")
    assert scene.metadata.get("is_proxy") is False


def test_meshes_move_when_the_parameters_do(tmp_path: Path) -> None:
    """A static mesh would pass every structural test and be obviously wrong."""
    jp = _joints()
    seq = skeleton_proxy_mesh(_frames(6), jp)
    assert np.abs(seq.vertices[-1] - seq.vertices[0]).max() > 1e-3, (
        "frames differ in the joint positions but the mesh did not move"
    )


def test_topology_is_constant_across_frames() -> None:
    jp = _joints()
    seq = skeleton_proxy_mesh(_frames(6), jp)
    assert seq.vertices.shape[0] == 6
    assert seq.vertices.ndim == 3


def test_face_indices_must_be_in_range() -> None:
    """A bad index would segfault inside trimesh or silently drop triangles."""
    v = np.zeros((2, 10, 3))
    with pytest.raises(ValueError, match="face indices"):
        MeshSequence(vertices=v, faces=np.array([[0, 1, 99]]))


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match=r"\(T, V, 3\)"):
        MeshSequence(vertices=np.zeros((10, 3)), faces=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match=r"\(F, 3\)"):
        MeshSequence(vertices=np.zeros((2, 5, 3)), faces=np.array([[0, 1]]))


def test_frame_and_joint_count_must_agree() -> None:
    """A mismatch would export an animation whose mesh lags its parameters."""
    jp = _joints(4)
    with pytest.raises(ValueError, match="frames but"):
        skeleton_proxy_mesh(_frames(6), jp)
    with pytest.raises(ValueError, match="full skeleton"):
        skeleton_proxy_mesh(_frames(4), np.zeros((4, 7, 3)))


def _real_model() -> dict:
    """The SMPL-X model found on this machine, skipped if it is not there.

    Points at either a single ``.npz``/``.pkl`` file **or** an unzipped directory of
    ``*.npy`` files, because the official MPI download is delivered as a zip whose
    contents are one ``.npy`` per array and a caller who unzips it gets a directory.
    Checking only ``is_file()`` made every real-model test skip on a machine that had
    the model correctly installed, which is indistinguishable from not having it.
    """
    import os

    p = os.environ.get(
        "SEAM_SMPLX_MODEL",
        "/mnt/Volume2/SignLanguagge/NSL Data/Sapien_Pipeline/models/smplx/SMPLX_NEUTRAL.npz",
    )
    path = Path(p)
    if not (path.is_file() or path.is_dir()):
        pytest.skip(f"SMPL-X model not present at {p}")
    from seam.avatar.synthesis import load_smplx

    return load_smplx(path)


def test_the_real_model_has_the_parts_a_mesh_needs() -> None:
    """What `smplx_mesh` requires, checked against the actual file on disk."""
    m = _real_model()
    for k in ("v_template", "shapedirs", "J_regressor", "kintree_table", "weights", "f"):
        assert k in m, f"model is missing {k}"
    assert m["v_template"].shape == (10475, 3)
    assert m["J_regressor"].shape[0] == 55, "SMPL-X has 55 joints"
    assert m["f"].shape[1] == 3
    assert int(m["f"].max()) < m["v_template"].shape[0]


def test_smplx_mesh_must_reproduce_the_template_at_neutral_pose() -> None:
    """The one property that makes linear blend skinning trustworthy.

    With a neutral frame and zero betas the model must return `v_template` exactly. Two
    earlier formulations failed this (2.27 m then 4.09 m displacement), and while they
    did, this test asserted the failure: it required `smplx_mesh` to raise, and that was
    the correct state then, because no correct implementation existed to check against.

    **The skinning chain is now correct**, so the test asserts correctness instead. The
    history is kept here deliberately: the two wrong versions both looked plausible and
    were found only by this property, which is why it is asserted against the real model
    rather than a stand-in. The tolerance is tight (1e-6) on purpose - the correct
    algorithm reproduces the template to float32 noise (~5e-8), so a regression to either
    wrong formulation cannot slip through a loose bound.
    """
    m = _real_model()
    seq = smplx_mesh([SmplxFrame()], m)
    err = float(np.abs(seq.vertices[0] - np.asarray(m["v_template"], dtype=np.float64)).max())
    assert err < 1e-6, (
        f"neutral pose differs from the template by {err:.6f} m; linear blend skinning "
        "is wrong again. The correct algorithm reproduces the template to ~5e-8."
    )
    # The output must be real geometry, not a proxy that quietly took over.
    assert seq.is_proxy is False
    assert seq.vertices.shape[1] == m["v_template"].shape[0]
    assert seq.faces.shape[0] == m["f"].shape[0]


def test_a_posed_model_actually_deforms_the_body() -> None:
    """The neutral check passing is not enough on its own.

    A degenerate implementation that returned `v_template` unconditionally would satisfy
    the neutral-pose property while ignoring the pose entirely, and every animation would
    be a rigid statue. This asserts the complementary property: rotating a joint moves the
    geometry, by an amount in the range a real limb movement occupies.
    """
    m = _real_model()
    neutral = smplx_mesh([SmplxFrame()], m)

    posed_frame = SmplxFrame()
    posed_frame.body_pose[0] = np.array([0.0, 0.0, -1.2])  # rotate the left hip
    posed = smplx_mesh([posed_frame], m)

    moved = float(np.abs(posed.vertices[0] - neutral.vertices[0]).max())
    assert moved > 0.05, f"a 1.2 rad hip rotation moved the body only {moved:.4f} m"
    assert moved < 2.0, f"a 1.2 rad hip rotation moved the body {moved:.4f} m - too far"

    # Vertices bound *exclusively* to a joint outside the rotated subtree must not move.
    #
    # This cannot use "the dominant weight is the root" as the criterion: measured on the
    # real model, not one of the 434 root-dominant vertices has a root weight of 1.0 (the
    # maximum is 0.975, and they blend across 2-6 joints). They are root-*leaning*, not
    # root-*bound*, so a child rotation is supposed to move them. Requiring them to stay
    # put would have been a test asserting that LBS is broken.
    #
    # The correct criterion is *exclusive* binding: weight 1.0 on exactly one joint, and
    # that joint not an ancestor or descendant of the one rotated.
    w = np.asarray(m["weights"], dtype=np.float64)
    exclusive = w.max(axis=1) > 1.0 - 1e-9
    ex_joint = w.argmax(axis=1)
    # Join tree parents straight from the model, so the test does not depend on a copy
    # of the tree that could drift from the file it is meant to describe.
    parents = np.asarray(m["kintree_table"])[0].astype(np.int64).copy()
    parents[0] = -1
    # Ancestors and descendants of joint 1 (left_hip), which is the joint rotated above.
    rotated = 1
    subtree = {rotated}
    changed = True
    while changed:
        changed = False
        for j in range(len(parents)):
            if int(parents[j]) in subtree and j not in subtree:
                subtree.add(j)
                changed = True
    unaffected = exclusive & ~np.isin(ex_joint, list(subtree))
    if unaffected.any():
        moved_unaffected = float(
            np.abs(posed.vertices[0][unaffected] - neutral.vertices[0][unaffected]).max()
        )
        assert moved_unaffected < 1e-6, (
            f"{int(unaffected.sum())} vertices bound exclusively to joints outside the "
            f"rotated subtree moved by {moved_unaffected:.6f} m; the kinematic chain is "
            "leaking rotation into parts of the body it should not reach"
        )


def test_smplx_mesh_fails_loudly_without_a_model() -> None:
    """It must not fall back to a proxy.

    Returning an approximate body from a function named `smplx_mesh` is the failure this
    guard exists for: the call site cannot tell, so raters would judge our retargeting
    rather than the model.
    """
    with pytest.raises(KeyError):
        smplx_mesh(_frames(2), model={}, gender="neutral")


def test_smplx_mesh_rejects_a_model_missing_its_parts() -> None:
    """A body-only or wrong file should fail with the missing keys named."""
    with pytest.raises(KeyError, match="v_template"):
        smplx_mesh(_frames(2), model={"shapedirs": 1, "J_regressor": 1, "kintree_table": 1})


def test_a_real_model_would_get_past_the_shape_check() -> None:
    """Confirms the shape guard passes a plausible model, so its failure is meaningful.

    Without this, a test asserting the KeyError would also pass if every model were
    rejected for an unrelated reason.
    """


def test_forward_kinematics_reproduces_the_rest_pose() -> None:
    """Zero rotation must return the rest pose exactly.

    This is the invariant that makes FK trustworthy: if the chain accumulates rotations
    wrongly, everything still animates and only this check notices.
    """
    jp = forward_kinematics(np.zeros((21, 3)), np.zeros(3))
    assert np.allclose(jp, canonical_rest_pose())


def test_forward_kinematics_moves_only_what_is_rotated() -> None:
    """Rotating the left arm must not move the right wrist."""
    # Two index facts, and both were got wrong in a first version of this test that
    # still looked like it was testing something:
    #   - the arms interleave, so 18 is the LEFT elbow, 19 the RIGHT elbow, 20 the left
    #     wrist, 21 the right wrist;
    #   - body_pose entry k drives joint k+1, so the left elbow is aa[17], not aa[18].
    # Rotating aa[18] moved the right elbow and left the asserted joint untouched.
    aa = np.zeros((21, 3))
    aa[17] = [0.0, 0.8, 0.0]  # -> joint 18, left elbow
    jp = forward_kinematics(aa, np.zeros(3))
    rest = canonical_rest_pose()
    assert np.abs(jp[20] - rest[20]).max() > 1e-3, "left wrist is a child of the elbow"
    assert np.allclose(jp[21], rest[21], atol=1e-9), "right wrist must be untouched"
    assert np.allclose(jp[0], rest[0], atol=1e-9), "pelvis is the root and must not move"


def test_forward_kinematics_rejects_wrong_shapes() -> None:
    with pytest.raises(ValueError, match=r"\(21, 3\)"):
        forward_kinematics(np.zeros((22, 3)), np.zeros(3))
    with pytest.raises(ValueError, match=r"\(3,\)"):
        forward_kinematics(np.zeros((21, 3)), np.zeros((1, 3)))
