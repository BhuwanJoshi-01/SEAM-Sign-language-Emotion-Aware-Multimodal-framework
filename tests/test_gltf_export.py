"""Tests for the animated glTF exporter.

The exporter has one job that is easy to get subtly wrong: produce a file whose animation
matches the meshes we already verified. Every test here is about that agreement, because a
GLB can be structurally valid, open in a viewer, and still animate limbs that detach -
which no amount of "did the file write" tells you.

Tests that need the licence-gated SMPL-X model are skipped without it, as elsewhere.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from seam.avatar.gltf_export import (
    _aa_to_quat,
    _mat_to_quat,
    _quat_to_mat,
    _vertex_normals,
    export_animated_glb,
    verify_glb_animation,
)

MODEL_ENV = "SEAM_SMPLX_MODEL"


def _model() -> dict:
    from seam.avatar.synthesis import load_smplx

    p = os.environ.get(MODEL_ENV)
    if not p or not Path(p).exists():
        pytest.skip(f"{MODEL_ENV} not set; animated GLB export needs the model")
    return load_smplx(Path(p))


def _betas(model: dict) -> np.ndarray:
    return np.zeros(np.asarray(model["shapedirs"]).shape[-1])


def _frames(n: int = 4) -> list:
    from seam.avatar.synthesis import SmplxFrame

    out = []
    for i in range(n):
        body = np.zeros((21, 3))
        body[3] = [0.15 * (i + 1), 0.0, 0.0]  # spine1, rotating per frame
        lh = np.zeros((15, 3))
        lh[:, 0] = 0.05 * (i + 1)
        out.append(
            SmplxFrame(
                global_orient=np.array([0.02 * i, 0.0, 0.0]),
                body_pose=body,
                left_hand_pose=lh,
                right_hand_pose=lh.copy(),
                global_transl=np.array([0.0, 0.0, 0.01 * i]),
            )
        )
    return out


def _read_gltf(path: Path) -> tuple[dict, bytes]:
    raw = Path(path).read_bytes()
    assert raw[:4] == b"glTF", "missing glTF magic"
    version, length = struct.unpack("<II", raw[4:12])
    assert version == 2, f"expected glTF 2.0, got {version}"
    assert length == len(raw), f"header length {length} != file size {len(raw)}"
    jlen, jtype = struct.unpack("<I", raw[12:16])[0], raw[16:20]
    assert jtype == b"JSON", f"first chunk must be JSON, got {jtype!r}"
    gj = json.loads(raw[20 : 20 + jlen].decode("utf-8"))
    blen, btype = struct.unpack("<I", raw[20 + jlen : 24 + jlen])[0], raw[24 + jlen : 28 + jlen]
    assert btype == b"BIN\x00", f"second chunk must be BIN, got {btype!r}"
    blob = raw[28 + jlen :]
    assert blen == len(blob), f"BIN length {blen} != actual {len(blob)}"
    return gj, blob


# ── maths that needs no model ────────────────────────────────────────────────


def test_quat_roundtrip() -> None:
    rng = np.random.default_rng(3)
    for _ in range(50):
        M = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(M) < 0:
            M[:, 0] *= -1
        assert np.allclose(_quat_to_mat(_mat_to_quat(M)), M, atol=1e-9)


@pytest.mark.parametrize("aa", [[0, 0, 0], [0, 0, np.pi / 2], [0.2, -0.3, 0.1], [np.pi, 0, 0]])
def test_aa_to_quat_is_unit(aa: list[float]) -> None:
    q = _aa_to_quat(np.asarray(aa, dtype=float))
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-12)


def test_vertex_normals_are_unit_and_outward() -> None:
    v = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2]], dtype=np.int64)
    n = _vertex_normals(v, f)
    assert np.allclose(np.linalg.norm(n, axis=1), 1.0)
    # (1,0,0) x (0,1,0) = (0,0,1): a CCW triangle in the xy plane faces +z.
    assert n[2, 2] > 0, "winding should give +z for a CCW triangle in the xy plane"


# ── the export itself ────────────────────────────────────────────────────────


def test_export_has_one_mesh_one_skin_and_animation(tmp_path: Path) -> None:
    model = _model()
    out = export_animated_glb(_frames(), model, tmp_path / "a.glb", fps=10.0)
    gj, _ = _read_gltf(out)
    assert len(gj["meshes"]) == 1, "one skinned mesh, not one per frame"
    assert len(gj["skins"]) == 1
    assert len(gj["animations"]) == 1
    prim = gj["meshes"][0]["primitives"][0]
    for attr in ("POSITION", "NORMAL", "JOINTS_0", "WEIGHTS_0"):
        assert attr in prim["attributes"], f"missing {attr}"
    n_joints = len(gj["skins"][0]["joints"])
    assert len(gj["nodes"]) == n_joints + 1
    assert len(gj["animations"][0]["channels"]) == 2 * n_joints


def test_export_is_not_a_frame_dump(tmp_path: Path) -> None:
    """The regression this whole module exists for.

    `mesh.export_glb` wrote one static mesh per frame: 40 nodes on clip 1372, no skin, no
    animation - a file that opens and shows a body that never moves.
    """
    import trimesh

    model = _model()
    out = export_animated_glb(_frames(8), model, tmp_path / "b.glb", fps=10.0)
    scene = trimesh.load(str(out), file_type="glb", force="scene")
    assert len(scene.geometry) == 1, f"expected 1 mesh, got {len(scene.geometry)}"


def test_attribute_names_are_the_ones_loaders_actually_map(tmp_path: Path) -> None:
    """JOINTS_0/WEIGHTS_0, not the _4 aliases.

    three.js maps only the _0 names, so a _4 file loads with `skinWeight` undefined and
    throws inside SkinnedMesh.normalizeSkinWeights. Every index in such a file is
    structurally valid, which is why an offline structural check passes and the page is
    the thing that catches it.
    """
    model = _model()
    out = export_animated_glb(_frames(2), model, tmp_path / "j.glb", fps=10.0)
    gj, _ = _read_gltf(out)
    attrs = gj["meshes"][0]["primitives"][0]["attributes"]
    assert "JOINTS_0" in attrs and "WEIGHTS_0" in attrs
    assert "JOINTS_4" not in attrs and "WEIGHTS_4" not in attrs


def test_samplers_are_interleaved_to_match_channels(tmp_path: Path) -> None:
    """Samplers are [t0, r0, t1, r1, ...] and channels index 2j / 2j+1.

    Building them as [all translations] + [all rotations] leaves every channel after the
    first joint pointing at the wrong path, which only shows up once something animates.
    """
    model = _model()
    out = export_animated_glb(_frames(3), model, tmp_path / "c.glb", fps=10.0)
    gj, _ = _read_gltf(out)
    anim = gj["animations"][0]
    for ch in anim["channels"]:
        assert ch["sampler"] == 2 * (ch["sampler"] // 2) + (
            0 if ch["target"]["path"] == "translation" else 1
        )


def test_inverse_bind_matrices_are_column_major(tmp_path: Path) -> None:
    model = _model()
    out = export_animated_glb(_frames(2), model, tmp_path / "d.glb", fps=10.0)
    gj, blob = _read_gltf(out)
    acc = gj["accessors"][gj["skins"][0]["inverseBindMatrices"]]
    view = gj["bufferViews"][acc["bufferView"]]
    off = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    flat = np.frombuffer(blob, dtype="<f4", count=acc["count"] * 16, offset=off)
    M = flat.reshape(-1, 4, 4).transpose(0, 2, 1)
    # Inverting a rest chain must land near the identity rotation block, not a transpose.
    rot = M[:, :3, :3]
    eye = np.einsum("nij,nkj->nik", rot, rot)
    assert np.allclose(eye, np.eye(3), atol=2e-3)


@pytest.mark.parametrize("frame", [0, 2])
def test_animation_matches_the_mesh_pipeline(tmp_path: Path, frame: int) -> None:
    """The load-bearing test: re-skinning the file must reproduce `smplx_mesh`."""
    model = _model()
    import os

    shapedirs = np.asarray(model["shapedirs"])
    betas = np.zeros(shapedirs.shape[-1])
    frames = _frames(4)
    out = export_animated_glb(frames, model, tmp_path / "e.glb", betas=betas, fps=10.0)
    v = verify_glb_animation(out, frames, model, frame=frame, betas=betas)
    assert v["skins"] == 1
    assert v["agrees"], f"frame {frame}: {v['max_abs_error_m']:.4f} m vs tol {v['tolerance_m']:.4f}"
    del os


def test_global_transl_is_carried(tmp_path: Path) -> None:
    """Omitting it puts the body the wrong distance from the origin in a viewer."""
    model = _model()
    shapedirs = np.asarray(model["shapedirs"])
    betas = np.zeros(shapedirs.shape[-1])
    frames = _frames(3)
    out = export_animated_glb(frames, model, tmp_path / "f.glb", betas=betas, fps=10.0)
    v = verify_glb_animation(out, frames, model, frame=2, betas=betas)
    assert v["agrees"]


def test_rejects_wrong_betas_width(tmp_path: Path) -> None:
    model = _model()
    with pytest.raises(ValueError, match="betas"):
        export_animated_glb(_frames(2), model, tmp_path / "g.glb", betas=np.zeros(7))


def test_rejects_empty_sequence(tmp_path: Path) -> None:
    model = _model()
    with pytest.raises(ValueError, match="no frames"):
        export_animated_glb([], model, tmp_path / "h.glb")


def test_dropped_weight_mass_is_reported(tmp_path: Path) -> None:
    """SMPL-X has up to 10 influences per vertex; glTF carries 4. That gap is reported."""
    model = _model()
    shapedirs = np.asarray(model["shapedirs"])
    betas = np.zeros(shapedirs.shape[-1])
    frames = _frames(3)
    out = export_animated_glb(frames, model, tmp_path / "i.glb", betas=betas, fps=10.0)
    v = verify_glb_animation(out, frames, model, frame=0, betas=betas)
    assert 0.0 <= v["weight_mass_dropped_max"] < 0.10
    assert v["tolerance_m"] > v["weight_mass_dropped_max"]
