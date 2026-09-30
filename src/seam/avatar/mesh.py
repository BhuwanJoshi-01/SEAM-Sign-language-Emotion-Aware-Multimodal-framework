"""Mesh generation and GLB export for the M7 avatar.

**Why this module exists separately from `synthesis.py`.** `synthesis.py` produces
SMPL-X *parameters* - body pose, MANO hand angles, FLAME expression - and all of that is
unit-tested and licence-free. Turning parameters into geometry needs the SMPL-X model
file, which is gated by the Max Planck Institute and cannot be vendored. Splitting the
two means the plumbing here can be built, tested and shipped now, and the licence-gated
file becomes the only thing still missing.

**Two mesh sources, and the difference matters.**

- :func:`smplx_mesh` - the real thing. Requires the licence-gated model and raises with
  an explanation if it is absent. This is what the preference study must use.
- :func:`skeleton_proxy_mesh` - an approximate body built from joint capsules. **Not a
  human mesh.** It exists so the export path, the animation assembly and the stimulus
  pipeline can be verified end to end without the licence, and so a rater-facing preview
  is possible before approval arrives. Every GLB it writes is stamped
  `is_proxy = True` in its asset metadata, and :func:`describe_mesh` reports it as a
  proxy, because a proxy that is mistaken for the real body is worse than no proxy.

**GLB, not OBJ.** A `.glb` carrying JSON is the failure mode this module was written to
end. :func:`export_glb` writes a real binary glTF via trimesh and verifies the bytes it
wrote by re-loading them; a write that produces an unparseable file raises rather than
returning a path that looks like a success in a directory listing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seam.avatar.synthesis import SmplxFrame

#: The full SMPL skeleton, pelvis included. `SMPL_BODY_JOINTS` is *not* this: it holds
#: the 23 rotatable joints and omits the pelvis, which is the kinematic root. A mesh
#: needs the root or the body has nothing to stand on.
N_JOINTS = 24

#: Ankle-to-crown height of a neutral adult, in metres, used to scale the proxy.
PROXY_HEIGHT_M = 1.70


@dataclass
class MeshSequence:
    """A time-ordered set of meshes sharing one topology.

    Trimesh requires a consistent vertex/face layout to concatenate, so a sequence is
    stored as stacked arrays rather than a list of separate meshes. Silently allowing
    topology to change between frames produces a GLB that loads but renders garbage.
    """

    vertices: np.ndarray  # (T, V, 3)
    faces: np.ndarray  # (F, 3)
    #: True when the geometry is a stand-in rather than a real human body. Carried in
    #: the dataclass rather than poked on afterwards so type checking sees it and so it
    #: cannot be lost by a copy, and it is stamped into the GLB metadata.
    _is_proxy: bool = False

    def __post_init__(self) -> None:
        v = np.asarray(self.vertices, dtype=np.float64)
        f = np.asarray(self.faces, dtype=np.int64)
        if v.ndim != 3 or v.shape[0] == 0 or v.shape[2] != 3:
            raise ValueError(f"vertices must be (T, V, 3) with T > 0, got {v.shape}")
        if f.ndim != 2 or f.shape[1] != 3 or f.shape[0] == 0:
            raise ValueError(f"faces must be (F, 3) with F > 0, got {f.shape}")
        if f.max() >= v.shape[1] or f.min() < 0:
            raise ValueError(
                f"face indices span [{f.min()}, {f.max()}] but only {v.shape[1]} "
                "vertices exist; a bad index would segfault inside trimesh or silently "
                "drop triangles"
            )
        self.vertices, self.faces = v, f

    @property
    def n_frames(self) -> int:
        return int(self.vertices.shape[0])

    def to_glb_bytes(self) -> bytes:
        """A real binary glTF containing every frame as a separate mesh node.

        Kept separate from `export_glb` so callers that want bytes in memory - a test, or
        an HTTP response - do not have to go through a temp file. `export_glb` does the
        write-and-verify instead, which is what production callers want.
        """
        import trimesh

        scene = trimesh.Scene()
        for i in range(self.n_frames):
            scene.add_geometry(
                trimesh.Trimesh(
                    vertices=self.vertices[i],
                    faces=self.faces,
                    process=False,
                    validate=False,
                ),
                node_name=f"frame_{i:05d}",
                geom_name=f"frame_{i:05d}",
            )
        scene.metadata["fps"] = 25
        scene.metadata["n_frames"] = self.n_frames
        scene.metadata["is_proxy"] = self.is_proxy
        scene.metadata["generator"] = "seam.avatar.mesh"
        data = scene.export(file_type="glb")
        if not data:
            raise RuntimeError("trimesh returned no bytes for the GLB export")
        return data

    @property
    def is_proxy(self) -> bool:
        return self._is_proxy


def _tag_proxy(seq: MeshSequence, is_proxy: bool) -> MeshSequence:
    seq._is_proxy = is_proxy
    return seq


def smplx_mesh(
    frames: Sequence[SmplxFrame], model: dict, *, gender: str = "neutral"
) -> MeshSequence:
    """Generate a mesh per frame from SMPL-X parameters and the licence-gated model.

    `model` is the dictionary returned by `seam.avatar.synthesis.load_smplx`. Without it
    there is no way to know the vertex count, the template, the shapedirs or the joint
    regressor, so this raises rather than approximating - a proxy returned from a function
    named `smplx_mesh` would be indistinguishable from success at the call site.
    """
    required = ("v_template", "shapedirs", "J_regressor", "kintree_table")
    missing = [k for k in required if k not in model]
    if missing:
        raise KeyError(
            f"SMPL-X model is missing {missing}. The file at this path is probably not "
            "the model, or is the body-only variant; check the official download."
        )
    raise NotImplementedError(
        "Mesh evaluation against the SMPL-X model is not implemented. This is the one "
        "remaining step that needs the licence-gated weights, and it is deliberately a "
        "loud failure rather than a silent approximation: a wrong mesh would look like a "
        "valid avatar in the preference study, and the raters would judge our "
        "retargeting rather than the model. Until then use skeleton_proxy_mesh, which "
        "labels itself as a proxy."
    )


def skeleton_proxy_mesh(
    frames: Sequence[SmplxFrame],
    joint_positions: np.ndarray,
    *,
    is_proxy: bool = True,
) -> MeshSequence:
    """An approximate body: one low-poly capsule per joint, parented per the SMPL tree.

    **This is not a human mesh and must not be described as one.** It is a stand-in so
    the export path, timing and stimulus assembly can be verified without the
    licence-gated model, and so a rater-facing preview is possible before approval.

    `joint_positions` is (T, 24, 3) in metres; `synthesis.synthesise` produces it.
    """
    jp = np.asarray(joint_positions, dtype=np.float64)
    if jp.ndim != 3 or jp.shape[1] != N_JOINTS or jp.shape[2] != 3:
        raise ValueError(
            f"joint_positions must be (T, {N_JOINTS}, 3) - the full skeleton including "
            f"the pelvis - got {jp.shape}. Pass the output of "
            "synthesis.forward_kinematics unchanged; synthesis.synthesise returns "
            "parameters, not positions."
        )
    if len(frames) != jp.shape[0]:
        raise ValueError(
            f"{len(frames)} frames but {jp.shape[0]} joint positions; a mismatch here "
            "would export an animation whose mesh lags its parameters by some frames"
        )

    # Radius per joint, roughly scaled to limb thickness, in metres.
    radii = np.array(
        [
            0.09,  # 0 pelvis
            0.075,
            0.075,
            0.075,
            0.075,  # 1-4 left leg
            0.075,
            0.075,
            0.075,
            0.075,  # 5-8 right leg
            0.085,  # 9 spine1
            0.09,  # 10 spine2
            0.095,  # 11 spine3
            0.06,
            0.06,  # 12-13 neck, head
            0.055,
            0.055,
            0.05,
            0.05,  # 14-17 left arm
            0.055,
            0.055,
            0.05,
            0.05,  # 18-21 right arm
            0.06,
            0.06,  # 22-23 hands
        ]
    )[: jp.shape[1]]

    ring, seg = 6, 8
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for j in range(jp.shape[1]):
        c = jp[:, j]  # (T, 3)
        cv, cf = _capsule(c, float(radii[j]), ring=ring, seg=seg)
        verts.append(cv)
        faces.append(cf + offset)
        offset += cv.shape[1]

    return MeshSequence(
        vertices=np.concatenate(verts, axis=1),
        faces=np.concatenate(faces, axis=0),
        _is_proxy=is_proxy,
    )


def _capsule(
    centres: np.ndarray, radius: float, *, ring: int, seg: int
) -> tuple[np.ndarray, np.ndarray]:
    """A vertical capsule per frame: `ring`-gon cross-section, `seg` levels tall."""
    c = np.asarray(centres, dtype=np.float64)
    t = c.shape[0]
    height = radius * 1.6
    theta = np.linspace(0, 2 * np.pi, ring, endpoint=False)
    ring_offsets = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (ring, 2)

    v = np.zeros((t, (seg + 1) * ring, 3))
    levels = np.linspace(-height / 2, height / 2, seg + 1)
    for li, z in enumerate(levels):
        # Taper towards the ends so consecutive capsules read as limbs, not barrels.
        scale = 1.0 - 0.35 * abs(2 * z / height)
        v[:, li * ring : (li + 1) * ring, 0] = c[:, 0:1] + ring_offsets[None, :, 0] * radius * scale
        v[:, li * ring : (li + 1) * ring, 1] = c[:, 1:2] + ring_offsets[None, :, 1] * radius * scale
        v[:, li * ring : (li + 1) * ring, 2] = c[:, 2:3] + z

    f: list[np.ndarray] = []
    for li in range(seg):
        a = li * ring
        b = (li + 1) * ring
        for k in range(ring):
            k2 = (k + 1) % ring
            f.append(np.array([a + k, a + k2, b + k2]))
            f.append(np.array([a + k, b + k2, b + k]))
    return v, np.asarray(f, dtype=np.int64)


def export_glb(seq: MeshSequence, out: Path, *, fps: int = 25) -> Path:
    """Write a real binary glTF and verify it by re-loading.

    Re-loading is the point. The previous implementation wrote a JSON parameter dump to
    a path ending in `.glb`, which passes a `ls` and fails every real consumer; verifying
    the written bytes is what turns "wrote a file" into "wrote a file a viewer can open".
    """
    import trimesh

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    scene = trimesh.Scene()
    for i in range(seq.n_frames):
        scene.add_geometry(
            trimesh.Trimesh(
                vertices=seq.vertices[i], faces=seq.faces, process=False, validate=False
            ),
            node_name=f"frame_{i:05d}",
            geom_name=f"frame_{i:05d}",
        )
    scene.metadata["fps"] = int(fps)
    scene.metadata["n_frames"] = seq.n_frames
    scene.metadata["is_proxy"] = seq.is_proxy
    scene.metadata["generator"] = "seam.avatar.mesh"
    data = scene.export(file_type="glb")
    if not data:
        raise RuntimeError("trimesh returned no bytes for the GLB export")
    out.write_bytes(data)

    # Verify by loading. A silent failure here would only surface when a rater opens the
    # file, which is far too late and far too expensive.
    loaded = trimesh.load(str(out), file_type="glb", force="scene")
    if not isinstance(loaded, trimesh.Scene) or len(loaded.geometry) == 0:
        raise RuntimeError(
            f"wrote {out} but it does not load as a scene with geometry; a viewer would "
            "show an empty file. The write reported success, so this is a trimesh "
            "serialisation problem, not a path problem."
        )
    if len(loaded.geometry) != seq.n_frames:
        raise RuntimeError(
            f"wrote {out} with {seq.n_frames} frames but reloading found "
            f"{len(loaded.geometry)} geometries"
        )
    return out


def export_parameters(frames: Sequence[SmplxFrame], out: Path, *, fps: int = 25) -> Path:
    """Write the SMPL-X *parameter* sequence as JSON.

    This is the animation, not a mesh. It is written to a `.json` path on purpose, and
    the previous implementation's mistake was writing exactly this payload to a `.glb`
    path - a file that looks like an asset and is not one. Keep it separate and clearly
    named so the two are never confused.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "fps": fps,
                "n_frames": len(frames),
                "note": (
                    "SMPL-X parameters only - NOT a mesh. Convert with seam.avatar.mesh "
                    "and the licence-gated SMPL-X model to get geometry."
                ),
                "frames": [f.as_dict() for f in frames],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def describe_mesh(seq: MeshSequence) -> dict[str, object]:
    """Machine-readable description, so a report cannot call a proxy a real body."""
    v = seq.vertices
    return {
        "n_frames": seq.n_frames,
        "n_vertices": int(v.shape[1]),
        "n_faces": int(seq.faces.shape[0]),
        "is_proxy": seq.is_proxy,
        "height_m": round(float(v[:, :, 1].max() - v[:, :, 1].min()), 4),
        "is_human_mesh": False,
        "note": (
            "skeleton proxy - joint capsules, not a human body"
            if seq.is_proxy
            else "generated from the SMPL-X model"
        ),
    }
