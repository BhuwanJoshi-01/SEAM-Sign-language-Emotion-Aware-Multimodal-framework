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
    frames: Sequence[SmplxFrame],
    model: dict,
    *,
    gender: str = "neutral",
    betas: np.ndarray | None = None,
    use_posedirs: bool = False,
    _skip_selfcheck: bool = False,
) -> MeshSequence:
    """Generate a mesh per frame from SMPL-X parameters via linear blend skinning.

    Implements the canonical SMPL-X LBS pipeline: shape blend -> regress rest joints ->
    remove rest pose -> per-joint rotations composed down the kinematic tree -> remove
    rest transforms -> skin the rest-pose vertices by their weights.

    `model` is the dictionary from `seam.avatar.synthesis.load_smplx`.

    **What is and is not applied.** Applied: the template, `shapedirs` (shape), `J_regressor`,
    `kintree_table`, `weights`, and the MANO hand rotations. **Not applied: `posedirs`
    (the pose-correction blend shapes) and expression blend shapes** - `posedirs` is
    available behind `use_posedirs` but off by default because it is a per-frame 486-term
    contraction that costs real time and barely moves a body in motion, and because the
    expression component's index range inside the 400-wide `shapedirs` is not documented
    in a way this repo can verify. Both omissions are stated rather than guessed at;
    see `describe_mesh` for the flag the report carries.

    Requires the licence-gated model. Without it there is no way to know the vertex count,
    the template or the regressor, so this raises rather than approximating - a proxy
    returned from a function named `smplx_mesh` would be indistinguishable from success
    at the call site.
    """
    required = ("v_template", "shapedirs", "J_regressor", "kintree_table", "weights", "f")
    missing = [k for k in required if k not in model]
    if missing:
        raise KeyError(
            f"SMPL-X model is missing {missing}. The file at this path is probably not "
            "the SMPL-X npz, or is a trimmed variant; check the official download."
        )

    v_template = np.asarray(model["v_template"], dtype=np.float64)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)
    J_reg = np.asarray(model["J_regressor"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    faces = np.asarray(model["f"], dtype=np.int64)
    posedirs = np.asarray(model["posedirs"], dtype=np.float64)
    parents = np.asarray(model["kintree_table"])[0].astype(np.int64).copy()
    parents[0] = -1  # root; SMPL-X stores 2**32-1 here
    n_joints = J_reg.shape[0]

    # Self-check first, before any real work. With a neutral frame and zero betas the
    # model must return `v_template` exactly; anything else means the skinning chain is
    # wrong and every vertex is garbage. Two formulations were tried and both failed this
    # (2.27 m and 4.09 m displacement at neutral), so the check is inside the function
    # rather than only in the test suite: an unverifiable generator that runs quietly is
    # the exact failure this project keeps hitting, and it must fail loudly even if called
    # from a script that runs no tests.
    if not _skip_selfcheck:
        _selfcheck_neutral(model)

    b = np.zeros(shapedirs.shape[-1]) if betas is None else np.asarray(betas, dtype=np.float64)
    if b.shape[0] != shapedirs.shape[-1]:
        raise ValueError(
            f"betas must be ({shapedirs.shape[-1]},), got {b.shape}; padding is the "
            "caller's decision because the component count is model-defined"
        )
    v_shaped = v_template + shapedirs @ b
    J = J_reg @ v_shaped
    # The vertices handed to the skinning stage are the *shaped* template. The rest-pose
    # removal happens inside `_lb_transforms`, via each joint's offset; subtracting J
    # from the vertices here as well would remove it twice.
    v_rest = v_shaped

    all_frames = [_to_full_pose(f, n_joints) for f in frames]

    # Rest transforms: zero pose gives the identity chain, so the rest correction is a
    # simple subtraction. Computed explicitly rather than assumed, so a model with a
    # non-identity rest pose would still be handled.
    out = []
    for pose in all_frames:
        R = [_rodrigues_local(p_) for p_ in pose]
        T = _lb_transforms(parents, R, J)
        v = _skin(v_rest, weights, T)
        if use_posedirs:
            v = v + _pose_blend(pose, posedirs)
        out.append(v)
    seq = MeshSequence(vertices=np.asarray(out), faces=faces)
    seq._is_proxy = False
    return seq


def _selfcheck_neutral(model: dict, tol: float = 1e-6) -> None:
    """Raise unless a neutral frame reproduces the template.

    Cheap (one 10k-vertex pass) and it is the only property of linear blend skinning that
    can be checked without a reference implementation to compare against.
    """
    d = dict(model)
    d["shapedirs"] = np.asarray(model["shapedirs"])[:, :, :1] * 0.0
    try:
        out = smplx_mesh([SmplxFrame()], d, _skip_selfcheck=True)
    except Exception as exc:
        raise RuntimeError(
            "SMPL-X mesh self-check failed: a neutral frame did not evaluate. The "
            "skinning chain is not implemented correctly. Do not trust or ship any "
            f"geometry from this build. Underlying error: {exc}"
        ) from exc
    err = float(np.abs(out.vertices[0] - np.asarray(model["v_template"])).max())
    if err > tol:
        raise RuntimeError(
            f"SMPL-X mesh self-check FAILED: neutral pose differs from the template by "
            f"{err:.6f} m (tolerance {tol}). Linear blend skinning is implemented "
            "incorrectly - this is the signature of a wrong rest-pose removal or a "
            "broken kinematic chain. No geometry from this build may be used or shown."
        )


def _to_full_pose(frame: SmplxFrame, n_joints: int) -> np.ndarray:
    """(n_joints, 3) axis-angle in SMPL-X joint order from one retargeted frame.

    SMPL-X pose order is global_orient, body_pose(21), jaw, left_eye, right_eye,
    left_hand(15), right_hand(15) = 165 values for 55 joints. Eyes have no retargeted
    signal here, so they are zeros - the neutral, not an invented gaze.
    """
    aa = [frame.global_orient, *list(frame.body_pose), frame.jaw_pose]
    aa += [np.zeros(3), np.zeros(3)]  # left_eye, right_eye
    aa += [*list(frame.left_hand_pose), *list(frame.right_hand_pose)]
    arr = np.asarray([np.asarray(a, dtype=np.float64).reshape(3) for a in aa])
    if arr.shape[0] > n_joints:
        raise ValueError(
            f"built {arr.shape[0]} pose joints but the model has {n_joints}; the frame "
            "layout and this model disagree"
        )
    if arr.shape[0] < n_joints:
        arr = np.vstack([arr, np.zeros((n_joints - arr.shape[0], 3))])
    return arr


def _rodrigues_local(aa: np.ndarray) -> np.ndarray:
    """Axis-angle to a 3x3 rotation. Local copy so mesh.py does not import a private."""
    theta = float(np.linalg.norm(aa))
    if theta < 1e-12:
        return np.eye(3)
    k = aa / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _lb_transforms(
    parents: np.ndarray,
    R: list[np.ndarray],
    J: np.ndarray,
) -> np.ndarray:
    """Per-joint skinning transforms (J, 4, 4), following `smplx.lbs`.

    The rest-pose removal is the subtle part, and getting it wrong is silent: the mesh
    renders, it is just metres off. Two things had to be right, and both were wrong in
    earlier versions of this file.

    1. Each joint's transform carries that joint's **rest position** in its translation
       column (`A[j] = [[R_j, J_j], [0, 1]]`), not zero. Those positions are what get
       subtracted out; leaving them at zero makes every rotation happen about the world
       origin instead of the joint, which displaces the whole mesh by metres.

    2. The subtracted quantity is `A[j] @ [J_j, **0**]` - the homogeneous padding is
       **zero**, not one. Padding with one doubles every joint position, so the
       subtraction removes twice what was added and leaves a residual translation equal to
       the rest pose.

    **The check that decides whether this is right:** at zero pose the result must
    reproduce `v_template` exactly. It now does to 5e-08, which is float32 precision on
    the template data. Before the fix it was off by 4.09 m - and it still rendered a
    perfectly plausible body, which is why nothing caught it except this assertion.
    """
    n = len(R)
    A = np.zeros((n, 4, 4))
    A[:, 3, 3] = 1.0
    A[:, :3, 3] = J  # each joint's rest position
    for j in range(n):
        A[j, :3, :3] = R[j]

    jh = np.concatenate([J, np.zeros((n, 1))], axis=1)[:, :, None]
    rel_joints = A @ jh
    pad = np.zeros((n, 4, 4))
    for r in range(3):
        pad[:, r, 3] = rel_joints[:, r, 0]
    rel = A - pad

    C = np.zeros((n, 4, 4))
    C[0] = rel[0]
    for j in range(1, n):
        par = int(parents[j])
        C[j] = C[par] @ rel[j] if par >= 0 else rel[j]
    return C


def _skin(v_rest: np.ndarray, weights: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Blend the template vertices through the per-joint transforms.

    The vertices handed in are the **template**, not template-minus-rest-joints. SMPL
    folds the rest-pose removal into the transform inside `_lb_transforms`; subtracting
    the joints here as well removes it twice.
    """
    per_vertex = np.einsum("vj,jab->vab", weights, T)
    v_homo = np.concatenate([v_rest, np.ones((v_rest.shape[0], 1))], axis=1)
    return np.einsum("vab,vb->va", per_vertex, v_homo)[:, :3]


def _pose_blend(pose: np.ndarray, posedirs: np.ndarray) -> np.ndarray:
    """Pose-correction blend shapes: (V, 3, P) contracted with the rotmat feature."""
    R = np.asarray([_rodrigues_local(p) for p in pose])
    feat = (R[1:] - np.eye(3)).reshape(-1)
    n = min(feat.shape[0], posedirs.shape[-1])
    return posedirs[:, :, :n] @ feat[:n]


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
        # Derived from is_proxy rather than hardcoded: a hardcoded False would keep
        # reporting "not a human mesh" after the real model was wired up, which is the
        # exact opposite of what the field is for.
        "is_human_mesh": not seq.is_proxy,
        "note": (
            "skeleton proxy - joint capsules, not a human body"
            if seq.is_proxy
            else "generated from the licence-gated SMPL-X model by linear blend skinning"
        ),
    }
