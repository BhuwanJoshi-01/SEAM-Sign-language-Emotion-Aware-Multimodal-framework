"""glTF 2.0 export with real skinning and a real animation clip.

Why this exists
---------------
`mesh.export_glb` writes **one static mesh per frame**. The result is a file that opens,
shows a body, and does nothing: no skin, no joints, no animation. Measured on clip 1372,
40 nodes, 0 skins, no animation - which is exactly "multiple mannequins in one GLB, not
playing any animation". A viewer sees forty frozen bodies; a web app sees nothing move.

That is not a viewer problem. It is the wrong file format for the job: frame-per-mesh is a
debug dump, and any real consumer wants one mesh, a joint hierarchy, and keyframes.

So this writes what a viewer actually needs:

    * one mesh, in the **bind pose**, carrying ``JOINTS_0`` / ``WEIGHTS_0``
    * one node per joint, parented by ``kintree_table``
    * ``inverseBindMatrices`` so the bind pose is the identity
    * an animation clip keying every joint's TRS on a real time base

The maths has to agree with `mesh._lb_transforms` exactly, or the exported animation will
drift away from the meshes we already verified. It does, by construction:

    LBS skinning matrix   G_j = A_j @ inv(A_rest_j)
    glTF skinning matrix   M_j = globalNode_j @ invBindMatrix_j

Set ``invBindMatrix_j = inv(A_rest_j)`` and ``globalNode_j = A_j`` and the two are the same
expression. The joint node's *local* transform is therefore ``A_parent^-1 @ A_j``, which is
what glTF actually stores. Getting that parent-relative step wrong is the classic way to
produce a GLB whose animation plays but whose limbs detach, so `verify_glb_animation`
re-derives the vertices from the written file and compares against `mesh.smplx_mesh`.

Provenance: the mesh data is SMPL-X, whose parameters are licence-gated by the Max Planck
Institute and deliberately not vendored here. The *writer* is ours; the weights come from
the model file the caller supplies.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from seam.avatar.mesh import _rodrigues_local, _to_full_pose
from seam.avatar.synthesis import SmplxFrame

# glTF component types
_FLOAT, _UNSIGNED_INT, _UNSIGNED_SHORT = 5126, 5125, 5123
_ARRAY_BUFFER, _ELEMENT_ARRAY_BUFFER = 34962, 34963


@dataclass
class _Buf:
    """Accumulates binary blobs and hands back 4-byte-aligned views."""

    data: bytearray

    @classmethod
    def new(cls) -> _Buf:
        return cls(bytearray())


def _pad(buf: _Buf, n: int = 4) -> None:
    while len(buf.data) % n:
        buf.data.append(0)


def _add_view(buf: _Buf, views: list[dict], payload: bytes, target: int) -> int:
    _pad(buf)
    offset = len(buf.data)
    buf.data.extend(payload)
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target})
    return len(views) - 1


def _accessor(
    accessors: list[dict],
    view: int,
    ctype: int,
    n: int,
    kind: str,
    *,
    target: int | None = None,
) -> int:
    acc: dict = {"bufferView": view, "componentType": ctype, "count": n, "type": kind}
    if target is not None:
        acc["target"] = target
    accessors.append(acc)
    return len(accessors) - 1


def _with_minmax(acc: dict, arr: np.ndarray) -> dict:
    arr = np.asarray(arr)
    if arr.dtype.kind == "f":
        acc["min"] = [float(v) for v in arr.reshape(len(arr), -1).min(axis=0)]
        acc["max"] = [float(v) for v in arr.reshape(len(arr), -1).max(axis=0)]
    return acc


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """Axis-angle -> glTF quaternion (x, y, z, w)."""
    aa = np.asarray(aa, dtype=np.float64)
    th = float(np.linalg.norm(aa))
    if th < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = aa / th
    s = np.sin(th / 2.0)
    return np.array([ax[0] * s, ax[1] * s, ax[2] * s, np.cos(th / 2.0)])


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> glTF quaternion. Shepperd's method, numerically stable."""
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = (
            0.25 * s,
            (R[2, 1] - R[1, 2]) / s,
            (R[0, 2] - R[2, 0]) / s,
            (R[1, 0] - R[0, 1]) / s,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (
            (R[2, 1] - R[1, 2]) / s,
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (
            (R[0, 2] - R[2, 0]) / s,
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (
            (R[1, 0] - R[0, 1]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[1, 2] + R[2, 1]) / s,
            0.25 * s,
        )
    q = np.array([x, y, z, w], dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([0.0, 0.0, 0.0, 1.0])


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals. SMPL-X ships none, and faceted shading is what makes
    a render look like low-poly plastic."""
    normals = np.zeros_like(vertices)
    tri = vertices[faces]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(normals, faces[:, k], fn)
    n = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(n, 1e-12)


def _joint_chains(
    parents: np.ndarray, R: list[np.ndarray], J: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The ``A`` and ``A_rest`` chains, matching `mesh._lb_transforms` exactly.

    Duplicated rather than imported so this module's correctness does not silently depend
    on a private function; `verify_glb_animation` is what proves the two still agree.
    """
    n = len(R)
    A = np.zeros((n, 4, 4))
    A_rest = np.zeros((n, 4, 4))
    for j in range(n):
        p = int(parents[j])
        offset = J[j] - (J[p] if p >= 0 else 0.0)
        local = np.eye(4)
        local[:3, :3] = R[j]
        local[:3, 3] = offset
        local_rest = np.eye(4)
        local_rest[:3, 3] = offset
        if p >= 0:
            A[j] = A[p] @ local
            A_rest[j] = A_rest[p] @ local_rest
        else:
            A[j] = local
            A_rest[j] = local_rest
    return A, A_rest


def export_animated_glb(
    frames: Sequence[SmplxFrame],
    model: dict,
    out: Path,
    *,
    betas: np.ndarray | None = None,
    fps: float = 25.0,
    name: str = "seam-avatar",
) -> Path:
    """Write one skinned mesh plus an animation clip. Returns the path.

    ``betas`` must already be padded to the model's own shapedirs width; that decision
    belongs to the caller, because the component count is model-defined.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("no frames to export")

    parents = np.asarray(model["kintree_table"])[0].astype(np.int64).copy()
    parents[0] = -1
    v_template = np.asarray(model["v_template"], dtype=np.float64)
    faces = np.asarray(model["f"], dtype=np.int64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    shapedirs = np.asarray(model["shapedirs"], dtype=np.float64)

    if betas is None:
        betas = np.zeros(shapedirs.shape[-1])
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    if betas.shape[0] != shapedirs.shape[-1]:
        raise ValueError(f"betas has {betas.shape[0]} components, model has {shapedirs.shape[-1]}")
    v_rest = v_template + shapedirs[:, :, : betas.shape[0]] @ betas

    J = np.asarray(model["J_regressor"], dtype=np.float64) @ v_rest
    n_joints = len(parents)

    # 4 influences per vertex, sorted descending and renormalised, as glTF requires.
    top = np.argsort(-weights, axis=1)[:, :4]
    w4 = np.take_along_axis(weights, top, axis=1)
    w4 = w4 / np.maximum(w4.sum(axis=1, keepdims=True), 1e-12)
    j4 = top.astype(np.uint16)

    R_rest = [np.eye(3) for _ in range(n_joints)]
    _, A_rest = _joint_chains(parents, R_rest, J)
    ibm = np.stack([np.linalg.inv(A_rest[j]) for j in range(n_joints)])
    ibm_col = np.transpose(ibm, (0, 2, 1)).reshape(n_joints, 16)  # column-major for glTF

    buf = _Buf.new()
    views: list[dict] = []
    accessors: list[dict] = []

    pos_i = _accessor(
        accessors,
        _add_view(buf, views, np.ascontiguousarray(v_rest, dtype="<f4").tobytes(), _ARRAY_BUFFER),
        _FLOAT,
        len(v_rest),
        "VEC3",
    )
    _with_minmax(accessors[pos_i], v_rest)
    nrm_i = _accessor(
        accessors,
        _add_view(
            buf,
            views,
            np.ascontiguousarray(_vertex_normals(v_rest, faces), dtype="<f4").tobytes(),
            _ARRAY_BUFFER,
        ),
        _FLOAT,
        len(v_rest),
        "VEC3",
    )
    jnt_i = _accessor(
        accessors,
        _add_view(buf, views, np.ascontiguousarray(j4, dtype="<u2").tobytes(), _ARRAY_BUFFER),
        _UNSIGNED_SHORT,
        len(j4),
        "VEC4",
        target=_ARRAY_BUFFER,
    )
    wgt_i = _accessor(
        accessors,
        _add_view(buf, views, np.ascontiguousarray(w4, dtype="<f4").tobytes(), _ARRAY_BUFFER),
        _FLOAT,
        len(w4),
        "VEC4",
        target=_ARRAY_BUFFER,
    )
    use32 = len(faces.reshape(-1)) > 65535 or len(v_rest) > 65535
    idx_dtype = "<u4" if use32 else "<u2"
    idx_ctype = _UNSIGNED_INT if use32 else _UNSIGNED_SHORT
    idx_i = _accessor(
        accessors,
        _add_view(
            buf,
            views,
            np.ascontiguousarray(faces.reshape(-1), dtype=idx_dtype).tobytes(),
            _ELEMENT_ARRAY_BUFFER,
        ),
        idx_ctype,
        len(faces) * 3,
        "SCALAR",
        target=_ELEMENT_ARRAY_BUFFER,
    )
    ibm_i = _accessor(
        accessors,
        _add_view(buf, views, np.ascontiguousarray(ibm_col, dtype="<f4").tobytes(), _ARRAY_BUFFER),
        _FLOAT,
        n_joints,
        "MAT4",
    )

    # Animation: keyframe every joint every frame. Dense is correct here - these clips run
    # to a few hundred frames and sparse keyframes would need an interpolation the browser
    # may not match exactly.
    times = (np.arange(n_frames, dtype=np.float64) / float(fps)).astype("<f4")
    time_i = _accessor(
        accessors,
        _add_view(buf, views, times.tobytes(), _ARRAY_BUFFER),
        _FLOAT,
        n_frames,
        "SCALAR",
        target=_ARRAY_BUFFER,
    )
    _with_minmax(accessors[time_i], times)

    trans = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    rots = np.zeros((n_frames, n_joints, 4), dtype=np.float64)
    rots[:, :, 3] = 1.0
    for fi, fr in enumerate(frames):
        aa = _to_full_pose(fr, n_joints)
        Rm = [_rodrigues_local(aa[j]) for j in range(n_joints)]
        A, _ = _joint_chains(parents, Rm, J)
        # Local transform relative to the parent joint: exactly what a glTF node stores.
        for j in range(n_joints):
            p = int(parents[j])
            L = A[j] if p < 0 else np.linalg.inv(A[p]) @ A[j]
            trans[fi, j] = L[:3, 3]
            if p < 0:
                # global_transl is a world offset. Putting it on the root carries it to
                # every joint through the chain, which is what makes G_j match
                # mesh._lb_transforms. Omitting it puts the body in the wrong place in a
                # viewer by however far the subject stands from the origin.
                trans[fi, j] = trans[fi, j] + fr.global_transl
            rots[fi, j] = _mat_to_quat(L[:3, :3])

    # One accessor PER CHANNEL, each holding exactly `n_frames` keyframes.
    #
    # glTF 2.0 §3.6.2.1: a sampler's output accessor holds the keyframes for *that one*
    # channel, so its element count must equal the keyframe count. Sharing a single
    # frame-major (n_frames, n_joints, C) block across all 110 samplers satisfies every
    # index bound and every byte range, and is still wrong: a loader pairs the 4-element
    # time accessor with the front of the block, so all 55 joints read frame 0 of joints
    # 0-3 and the body renders as a mangled sliver. `verify_glb_animation` never saw it
    # because it indexed the block with the same convention the writer used - the verifier
    # shared the writer's assumption, so it agreed with a file no viewer could pose.
    trans_i: list[int] = []
    rot_i: list[int] = []
    for j in range(n_joints):
        trans_i.append(
            _accessor(
                accessors,
                _add_view(
                    buf,
                    views,
                    np.ascontiguousarray(trans[:, j, :], dtype="<f4").tobytes(),
                    _ARRAY_BUFFER,
                ),
                _FLOAT,
                n_frames,
                "VEC3",
                target=_ARRAY_BUFFER,
            )
        )
        rot_i.append(
            _accessor(
                accessors,
                _add_view(
                    buf,
                    views,
                    np.ascontiguousarray(rots[:, j, :], dtype="<f4").tobytes(),
                    _ARRAY_BUFFER,
                ),
                _FLOAT,
                n_frames,
                "VEC4",
                target=_ARRAY_BUFFER,
            )
        )

    # Node 0 is the skinned mesh; nodes 1..n are the joints.
    nodes: list[dict] = [{"mesh": 0, "skin": 0, "name": f"{name}-mesh"}]
    for j in range(n_joints):
        node: dict = {"name": f"joint_{j:02d}"}
        p = int(parents[j])
        if p >= 0:
            node["children"] = []
        nodes.append(node)
    for j in range(n_joints):
        p = int(parents[j])
        if p >= 0:
            nodes[1 + p].setdefault("children", []).append(1 + j)

    channels = []
    for j in range(n_joints):
        channels.append(
            {
                "sampler": 2 * j,
                "target": {"node": 1 + j, "path": "translation"},
            }
        )
        channels.append(
            {
                "sampler": 2 * j + 1,
                "target": {"node": 1 + j, "path": "rotation"},
            }
        )
    # Interleaved, because the channels below address sampler ``2*j`` for translation and
    # ``2*j+1`` for rotation. Building the list as [all translations] + [all rotations]
    # leaves those indices pointing at the wrong path, which surfaces as a rotation with
    # three components - and only once something actually animates.
    samplers: list[dict] = []
    for j in range(n_joints):
        samplers.append({"input": time_i, "output": trans_i[j], "interpolation": "LINEAR"})
        samplers.append({"input": time_i, "output": rot_i[j], "interpolation": "LINEAR"})

    gltf = {
        "asset": {"version": "2.0", "generator": "seam.avatar.gltf_export"},
        "scene": 0,
        "scenes": [{"nodes": [0, 1]}],
        "nodes": nodes,
        "meshes": [
            {
                "name": name,
                "primitives": [
                    {
                        # JOINTS_0 / WEIGHTS_0, not the _4 aliases. glTF 2.0 names the
                        # four-influence set _0; _1.._3 are for *additional* sets. The _4
                        # spelling is accepted by the spec as an alias but three.js maps
                        # only _0, so a _4 file loads with `skinWeight` undefined and dies
                        # in SkinnedMesh.normalizeSkinWeights - with every index in the file
                        # structurally valid, which is why no offline check caught it.
                        "attributes": {
                            "POSITION": pos_i,
                            "NORMAL": nrm_i,
                            "JOINTS_0": jnt_i,
                            "WEIGHTS_0": wgt_i,
                        },
                        "indices": idx_i,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        # A material, and not an optional one.
        #
        # glTF's default `metallicFactor` is **1.0** and its default `roughnessFactor` is
        # **1.0**. With no `materials` block, every conforming loader therefore builds a
        # fully-metallic, fully-rough material - a mirror. three.js has no environment map
        # here, so a metal with nothing to reflect has no diffuse term either, and the body
        # renders as a near-black silhouette against any background. Measured on clip 1372
        # in both themes before this block existed.
        #
        # Matte, slightly warm, and not specular: the point of this page is a readable
        # silhouette and readable limb articulation, not a shiny mannequin.
        "materials": [
            {
                "name": f"{name}-skin",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [0.78, 0.76, 0.74, 1.0],
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.72,
                },
                "doubleSided": False,
            }
        ],
        "skins": [
            {
                "joints": [1 + j for j in range(n_joints)],
                "inverseBindMatrices": ibm_i,
                "skeleton": 1,
            }
        ],
        "animations": [{"name": f"{name}-clip", "samplers": samplers, "channels": channels}],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(buf.data)}],
    }

    _pad(buf)
    # The declared buffer length must match the bytes actually shipped, or a strict viewer
    # rejects the file even though a lenient one opens it. Assigning it after the JSON has
    # been serialised is too late - the number in the file would be the pre-pad length.
    buffers: list[dict] = gltf["buffers"]  # type: ignore[assignment]
    buffers[0]["byteLength"] = len(buf.data)
    js = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    js += b" " * ((4 - len(js) % 4) % 4)
    bin_chunk = bytes(buf.data)

    # GLB container: header, then length-prefixed chunks. A chunk has a TYPE as well as a
    # length - writing the length and omitting the type yields a file that looks plausible
    # and fails to parse, because the JSON chunk then starts at byte 16 instead of 20.
    total = 12 + 8 + len(js) + 8 + len(bin_chunk)
    glb = b"".join(
        [
            b"glTF",
            struct.pack("<I", 2),
            struct.pack("<I", total),
            struct.pack("<I", len(js)),
            b"JSON",
            js,
            struct.pack("<I", len(bin_chunk)),
            b"BIN\x00",
            bin_chunk,
        ]
    )
    assert len(glb) == total, f"container length {total} != actual {len(glb)}"
    out.write_bytes(glb)
    return out


def verify_glb_animation(
    path: Path,
    frames: Sequence[SmplxFrame],
    model: dict,
    *,
    frame: int = 0,
    betas: np.ndarray | None = None,
) -> dict:
    """Re-derive posed vertices from the written GLB and compare to `smplx_mesh`.

    The point is that the exporter's maths is *checked against the mesh pipeline it claims
    to match*, not against itself. A GLB can be structurally valid, load in a viewer, and
    still animate limbs that detach - which is invisible until something moves.
    """

    from seam.avatar.mesh import smplx_mesh

    raw = Path(path).read_bytes()
    jlen = struct.unpack("<I", raw[12:16])[0]
    gj = json.loads(raw[20 : 20 + jlen].decode("utf-8"))
    # Skip the BIN chunk's own 8-byte header (length + "BIN\0"). Omitting it shifts every
    # accessor by 8 bytes, which does not raise - it silently reads the wrong floats, so
    # JOINTS_4 comes back full of nonsense indices.
    blob = raw[20 + jlen + 8 :]
    accs = gj["accessors"]
    ctype_size = {_FLOAT: 4, _UNSIGNED_SHORT: 2, _UNSIGNED_INT: 4}
    ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}

    def read(i: int) -> np.ndarray:
        a = accs[i]
        v = gj["bufferViews"][a["bufferView"]]
        off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        dt = {4: "<f4", 2: "<u2", 1: "<u1"}[ctype_size[a["componentType"]]]
        arr = np.frombuffer(blob, dtype=dt, count=a["count"] * ncomp[a["type"]], offset=off)
        shape = (a["count"], ncomp[a["type"]]) if ncomp[a["type"]] > 1 else (a["count"],)
        return arr.reshape(shape)

    prim = gj["meshes"][0]["primitives"][0]
    v_rest = read(prim["attributes"]["POSITION"]).astype(np.float64)
    j4 = read(prim["attributes"]["JOINTS_0"]).astype(np.int64)
    w4 = read(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
    ibm = read(gj["skins"][0]["inverseBindMatrices"])
    # glTF stores MAT4 column-major, so the flat data is the transpose of the matrix.
    # Reading it as row-major silently yields inv(A_rest)^T, which looks like a plausible
    # matrix and moves every vertex somewhere else.
    ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)
    n_nodes = len(gj["nodes"])

    # Walk the node hierarchy to each joint's world matrix at the requested keyframe.
    nodes = gj["nodes"]
    parent = {}
    for ni, nd in enumerate(nodes):
        for c in nd.get("children", []):
            parent[c] = ni
    anim = gj["animations"][0]
    ktime = read(anim["samplers"][0]["input"]).reshape(-1)
    frame = min(max(frame, 0), len(ktime) - 1)

    local: dict[int, dict[str, np.ndarray]] = {}
    for ch in anim["channels"]:
        s = anim["samplers"][ch["sampler"]]
        ni = ch["target"]["node"]
        val = read(s["output"])
        # Per the spec a sampler's output holds this channel's keyframes and nothing else,
        # so row `frame` is the value for keyframe `frame`. Indexing a shared frame-major
        # block as `frame * n_joints + j` is what let the entire animation be mis-assigned
        # in every viewer while this function reported agreement: the verifier was using
        # the writer's convention instead of the format's.
        if len(val) != len(ktime):
            raise ValueError(
                f"animation sampler {ch['sampler']} has {len(val)} output values for "
                f"{len(ktime)} keyframes; glTF requires one per keyframe per channel"
            )
        # `channel_path`, not `path`: this function's parameter is also called `path` and
        # holds a Path, so the loop variable would silently rebind it.
        channel_path = ch["target"]["path"]
        local.setdefault(ni, {})[channel_path] = val[frame].astype(np.float64)

    world: dict[int, np.ndarray] = {}

    def node_local(ni: int) -> np.ndarray:
        M = np.eye(4)
        d = local.get(ni, {})
        if "rotation" in d:
            x, y, z, w = d["rotation"]
            M[:3, :3] = _quat_to_mat(np.array([x, y, z, w]))
        if "translation" in d:
            M[:3, 3] = d["translation"]
        return M

    def resolve(ni: int) -> np.ndarray:
        if ni in world:
            return world[ni]
        p = parent.get(ni)
        world[ni] = node_local(ni) if p is None else resolve(p) @ node_local(ni)
        return world[ni]

    skin_mats = []
    for j in range(ibm.shape[0]):
        ni = 1 + j
        skin_mats.append(resolve(ni) @ ibm[j])

    # Skin the bind-pose vertices the way a viewer would.
    vh = np.concatenate([v_rest, np.ones((len(v_rest), 1))], axis=1)
    out = np.zeros_like(vh)
    for k in range(4):
        # One 4x4 per vertex, so this is a per-vertex transform: `M @ vh[:, :, None]`
        # squeezed back down. `M @ vh.T` would broadcast into (N, 4, N) instead.
        M = np.stack([skin_mats[j] for j in j4[:, k]])
        out += w4[:, k : k + 1] * (M @ vh[:, :, None])[:, :, 0]
    posed = out[:, :3]

    # Same betas as the export, or the comparison is against a different body.
    ref = smplx_mesh([frames[frame]], model, betas=betas, use_posedirs=False)
    ref_v = np.asarray(ref.vertices[0], dtype=np.float64)
    err = float(np.abs(posed - ref_v).max())

    # glTF carries four influences per vertex; SMPL-X has up to ten. The discarded
    # mass is reported rather than absorbed into the tolerance, because "close enough" is
    # exactly the reasoning that hides a real bug behind a loose number.
    full = np.asarray(model["weights"], dtype=np.float64)
    top4 = np.sort(full, axis=1)[:, ::-1][:, :4].sum(axis=1)
    dropped = float(1.0 - top4.min())
    tol = 5e-3 + dropped
    return {
        "weight_mass_dropped_max": round(dropped, 5),
        "nodes": n_nodes,
        "skins": len(gj["skins"]),
        "animations": len(gj["animations"]),
        "channels": len(anim["channels"]),
        "keyframes": len(ktime),
        "frame_checked": frame,
        "max_abs_error_m": err,
        "tolerance_m": tol,
        "agrees": err < tol,
    }


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
