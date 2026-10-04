"""Dump our reader's skinned vertices next to three.js's for the same file.

Isolates "is the format convention wrong" from anything SMPLer-X or SMPL-X specific.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seam.avatar.gltf_export import _quat_to_mat, export_animated_glb  # noqa: E402
from seam.avatar.synthesis import SmplxFrame, load_smplx  # noqa: E402

MODEL = os.environ["SEAM_SMPLX_MODEL"]
OUT = Path("/tmp/opencode/cmp/synth.glb")

model = load_smplx(Path(MODEL))
shapedirs = np.asarray(model["shapedirs"])
betas = np.zeros(shapedirs.shape[-1])

frames = []
for i in range(4):
    body = np.zeros((21, 3))
    body[3] = [0.15 * (i + 1), 0.0, 0.0]
    lh = np.zeros((15, 3))
    lh[:, 0] = 0.05 * (i + 1)
    frames.append(
        SmplxFrame(
            global_orient=np.array([0.02 * i, 0.0, 0.0]),
            body_pose=body,
            left_hand_pose=lh,
            right_hand_pose=lh.copy(),
            global_transl=np.array([0.0, 0.0, 0.01 * i]),
        )
    )

export_animated_glb(frames, model, OUT, betas=betas, fps=10.0)

# ── our own reading of the file, following the glTF spec the same way the
#    verifier does, and independently the reference mesh ────────────────────────
raw = OUT.read_bytes()
jlen = struct.unpack("<I", raw[12:16])[0]
gj = json.loads(raw[20 : 20 + jlen].decode())
blob = raw[20 + jlen + 8 :]
accs, views = gj["accessors"], gj["bufferViews"]
ctype = {5126: 4, 5123: 2, 5125: 4}
ncomp = {"SCALAR": 1, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read(i):
    a = accs[i]
    v = views[a["bufferView"]]
    dt = {4: "<f4", 2: "<u2"}[ctype[a["componentType"]]]
    arr = np.frombuffer(
        blob, dtype=dt, count=a["count"] * ncomp[a["type"]], offset=v.get("byteOffset", 0)
    )
    return arr.reshape(a["count"], ncomp[a["type"]])


prim = gj["meshes"][0]["primitives"][0]
v_rest = read(prim["attributes"]["POSITION"]).astype(np.float64)
j4 = read(prim["attributes"]["JOINTS_0"]).astype(np.int64)
w4 = read(prim["attributes"]["WEIGHTS_0"]).astype(np.float64)
ibm = read(gj["skins"][0]["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
n_joints = ibm.shape[0]

nodes = gj["nodes"]
parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
anim = gj["animations"][0]
frame = 0
local: dict[int, dict[str, np.ndarray]] = {}
for ch in anim["channels"]:
    s = anim["samplers"][ch["sampler"]]
    val = read(s["output"])
    assert len(val) == 4, f"sampler {ch['sampler']} has {len(val)} values for 4 keyframes"
    local.setdefault(ch["target"]["node"], {})[ch["target"]["path"]] = val[frame].astype(
        np.float64
    )

world: dict[int, np.ndarray] = {}


def node_local(ni):
    M = np.eye(4)
    d = local.get(ni, {})
    if "rotation" in d:
        M[:3, :3] = _quat_to_mat(d["rotation"])
    if "translation" in d:
        M[:3, 3] = d["translation"]
    return M


def resolve(ni):
    if ni in world:
        return world[ni]
    p = parent.get(ni)
    world[ni] = node_local(ni) if p is None else resolve(p) @ node_local(ni)
    return world[ni]


skin_mats = [resolve(1 + j) @ ibm[j] for j in range(n_joints)]
vh = np.concatenate([v_rest, np.ones((len(v_rest), 1))], axis=1)
ours = np.zeros_like(vh)
for k in range(4):
    M = np.stack([skin_mats[j] for j in j4[:, k]])
    ours += w4[:, k : k + 1] * (M @ vh[:, :, None])[:, :, 0]

from seam.avatar.mesh import smplx_mesh  # noqa: E402

ref = np.asarray(smplx_mesh([frames[0]], model, betas=betas, use_posedirs=False).vertices[0])

parents_head = [int(p) for p in np.asarray(model["kintree_table"])[0][:8]]
dump = {
    "file": str(OUT),
    "n_joints": n_joints,
    "n_verts": int(len(v_rest)),
    "ours_first3": np.round(ours[:3, :3], 6).tolist(),
    "ref_first3": np.round(ref[:3, :3], 6).tolist(),
    "ours_vs_ref_max_mm": float(np.abs(ours[:, :3] - ref).max() * 1000),
    "bind_first3": np.round(v_rest[:3], 6).tolist(),
    "ours_extent": np.round(ours[:, :3].max(axis=0) - ours[:, :3].min(axis=0), 4).tolist(),
    "ref_extent": np.round(ref.max(axis=0) - ref.min(axis=0), 4).tolist(),
    # Bone world matrices and skinning matrices, column-major flattened - the layout
    # three.js's Matrix4.fromArray / Skeleton.boneMatrices use - so the JS probe can be
    # diffed against them directly instead of by eye.
    "parents_head": parents_head,
    "node_local_head": [np.round(node_local(1 + j), 5).tolist() for j in range(4)],
    "world_head": [np.round(resolve(1 + j), 5).tolist() for j in range(4)],
    "ibm_head_colmajor": [np.round(ibm[j].T.reshape(16), 5).tolist() for j in range(4)],
    "skin_mat_head": [np.round(skin_mats[j], 5).tolist() for j in range(4)],
    "joints4_head": j4[:4].tolist(),
    "weights4_head": np.round(w4[:4], 5).tolist(),
}
Path("/tmp/opencode/cmp/ours.json").write_text(json.dumps(dump))
print(json.dumps({k: dump[k] for k in ("ours_vs_ref_max_mm", "ours_extent", "parents_head")}, indent=1))