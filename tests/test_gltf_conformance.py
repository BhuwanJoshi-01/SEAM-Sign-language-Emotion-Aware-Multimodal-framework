"""Does a real glTF runtime open what we wrote?

Why this file exists
--------------------
Every other check on the exporter re-reads the GLB with code that shares the writer's
assumptions. That is the blind spot this project has now hit twice:

* the exporter wrote ``JOINTS_4`` / ``WEIGHTS_4``. Every index in the file was
  structurally valid, ``verify_glb_animation`` reported ``agrees: true`` at 6.4 mm - and
  three.js threw ``Cannot read properties of undefined (reading 'count')`` inside
  ``SkinnedMesh.normalizeSkinWeights``, because it maps only the ``_0`` names.
* the page then crashed for two further reasons that no structural check can see: an
  ``AnimationMixer`` built with no scene root, and a bounding box measured before the
  skeleton was bound (0.40 m for a 1.72 m body).

A verifier that shares the writer's assumptions cannot detect either class. So this file
checks the file against things the writer does not control:

1. `test_attribute_names_are_in_the_spec_enum` - against the glTF 2.0 specification's own
   list of legal mesh attribute names, not against what our exporter happens to emit.
   Runs anywhere, no browser.
2. `test_three_js_opens_the_file` - loads the GLB through **three.js's own GLTFLoader and
   its own skinning implementation** in headless Chrome and asserts the result is a
   skinned mesh with the expected bone count, animation, and duration. three.js is
   vendored under `src/seam/web/vendor/three`, so this needs no network.
3. `test_posed_body_is_about_a_metre_and_a_half_tall` - the same browser reports the
   skinned vertex extent, which is what the page's camera framing depends on. This is the
   check that would have caught the collapsed-bounding-box bug at the source rather than
   as a stage full of geometry.

Skipped when Chrome is absent; never skipped silently when it is present but broken.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from seam.avatar.gltf_export import export_animated_glb

MODEL_ENV = "SEAM_SMPLX_MODEL"
VENDOR = Path(__file__).resolve().parents[1] / "src" / "seam" / "web" / "vendor" / "three"

# glTF 2.0 §3.6.2.1, mesh.primitive.attributes. Anything outside this set is a custom
# attribute: legal to *add*, but it is not skinning data, and a loader will ignore it.
SPEC_MESH_ATTRIBUTES = {
    "POSITION",
    "NORMAL",
    "TANGENT",
    "TEXCOORD",
    "TEXCOORD_1",
    "COLOR",
    "JOINTS",
    "WEIGHTS",
}

# The set-mapped names carry a semantic index: POSITION_0 is meaningless, TEXCOORD_3 is.
SPEC_INDEXED_ATTRIBUTES = {"TEXCOORD", "COLOR", "JOINTS", "WEIGHTS"}

# Names loaders actually bind to skinning. three.js r160's GLTFLoader maps exactly these
# two and nothing else - a different file, a different loader, and the alias set differs
# again, which is why this list is duplicated here rather than imported from the writer.
SKINNING_ATTRIBUTES = {"JOINTS_0", "WEIGHTS_0"}


def _model() -> dict:
    from seam.avatar.synthesis import load_smplx

    p = os.environ.get(MODEL_ENV)
    if not p or not Path(p).exists():
        pytest.skip(f"{MODEL_ENV} not set; glTF conformance needs the model")
    return load_smplx(Path(p))


def _betas(model: dict) -> np.ndarray:
    return np.zeros(np.asarray(model["shapedirs"]).shape[-1])


def _frames(n: int = 4) -> list:
    from seam.avatar.synthesis import SmplxFrame

    out = []
    for i in range(n):
        body = np.zeros((21, 3))
        body[3] = [0.15 * (i + 1), 0.0, 0.0]
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


def _gltf_json(path: Path) -> dict:
    """Parse the GLB container and return the JSON chunk.

    A deliberately separate implementation from the one in `test_gltf_export`: that file
    is about the exporter's arithmetic, this one is about the bytes on disk.
    """
    import struct

    raw = Path(path).read_bytes()
    if raw[:4] != b"glTF":
        raise AssertionError("missing glTF magic")
    version, total = struct.unpack("<II", raw[4:12])
    if version != 2:
        raise AssertionError(f"expected glTF 2.0, got {version}")
    if total != len(raw):
        raise AssertionError(f"header length {total} != file size {len(raw)}")
    jlen = struct.unpack("<I", raw[12:16])[0]
    jtype = raw[16:20]
    if jtype != b"JSON":
        raise AssertionError(f"first chunk must be JSON, got {jtype!r}")
    return json.loads(raw[20 : 20 + jlen].decode("utf-8"))


# ── 1. spec conformance, no browser needed ───────────────────────────────────


def test_attribute_names_are_in_the_spec_enum(tmp_path: Path) -> None:
    """Every attribute name must be a glTF 2.0 semantic, not a lookalike.

    Checked against the specification's enum rather than against the writer, so renaming
    `JOINTS_0` to `JOINTS_4` - or to anything else - fails here rather than in a browser.
    """
    model = _model()
    out = export_animated_glb(_frames(2), model, tmp_path / "spec.glb", fps=10.0)
    attrs = _gltf_json(out)["meshes"][0]["primitives"][0]["attributes"]

    for name in attrs:
        base, _, suffix = name.rpartition("_")
        legal = name in SPEC_MESH_ATTRIBUTES or base in SPEC_INDEXED_ATTRIBUTES
        assert legal, (
            f"{name!r} is not a glTF 2.0 mesh attribute. The spec names the joint and "
            f"weight sets JOINTS_0/WEIGHTS_0; a loader that does not recognise the name "
            f"stores it as a custom attribute and the mesh never becomes skinned."
        )
        if base in SPEC_INDEXED_ATTRIBUTES:
            assert suffix.isdigit(), f"{name!r} needs a semantic index, got {suffix!r}"

    assert SKINNING_ATTRIBUTES <= set(attrs), (
        f"skinning attributes missing: {sorted(SKINNING_ATTRIBUTES - set(attrs))}"
    )


def test_every_index_in_the_file_is_in_range(tmp_path: Path) -> None:
    """Structural bounds: a wrong index is legal JSON and an illegal document.

    Cheap, catches the whole family of "plausible file, unparseable in a viewer" bugs.
    """
    model = _model()
    out = export_animated_glb(_frames(3), model, tmp_path / "idx.glb", fps=10.0)
    gj = _gltf_json(out)

    n_acc = len(gj["accessors"])
    n_view = len(gj["bufferViews"])
    n_node = len(gj["nodes"])
    n_mesh = len(gj["meshes"])
    n_sampler = len(gj["animations"][0]["samplers"])

    def check(i: int, limit: int, what: str) -> None:
        assert isinstance(i, int) and 0 <= i < limit, f"{what} index {i} out of range [0,{limit})"

    for prim in (p for m in gj["meshes"] for p in m["primitives"]):
        for name, ai in prim["attributes"].items():
            check(ai, n_acc, f"attribute {name}")
        if "indices" in prim:
            check(prim["indices"], n_acc, "indices")
        if "targets" in prim:
            for t in prim["targets"]:
                for name, ai in t.items():
                    check(ai, n_acc, f"target {name}")
        mat = prim.get("material")
        if mat is not None:
            check(mat, len(gj.get("materials", [])), "material")

    for i, acc in enumerate(gj["accessors"]):
        if "bufferView" in acc:
            check(acc["bufferView"], n_view, f"accessor {i} bufferView")

    for skin in gj["skins"]:
        if "inverseBindMatrices" in skin:
            check(skin["inverseBindMatrices"], n_acc, "skin inverseBindMatrices")
        for j in skin["joints"]:
            check(j, n_node, "skin joint")

    for node in gj["nodes"]:
        if "mesh" in node:
            check(node["mesh"], n_mesh, "node mesh")
        if "skin" in node:
            check(node["skin"], len(gj["skins"]), "node skin")
        for c in node.get("children", []):
            check(c, n_node, "node child")

    for ch in gj["animations"][0]["channels"]:
        check(ch["sampler"], n_sampler, "channel sampler")
        check(ch["target"]["node"], n_node, "channel target node")
        assert ch["target"]["path"] in {"translation", "rotation", "scale", "weights"}, (
            f"unknown animation path {ch['target']['path']!r}"
        )

    for node in gj["scenes"][gj["scene"]]["nodes"]:
        check(node, n_node, "scene node")


def test_the_node_tree_has_no_cycle(tmp_path: Path) -> None:
    """A cyclic node graph makes three.js recurse until the tab dies.

    Nothing else in the suite would notice, and it is one line to get wrong when the
    parent list comes from `kintree_table`.
    """
    model = _model()
    out = export_animated_glb(_frames(2), model, tmp_path / "tree.glb", fps=10.0)
    nodes = _gltf_json(out)["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    for start in range(len(nodes)):
        seen, cur = set(), start
        while cur in parent:
            assert cur not in seen, f"cycle in the node tree through node {cur}"
            seen.add(cur)
            cur = parent[cur]


# ── 2 & 3. a real glTF runtime ───────────────────────────────────────────────


def _chrome() -> str:
    exe = shutil.which("google-chrome") or shutil.which("google-chrome-stable")
    if not exe:
        pytest.skip("no Chrome; the real-runtime glTF check is skipped")
    return exe


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *a: object) -> None:
        pass


def _load_in_three_js(glb: Path, root: Path) -> dict:
    """Serve `root` and let headless Chrome load `glb` through three.js's GLTFLoader.

    Returns the harness's JSON report. Raises AssertionError carrying the browser's
    message on failure, because "three.js said no" with no reason is the failure mode
    this whole file was written to eliminate.
    """
    harness = f"""<!doctype html><meta charset="utf-8"><body><pre id="r"></pre>
<script type="importmap">{{"imports":{{"three":"/three/three.module.js"}}}}</script>
<script type="module">
const out = {{}};
try {{
  const THREE = await import('three');
  const {{ GLTFLoader }} = await import('/three/examples/jsm/loaders/GLTFLoader.js');
  const g = await new GLTFLoader().loadAsync('/{glb.name}');
  let meshes = 0, skinned = 0, bones = 0;
  let sk = null;
  g.scene.traverse(o => {{
    if (o.isMesh) {{ meshes++; if (o.isSkinnedMesh) {{ skinned++; bones = o.skeleton.bones.length; sk = o; }} }}
  }});
  out.meshes = meshes; out.skinned = skinned; out.bones = bones;
  out.animations = (g.animations || []).length;
  out.duration = (g.animations && g.animations[0]) ? g.animations[0].duration : 0;
  out.tracks = (g.animations && g.animations[0]) ? g.animations[0].tracks.length : 0;
  if (sk) {{
    out.hasSkinIndex = sk.geometry.attributes.skinIndex !== undefined;
    out.hasSkinWeight = sk.geometry.attributes.skinWeight !== undefined;
    // three.js skins on the GPU, so `position` stays at the bind pose forever. Measure the
    // *skinned* result through three.js's own CPU path (SkinnedMesh.applyBoneTransform,
    // used by its raycaster) rather than by reading the attribute back.
    const mixer = new THREE.AnimationMixer(g.scene);
    const act = mixer.clipAction(g.animations[0]); act.reset().play();
    const pos = sk.geometry.attributes.position;
    const posed = (t) => {{
      mixer.update(t); g.scene.updateMatrixWorld(true); sk.skeleton.update();
      const a = new Float64Array(pos.count * 3); const v = new THREE.Vector3();
      for (let i = 0; i < pos.count; i++) {{
        v.fromBufferAttribute(pos, i);
        sk.applyBoneTransform(i, v);
        v.applyMatrix4(sk.matrixWorld);
        a[3*i] = v.x; a[3*i+1] = v.y; a[3*i+2] = v.z;
      }}
      return a;
    }};
    const ext = (a) => {{
      const mn = [1e9,1e9,1e9], mx = [-1e9,-1e9,-1e9];
      for (let i = 0; i < a.length; i++) {{
        const c = i % 3;
        if (a[i] < mn[c]) mn[c] = a[i];
        if (a[i] > mx[c]) mx[c] = a[i];
      }}
      return {{ min: mn, max: mx, size: [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]] }};
    }};
    const p0 = posed(0);
    out.bind = ext(pos.array.length === pos.count * 3
      ? Float64Array.from(pos.array) : Float64Array.from(pos.array));
    out.posed = ext(p0);
    // Displacement between two times on the same clip. A bind pose with a dummy clip
    // attached loads, has a duration, and never moves - so neither a time base nor a
    // plausible duration is evidence that anything animates.
    const dur = g.animations[0].duration;
    const p1 = posed(Math.min(dur * 0.6, 0.3));
    let moved = 0;
    for (let i = 0; i < p0.length; i++) moved = Math.max(moved, Math.abs(p0[i] - p1[i]));
    out.maxDisplacement = moved;
  }}
  out.ok = true;
}} catch (e) {{
  out.ok = false; out.error = String((e && e.stack) || e);
}}
document.getElementById('r').textContent = '__RESULT__' + JSON.stringify(out) + '__END__';
</script></body>"""
    (root / "_harness.html").write_text(harness, encoding="utf-8")
    target = root / glb.name
    if glb.resolve() != target.resolve():
        shutil.copy(glb, target)
    for sub in ("examples/jsm/loaders", "examples/jsm/utils"):
        dst = root / "three" / sub
        dst.mkdir(parents=True, exist_ok=True)
        for f in (VENDOR / sub).glob("*.js"):
            shutil.copy(f, dst / f.name)
    dst = root / "three"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(VENDOR / "three.module.js", dst / "three.module.js")

    port = _free_port()
    handler = partial(_Quiet, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        proc = subprocess.run(  # noqa: S603
            [
                _chrome(),
                "--headless=new",
                "--disable-gpu",
                "--enable-unsafe-swiftshader",
                "--no-sandbox",
                "--virtual-time-budget=20000",
                "--dump-dom",
                f"http://127.0.0.1:{port}/_harness.html",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    finally:
        httpd.shutdown()
        httpd.server_close()

    dom = proc.stdout
    if "__RESULT__" not in dom:
        raise AssertionError(
            "the three.js harness never reported. The page could not run at all, which is "
            f"a different failure from the GLB being rejected.\n"
            f"chrome rc={proc.returncode}\n{proc.stderr[-2000:]}"
        )
    blob = dom.split("__RESULT__", 1)[1].split("__END__", 1)[0]
    return json.loads(blob.replace("&quot;", '"').replace("&amp;", "&").replace("&lt;", "<"))


@pytest.mark.slow
def test_three_js_opens_the_file(tmp_path: Path) -> None:
    """three.js must load this as a skinned, animated mesh.

    The check the old suite structurally could not make. `JOINTS_4` produced a file that
    passed every offline assertion in `test_gltf_export.py` and then threw inside
    `normalizeSkinWeights` here.
    """
    model = _model()
    frames = _frames(4)
    out = export_animated_glb(frames, model, tmp_path / "real.glb", fps=10.0)
    r = _load_in_three_js(out, tmp_path)

    assert r.get("ok"), f"three.js rejected the GLB:\n{r.get('error')}"
    assert r["skinned"] == 1, (
        f"expected 1 SkinnedMesh, got {r['skinned']} (meshes={r['meshes']}). "
        f"skinIndex={r.get('hasSkinIndex')} skinWeight={r.get('hasSkinWeight')} - a mesh "
        f"without both is not skinned, however correct the weights are."
    )
    n_joints = len(_gltf_json(out)["skins"][0]["joints"])
    assert r["bones"] == n_joints, f"bones {r['bones']} != skin joints {n_joints}"
    assert r["animations"] == 1, f"expected 1 animation, got {r['animations']}"
    # Keyframes sit at 0, 1/fps, ... (n-1)/fps, so the clip spans (n-1)/fps seconds - not
    # n/fps. Asserting the wrong one here would have masked a real off-by-one in `times`.
    expected = (len(frames) - 1) / 10.0
    assert r["duration"] == pytest.approx(expected, abs=1e-3), (
        f"clip duration {r['duration']} != {expected} s for {len(frames)} keys at 10 fps"
    )
    assert r["tracks"] == 2 * n_joints, (
        f"expected a TRS track per joint ({2 * n_joints}), got {r['tracks']}"
    )


@pytest.mark.slow
def test_posed_body_is_about_a_metre_and_a_half_tall(tmp_path: Path) -> None:
    """The skinned result must be human-sized, measured through three.js's skinning.

    This is the assertion the page's camera framing rests on. Framing before the mixer
    binds measures an identity skeleton: 0.40 x 0.46 x 0.31 m for a body that is
    1.72 x 1.72 x 0.29 m, which parks the camera inside the subject's head.
    """
    model = _model()
    frames = _frames(4)
    out = export_animated_glb(frames, model, tmp_path / "size.glb", fps=10.0)
    r = _load_in_three_js(out, tmp_path)
    assert r.get("ok"), f"three.js rejected the GLB:\n{r.get('error')}"

    w, h, d = r["posed"]["size"]
    assert 1.2 < h < 2.1, f"posed height {h:.3f} m is not a human body; got {r['posed']['size']}"
    assert 0.3 < w < 2.1, f"posed width {w:.3f} m is implausible; got {r['posed']['size']}"
    assert 0.1 < d < 1.2, f"posed depth {d:.3f} m is implausible; got {r['posed']['size']}"

    # The pose must also stay near the bind volume. A skeleton that flies apart under
    # skinning still has a plausible bounding box, so compare the two rather than only
    # checking absolute size.
    for i, axis in enumerate("xyz"):
        pb = r["posed"]["size"][i]
        bb = r["bind"]["size"][i]
        assert pb < 4 * bb + 0.2, (
            f"{axis}: posed extent {pb:.3f} m is more than 4x the bind extent {bb:.3f} m - "
            f"the skin is exploding rather than posing"
        )


@pytest.mark.slow
def test_the_mesh_animates(tmp_path: Path) -> None:
    """The clip must actually move vertices.

    A skin that loads and holds still is a bind pose with an animation attached, which is
    what a per-frame static dump looked like from the outside - and what a clip whose
    channels all carry the identity rotation looks like from the inside.
    """
    model = _model()
    frames = _frames(4)
    out = export_animated_glb(frames, model, tmp_path / "anim.glb", fps=10.0)
    r = _load_in_three_js(out, tmp_path)
    assert r.get("ok"), f"three.js rejected the GLB:\n{r.get('error')}"
    assert r["maxDisplacement"] > 0.01, (
        f"skin is static: max vertex displacement over the clip is "
        f"{r['maxDisplacement'] * 1000:.2f} mm"
    )