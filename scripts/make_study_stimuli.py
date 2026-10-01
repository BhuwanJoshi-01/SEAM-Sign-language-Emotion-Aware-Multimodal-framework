#!/usr/bin/env python
"""Build blinded pairwise stimulus videos for the M7 preference study.

    mesh sequence -> rendered frames -> blinded .mp4 per condition -> private key

**The blinding is done by this script, not by a human remembering to rename files.**
`guide.md` step 3 warns that naming the clips "A" and "B", or revealing which condition
produced which, is "the single easiest way to invalidate the study". A convention that
depends on someone renaming files correctly is a convention that fails once, silently,
and the study is then worthless with no visible symptom. So:

- the condition is never written into the filename, the video metadata, or the manifest
  that a rater could see;
- filenames are opaque tokens derived from a seed, not a counter;
- the condition -> filename mapping is written to **one** file, `blinding_key.json`,
  which lives in a directory named `private/` and is explicitly not part of the rater kit;
- the public manifest contains only randomised names, and a check asserts that no
  condition label appears anywhere in it.

**This script does not decide what the conditions are.** The comparison arm is a study
design choice owned by the author (see
`paper/provenance/TASK5_QUESTIONS_FOR_AUTHOR.md` §Q1), so the caller supplies the mesh
sequences per condition and says which condition each is. Nothing here invents a
baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from seam.avatar.render import Camera, render_sequence, write_mp4
from seam.seed import set_seed


def _render(seq, args, camera):
    """Render one mesh sequence, vectorised when asked.

    The vectorised renderer is bit-identical to the reference (asserted in
    tests/test_fast_render.py) and ~25x faster, which is the difference between a
    study that renders in minutes and one that renders in hours. The reference stays the
    default only so that a reader who has not met fast_render.py is not surprised.
    """
    if args.fast:
        from seam.avatar.fast_render import render_sequence_fast

        return render_sequence_fast(
            seq["vertices"], seq["faces"], width=args.width, height=args.height, camera=camera
        )
    return render_sequence(
        seq["vertices"], seq["faces"], width=args.width, height=args.height, camera=camera
    )

#: Words that must never appear in a filename a rater can see. Checked, not just avoided.
FORBIDDEN_IN_PUBLIC_NAMES = (
    "system",
    "baseline",
    "affect",
    "emotion",
    "condition",
    "proxy",
    "_a_",
    "_b_",
)


def blinding_token(seed: int, condition: str, clip_id: str) -> str:
    """An opaque, stable filename stem for one stimulus.

    Derived from a hash rather than a counter so the name carries no ordering
    information: a rater who notices that `clip_01.mp4` and `clip_02.mp4` are always the
    two conditions has learned the design from the filenames alone.
    """
    h = hashlib.sha256(f"{seed}:{condition}:{clip_id}".encode()).hexdigest()
    return h[:8]


def check_public_name_is_blind(name: str) -> None:
    """Raise if a rater-visible filename leaks the condition."""
    low = name.lower()
    for word in FORBIDDEN_IN_PUBLIC_NAMES:
        if word in low:
            raise ValueError(
                f"stimulus filename {name!r} contains {word!r}, which hints at the "
                "condition. A rater who can infer the arm from the filename makes the "
                "study worthless, so this is refused rather than renamed silently."
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        required=True,
        help=(
            "JSON describing the stimuli: a list of "
            '{"clip_id": str, "condition": str, "mesh": path-to-.glb or .npz}. '
            "The *.glb must be a frame-per-node mesh sequence (see mesh.export_glb)."
        ),
    )
    ap.add_argument("--out", default="artifacts/m7a/study")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help=(
            "trim every stimulus to its first N frames (0 = whole clip). Trimming, never "
            "padding or looping: a repeated motion would read as a different performance. "
            "Keeps paired stimuli a ratable length."
        ),
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="use the vectorised renderer (~25x faster, bit-identical output)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    spec = json.loads(Path(args.input).read_text(encoding="utf-8"))
    items = spec["stimuli"] if isinstance(spec, dict) else spec
    if not items:
        print("no stimuli in the input spec")
        return 1

    out = Path(args.out)
    public = out / "public"
    private = out / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    key: dict[str, dict[str, str]] = {}
    public_manifest: list[dict[str, object]] = []

    for item in items:
        clip_id = str(item["clip_id"])
        condition = str(item["condition"])
        mesh_path = Path(item["mesh"])
        if not mesh_path.is_file():
            print(f"missing mesh for {clip_id}/{condition}: {mesh_path}")
            return 2

        seq = _load_mesh_sequence(mesh_path)
        n_frames_full = int(seq["vertices"].shape[0])
        if args.max_frames and n_frames_full > args.max_frames:
            seq = {"vertices": seq["vertices"][: args.max_frames], "faces": seq["faces"]}
        frames = _render(seq, args, Camera(**(item.get("camera") or {})))

        token = blinding_token(args.seed, condition, clip_id)
        name = f"{token}.mp4"
        check_public_name_is_blind(name)
        write_mp4(frames, public / name, fps=args.fps)

        key.setdefault(clip_id, {})[token] = condition
        public_manifest.append(
            {
                "trial_id": clip_id,
                "clip": name,
                "n_frames": int(seq["vertices"].shape[0]),
                "fps": args.fps,
                "duration_s": round(seq["vertices"].shape[0] / args.fps, 3),
            }
        )
        print(f"  {clip_id:24s} -> {name}  ({seq['vertices'].shape[0]} frames)")

    (private / "blinding_key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    (public / "manifest.json").write_text(
        json.dumps({"stimuli": public_manifest}, indent=2) + "\n", encoding="utf-8"
    )

    # Last line of defence: assert the public manifest is actually blind before the
    # caller walks away believing it is.
    blob = (public / "manifest.json").read_text(encoding="utf-8").lower()
    for word in FORBIDDEN_IN_PUBLIC_NAMES:
        if word in blob:
            print(f"REFUSING: public manifest leaks the word {word!r}")
            return 3

    print()
    print(f"public stimuli : {public}")
    print(f"blinding key   : {private / 'blinding_key.json'}")
    print("DO NOT give any rater the private/ directory.")
    return 0


def _load_mesh_sequence(path: Path) -> dict[str, np.ndarray]:
    """Read a frame-per-node GLB into (T, V, 3) vertices plus one face array.

    All frames must share topology, which is what `MeshSequence` enforces on write.
    Checked again here because a GLB that arrived from elsewhere - a different tool, a
    hand edit - would not have been through that guard, and a topology change between
    frames renders as a body that tears apart mid-animation.
    """
    import trimesh

    scene = trimesh.load(str(path), file_type="glb", force="scene")
    if not isinstance(scene, trimesh.Scene) or not scene.geometry:
        raise ValueError(f"{path} does not load as a scene with geometry")

    names = sorted(scene.geometry.keys())
    verts: list[np.ndarray] = []
    faces_ref: np.ndarray | None = None
    for n in names:
        g = scene.geometry[n]
        if faces_ref is None:
            faces_ref = np.asarray(g.faces, dtype=np.int64)
        elif not np.array_equal(np.asarray(g.faces, dtype=np.int64), faces_ref):
            raise ValueError(
                f"{path}: geometry {n} has different faces from the first node; a mesh "
                "sequence must share one topology or the animation tears"
            )
        verts.append(np.asarray(g.vertices, dtype=np.float64))
    assert faces_ref is not None
    return {"vertices": np.stack(verts), "faces": faces_ref}


if __name__ == "__main__":
    raise SystemExit(main())
