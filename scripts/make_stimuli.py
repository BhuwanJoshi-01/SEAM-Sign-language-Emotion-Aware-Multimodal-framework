#!/usr/bin/env python
"""Generate avatar stimuli for the M7 preference study (M7a).

**What this does, end to end:**

    landmark .npz  ->  SMPL-X parameters  ->  joint positions  ->  mesh  ->  .glb
                     (synthesis)         (forward kinematics)  (mesh)

**Why it exists now, before the SMPL-X licence arrives.** The M7 preference study needs
avatar videos, and the project had none: the demo renders a table of marker magnitudes,
`serve` does not import the avatar at all, and the old `export_glb` wrote a JSON blob to
a `.glb` path. So the study had no stimuli and the reason was not the missing licence.

Everything in that chain except the final mesh evaluation is licence-free, and this
script runs it on real extracted landmarks now, producing a viewable GLB. When the
SMPL-X model arrives, `--model` switches the mesh source and nothing else changes.

**The mesh it writes today is a joint-capsule proxy, not a human body.** It is stamped
`is_proxy: true` in the GLB metadata and in the sidecar JSON, and
:func:`seam.avatar.mesh.describe_mesh` reports `is_human_mesh: false`. A proxy mistaken
for a real body is worse than no proxy, because raters would then be judging the
stand-in rather than the retargeting. **Do not collect study ratings on proxy output.**

Writes per clip:
  <out>/<utterance_id>.glb          one mesh node per frame
  <out>/<utterance_id>.params.json  the SMPL-X parameter sequence
  <out>/<utterance_id>.stimulus.json  provenance, sizes, proxy flag
and a manifest at <out>/manifest.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from seam.avatar.mesh import (
    N_JOINTS,
    MeshSequence,
    describe_mesh,
    export_glb,
    export_parameters,
    skeleton_proxy_mesh,
)
from seam.avatar.synthesis import forward_kinematics, synthesise_sequence
from seam.features.signpose import FACE_POINTS, POSE_POINTS, split_landmarks
from seam.paths import default_data_root
from seam.seed import set_seed

REPO = Path(__file__).resolve().parents[1]


def build_sequence(
    landmarks: np.ndarray, *, max_frames: int | None = None
) -> tuple[list, np.ndarray]:
    """SMPL-X parameters and joint positions for one clip."""
    _, pose, lh, rh = split_landmarks(landmarks)
    if max_frames is not None:
        pose, lh, rh = pose[:max_frames], lh[:max_frames], rh[:max_frames]

    # hand_seq is (T, 2, 21, 3): index 0 is the left hand, index 1 the right, which is the
    # order synthesise_sequence expects. transposing here would silently swap them, and
    # the two hands mirror each other so the animation would still look plausible.
    frames = synthesise_sequence(pose, hand_seq=np.stack([lh, rh], axis=1))

    joints = np.asarray([forward_kinematics(f.body_pose, f.global_orient) for f in frames])
    return frames, joints


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", default=None, help="single .npz; default: a few from data/")
    ap.add_argument("--out", default="artifacts/m7a/stimuli")
    ap.add_argument("--limit", type=int, default=3, help="clips when --landmarks is absent")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.landmarks:
        sources = sorted(Path(args.landmarks).parent.glob(f"{Path(args.landmarks).stem}.npz"))
    else:
        lm = default_data_root() / "emosign" / "landmarks"
        sources = sorted(lm.glob("*.npz"))[: args.limit]
    if not sources:
        print("no landmark .npz found; run `make landmarks` first")
        return 1

    manifest: dict[str, object] = {
        "generator": "scripts/make_stimuli.py",
        "fps": args.fps,
        "is_proxy_mesh": True,
        "is_human_mesh": False,
        "warning": (
            "Joint-capsule proxy, NOT a human body. Sufficient to verify the pipeline, "
            "timing and export path end to end. Do NOT use for preference-study ratings: "
            "raters would be judging the stand-in rather than the retargeting."
        ),
        "n_joints": N_JOINTS,
        "landmark_layout": {
            "face": FACE_POINTS,
            "pose": POSE_POINTS,
            "hands": "21 + 21",
        },
        "clips": [],
    }

    for src in sources:
        uid = src.stem
        with np.load(src) as d:
            landmarks = np.asarray(d["landmarks"], dtype=np.float32)
        frames, joints = build_sequence(landmarks, max_frames=args.max_frames)

        seq: MeshSequence = skeleton_proxy_mesh(frames, joints)
        glb = export_glb(seq, out / f"{uid}.glb", fps=args.fps)
        params = export_parameters(frames, out / f"{uid}.params.json", fps=args.fps)

        info = describe_mesh(seq)
        sidecar = {
            "utterance_id": uid,
            "source": str(src),
            "n_frames": seq.n_frames,
            "fps": args.fps,
            "glb": glb.name,
            "glb_bytes": glb.stat().st_size,
            "params": params.name,
            "mesh": info,
        }
        (out / f"{uid}.stimulus.json").write_text(json.dumps(sidecar, indent=2) + "\n")
        manifest["clips"].append(sidecar)  # type: ignore[union-attr]
        print(
            f"  {uid}: {seq.n_frames} frames, {info['n_vertices']} verts, "
            f"{info['n_faces']} faces, {glb.stat().st_size / 1024:.0f} KiB, "
            f"is_proxy={info['is_proxy']}"
        )

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    clips = manifest["clips"]
    n_clips = len(clips) if isinstance(clips, list) else 0
    print(f"\nwrote {out / 'manifest.json'} ({n_clips} clips)")
    print("REMINDER: this is a proxy mesh. Real geometry needs the SMPL-X model (guide.md step 1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
