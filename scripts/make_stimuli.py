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
    smplx_mesh,
)
from seam.avatar.pose_map import (
    anatomical_report,
    assert_anatomical,
    mediapipe_to_smpl24,
    normalise_to_rest_frame,
)
from seam.avatar.synthesis import (
    forward_kinematics,
    load_smplx,
    model_rest_pose,
    synthesise_sequence,
)
from seam.features.signpose import FACE_POINTS, POSE_POINTS
from seam.paths import default_data_root
from seam.seed import set_seed

REPO = Path(__file__).resolve().parents[1]


def _median_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Sliding-window median over time (axis 0), to kill single-frame spikes.

    MediaPipe's face and hand landmarks are not temporally smoothed (only the 33 pose
    points are, and only when the extractor enabled it), so a dropped or jittered
    detection makes a hand or the ear-derived head jump for one frame. A small median
    window removes those spikes without blurring the motion, because signing is slow
    relative to a 25 fps frame. A window of 1 is a no-op.
    """
    if window <= 1:
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    out = x.copy()
    half = window // 2
    for t in range(T):
        lo = max(0, t - half)
        hi = min(T, t + half + 1)
        out[t] = np.median(x[lo:hi], axis=0)
    return out


def build_sequence(
    landmarks: np.ndarray,
    *,
    max_frames: int | None = None,
    smooth_window: int = 5,
    model: dict | None = None,
) -> tuple[list, np.ndarray]:
    """SMPL-X parameters and joint positions for one clip.

    **Layout correction.** The extractor writes the 553 points *pose-first* -
    ``pose`` at 0..32, then the two 21-point hands, then the 478 face points
    (``tasks_api.PART_ORDER = ("pose", "left_hand", "right_hand", "face")``). The old
    :func:`seam.features.signpose.split_landmarks` assumed the opposite (face-first)
    layout and silently handed a slice of *face* landmarks to the body retargeter -
    the root cause of the anatomically impossible meshes. We slice the pose block
    explicitly here and route it through
    :func:`seam.avatar.pose_map.mediapipe_to_smpl24`, which maps MediaPipe's 33 joints
    onto SMPL's 24 and flips y (image y grows downward; the rest pose is y-up).

    The whole array's y is flipped once, up front, so the hands end up in the same
    world frame as the body they attach to - otherwise the fingers articulate in a
    frame reflected relative to the wrist. Finally a light temporal median filter
    removes per-frame spikes before retargeting, because the face/hand tracks are not
    otherwise temporally smoothed.
    """
    lm = np.asarray(landmarks, dtype=np.float32)
    if lm.shape[-2] != 553:
        raise ValueError(
            f"expected 553 landmark points (pose-first), got {lm.shape[-2]}. "
            "A different count means the extractor changed layout, and the offsets "
            "below are stale."
        )
    # image -> world: flip y for the entire array once, so body and hands share a frame.
    lm = lm.copy()
    lm[..., 1] = -lm[..., 1]

    pose_mp = lm[..., 0:33, :]   # MediaPipe Pose, 33 points
    lh = lm[..., 33:54, :]       # left hand, 21 points
    rh = lm[..., 54:75, :]       # right hand, 21 points
    if max_frames is not None:
        pose_mp, lh, rh = pose_mp[:max_frames], lh[:max_frames], rh[:max_frames]

    # Denoise per-frame spikes on the non-smoothed face/hand/pose tracks.
    pose_mp = _median_filter(pose_mp, smooth_window)
    lh = _median_filter(lh, smooth_window)
    rh = _median_filter(rh, smooth_window)

    # The reference pose must be the rest pose of the mesh that will be skinned - not the
    # hand-written stick figure the retargeter falls back to. With the real model loaded
    # this is its own joints (`J_regressor @ v_template`). Using the approximate figure
    # injects rotations the tracking never asked for; because the hips are mirrored those
    # injected errors oppose, which is what rendered as a straddle stance.
    #
    # Resolved *before* the mapping below, because the mapper uses it too: MediaPipe has no
    # spine landmarks, so the spine is derived, and deriving it as a straight line bends
    # the model's curved spine flat and leans the whole torso.
    rest = model_rest_pose(model) if model is not None else None

    # MediaPipe 33 -> SMPL 24 (y already flipped above, so flip_y=False here).
    body = mediapipe_to_smpl24(pose_mp, flip_y=False, rest=rest)

    # Put the tracked body into the rest pose's frame and units BEFORE retargeting.
    # `retarget_body` compares bone *directions* against the rest pose, so the two skeleta
    # must share an orientation and comparable proportions. The landmarks arrive in
    # image-pixel units with the limbs' depth on a different scale from the rest pose's
    # metres; retargeting those directly asks for huge rotations and the body renders lying
    # on its back. This is the step whose absence produced the "falling over" avatars.
    # It must target the same pose the retargeter will use, or the tracked body is scaled
    # to the wrong proportions before the retargeter ever sees it.
    body = normalise_to_rest_frame(body, rest=rest)

    # hand_seq is (T, 2, 21, 3): index 0 is the left hand, index 1 the right, which is
    # the order synthesise_sequence expects. transposing here would silently swap them.
    frames = synthesise_sequence(body, hand_seq=np.stack([lh, rh], axis=1), rest=rest)

    joints = np.asarray(
        [
            forward_kinematics(f.body_pose, f.global_orient, rest=rest)
            for f in frames
        ]
    )
    return frames, joints


def verify_anatomy(joints: np.ndarray) -> tuple[int, int]:
    """Count frames whose retargeted skeleton is ordered like a body.

    Returns ``(n_frames, n_passed)``. A failing frame is anatomically impossible
    (head below the pelvis, a knee above a hip, ...) and its geometry must not be
    shown or rated. We report rather than abort, because one dropout frame in a long
    clip should not throw away the whole clip.
    """
    js = np.asarray(joints, dtype=np.float64)
    if js.ndim != 3 or js.shape[-2:] != (24, 3):
        raise ValueError(f"expected (T, 24, 3) joints, got {js.shape}")
    n_pass = 0
    for j in js:
        try:
            assert_anatomical(j)
            n_pass += 1
        except ValueError:
            pass
    return js.shape[0], n_pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landmarks", default=None, help="single .npz; default: a few from data/")
    ap.add_argument("--out", default="artifacts/m7a/stimuli")
    ap.add_argument("--limit", type=int, default=3, help="clips when --landmarks is absent")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="temporal median window for denoising the face/hand tracks (1 disables)",
    )
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "SMPL-X model: either a directory of *.npy (an unzipped official download) "
            "or a single .npz/.pkl. When given, the mesh is real geometry; when absent, "
            "the joint-capsule proxy is written and the output is stamped is_proxy=true."
        ),
    )
    ap.add_argument(
        "--betas",
        type=float,
        nargs="*",
        default=None,
        help="shape coefficients; omitted means zeros (the neutral body)",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The mesh source is chosen once, here, so the per-clip loop cannot accidentally mix
    # a proxy clip into a real-geometry run. That mixture is the dangerous case: the
    # manifest would claim real geometry for part of the set and the study would silently
    # include stand-ins.
    model: dict | None = None
    if args.model:
        model = load_smplx(Path(args.model))
        missing = [
            k
            for k in ("v_template", "shapedirs", "J_regressor", "kintree_table", "weights", "f")
            if k not in model
        ]
        if missing:
            print(f"model at {args.model} is missing {missing}; refusing to run")
            return 2
        n_verts = int(np.asarray(model["v_template"]).shape[0])
        print(f"using the real SMPL-X model from {args.model} ({n_verts} vertices)")
    else:
        print(
            "no --model given: writing the joint-capsule PROXY. This is not a human body "
            "and must not be used for preference-study ratings."
        )
    is_proxy = model is None

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
        "is_proxy_mesh": is_proxy,
        "is_human_mesh": not is_proxy,
        "model_path": args.model,
        "warning": (
            "Joint-capsule proxy, NOT a human body. Sufficient to verify the pipeline, "
            "timing and export path end to end. Do NOT use for preference-study ratings: "
            "raters would be judging the stand-in rather than the retargeting."
            if is_proxy
            else (
                "Real SMPL-X geometry. This is the mesh source the preference study "
                "requires. `posedirs` and expression blend shapes are not applied - see "
                "mesh.smplx_mesh for what that omits."
            )
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
        frames, joints = build_sequence(
            landmarks,
            max_frames=args.max_frames,
            smooth_window=args.smooth_window,
            model=model,
        )
        n_j, n_pass = verify_anatomy(joints)
        if n_pass < n_j:
            print(
                f"  WARNING {uid}: {n_j - n_pass}/{n_j} frames fail anatomical "
                f"checks - geometry from these frames must not be rated"
            )

        seq: MeshSequence = (
            smplx_mesh(frames, model, betas=None if args.betas is None else np.array(args.betas))
            if model is not None
            else skeleton_proxy_mesh(frames, joints)
        )
        glb = export_glb(seq, out / f"{uid}.glb", fps=args.fps)
        params = export_parameters(frames, out / f"{uid}.params.json", fps=args.fps)

        info = describe_mesh(seq)
        if info["is_proxy"] != is_proxy:
            print(
                f"refusing to continue: manifest says is_proxy={is_proxy} but {uid} came "
                f"back is_proxy={info['is_proxy']}"
            )
            return 3
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
    if is_proxy:
        print(
            "REMINDER: this is a proxy mesh. Re-run with --model <smplx path> for real "
            "geometry before collecting any ratings."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
