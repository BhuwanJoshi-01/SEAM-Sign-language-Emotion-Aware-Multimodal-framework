"""The demo, end to end, on real clips: video in, comparable avatar videos out.

What this decides
-----------------
M7 needs a comparison arm, and both study scripts refused to invent one - correctly. This
script picks it, and the choice is *not* a synthetic stub:

    Arm A  SMPLer-X   learned whole-body SMPL-X regression (this project's new front end)
    Arm B  landmarks  MediaPipe -> per-joint rotation solve (what the project did before)

Both arms are real, both already exist in this repository, and neither was written for the
experiment. Arm B is the honest baseline because it is the thing the upgrade has to beat:
it is this project's own output, so a win is attributable to the front end rather than to
a stand-in built to lose. A proxy-mesh or scripted-motion baseline would answer a
different and less useful question.

Two arms of *existing* code can still fail in different ways, so the manifest records, per
clip and per arm, the measurements that distinguish a working pipeline from a plausible
one: person-detection coverage, how many frames the regressor collapsed and had to be
interpolated, how much the body actually moves, and whether the head stayed above the
pelvis. A clip can render a smooth, watchable, entirely meaningless video, so the demo
reports these instead of only shipping pictures.

Usage
-----
    PYTHONPATH=src python scripts/make_avatar_demo.py --limit 4
    PYTHONPATH=src python scripts/make_avatar_demo.py --limit 4 --fast
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from seam.avatar.gltf_export import export_animated_glb, verify_glb_animation
from seam.avatar.mesh import describe_mesh, smplx_mesh
from seam.avatar.pose_map import mediapipe_to_smpl24, normalise_to_rest_frame
from seam.avatar.render import Camera, render_frame
from seam.avatar.synthesis import (
    load_smplx,
    model_rest_pose,
    synthesise_sequence,
)
from seam.perception import smplerx as sx

DEFAULT_VIDEOS = Path("/mnt/DevProd/seam_data/emosign/video")
DEFAULT_LANDMARKS = Path("/mnt/DevProd/seam_data/emosign/landmarks")


def _median_filter(x: np.ndarray, window: int = 5) -> np.ndarray:
    """Per-frame median filter over time. Same denoising the landmark arm has always used."""
    if window < 3 or len(x) < window:
        return x
    pad = window // 2
    padded = np.pad(x, ((pad, pad), *((0, 0),) * (x.ndim - 1)), mode="edge")
    return np.stack([np.median(padded[i : i + window], axis=0) for i in range(len(x))], axis=0)


def _write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        for i, f in enumerate(frames):
            import PIL.Image

            PIL.Image.fromarray(f).save(Path(td) / f"{i:05d}.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-framerate",
            f"{fps}",
            "-i",
            f"{td}/%05d.png",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(path),
        ]
        subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def _fit_camera(
    vertices: list[np.ndarray], width: int, height: int, margin: float = 1.25
) -> Camera:
    """A camera far enough back that the whole body fits, for the whole clip.

    The renderer's default 2.6 m suits a standing figure filling the frame. A seated
    signer puts the pelvis at the origin with the head near 0.7 m and the feet near
    -0.5 m, so a fixed camera either crops the skull or - as a first attempt here did -
    shrinks the body to a speck, which communicates nothing about the pipeline.

    The distance is derived from the geometry once per clip, so it never changes frame to
    frame and cannot be mistaken for camera motion. ``fov_deg`` is vertical, so the
    vertical extent is what has to fit; the horizontal follows because the frame is
    taller than it is wide.
    """
    allv = np.concatenate(vertices)
    # 99th percentile, not max: a single stray vertex - and the landmark arm has them -
    # would otherwise push the camera far enough back to shrink both arms to specks.
    radius = float(np.percentile(np.linalg.norm(allv, axis=1), 99))
    half = np.radians(Camera().fov_deg) / 2.0
    # Farthest vertex must land inside the cone, whatever direction it lies in. Using the
    # bounding radius rather than the vertical extent is what keeps this correct for any
    # pose, including one where an arm is thrown toward the camera.
    return dataclasses.replace(Camera(), distance_m=max(0.6, radius / np.tan(half) * margin))


def _side_by_side(a: list[np.ndarray], b: list[np.ndarray], gap: int = 8) -> list[np.ndarray]:
    """Label-free A/B stack. No arm name is drawn on the frames on purpose.

    A demo that says which side is which invites the viewer to grade the pipeline instead
    of looking at it, and the same file feeds the blinded study, which must not leak.
    """
    n = min(len(a), len(b))
    out = []
    for i in range(n):
        ha, hb = a[i].shape[0], b[i].shape[0]
        h = max(ha, hb)
        canvas = np.full((h, a[i].shape[1] + b[i].shape[1] + gap, 3), 24, dtype=np.uint8)
        canvas[:ha, : a[i].shape[1]] = a[i]
        canvas[:hb, a[i].shape[1] + gap :] = b[i]
        out.append(canvas)
    return out


def landmark_arm(
    npz: Path, model: dict, rest: np.ndarray, max_frames: int | None
) -> tuple[list, dict]:
    """Arm B: the project's original MediaPipe -> rotation-solve retargeting."""
    from seam.avatar.synthesis import forward_kinematics
    from seam.features.signpose import split_landmarks

    with np.load(npz) as z:
        lm = np.asarray(z["landmarks"], dtype=np.float64)
    lm[..., 1] = -lm[..., 1]
    _, pose_mp, lh, rh = split_landmarks(lm)
    if max_frames:
        pose_mp, lh, rh = pose_mp[:max_frames], lh[:max_frames], rh[:max_frames]
    pose_mp, lh, rh = _median_filter(pose_mp), _median_filter(lh), _median_filter(rh)
    body = mediapipe_to_smpl24(pose_mp, flip_y=False, rest=rest)
    body = normalise_to_rest_frame(body, rest=rest)
    frames = synthesise_sequence(body, hand_seq=np.stack([lh, rh], axis=1), rest=rest)
    J = np.asarray([forward_kinematics(f.body_pose, f.global_orient, rest=rest) for f in frames])
    names = [
        "pelvis",
        "left_hip",
        "right_hip",
        "spine1",
        "left_knee",
        "right_knee",
        "spine2",
        "left_ankle",
        "right_ankle",
        "spine3",
        "left_foot",
        "right_foot",
        "neck",
        "left_collar",
        "right_collar",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]
    hip, knee, head_i = 0, names.index("left_knee"), names.index("head")
    torso = float(np.linalg.norm(J[:, head_i] - J[:, 0], axis=1).mean())
    ratio = J[:, knee, 1] - J[:, hip, 1]
    return frames, {
        "frames": len(frames),
        "hip_to_knee_over_torso_mean": float(np.abs(ratio).mean() / max(torso, 1e-9)),
        "hip_to_knee_over_torso_min": float(np.abs(ratio).min() / max(torso, 1e-9)),
        "hip_to_knee_over_torso_max": float(np.abs(ratio).max() / max(torso, 1e-9)),
    }


def smplerx_arm(
    video: Path, model: dict, fps: float | None, batch: int, cache: Path
) -> tuple[list, dict]:
    """Arm A: learned SMPL-X regression, made upright, with collapsed frames repaired."""
    res = sx.run(video, fps=fps, batch=batch, work_dir=cache)
    res = sx.upright(res)
    res, repaired = sx.repair_outliers(res)
    # Per-frame regression shakes and its depth estimate is noise. See `sx.stabilise` for
    # the measurements, including why the model's resting hand is deliberately not added.
    before = sx.jitter(res)
    res = sx.stabilise(res)
    after = sx.jitter(res)
    res = sx.recentre_to_origin(res)
    upright_stats = sx.check_upright(res)
    metrics = {
        "frames": len(res.frames),
        "detection_coverage": round(res.coverage, 4),
        "collapsed_frames_interpolated": repaired,
        "motion": {k: round(v, 4) for k, v in sx.motion_energy(res).items()},
        "upright": {k: round(v, 4) for k, v in upright_stats.items()},
        "stabilised": {
            "frame_to_frame_before": {k: round(v, 4) for k, v in before.items()},
            "frame_to_frame_after": {k: round(v, 4) for k, v in after.items()},
            "hand_mean_added": False,
        },
        "betas": [round(float(b), 4) for b in res.betas],
        "source": "SMPLer-X smpler_x_b32 (third-party, third_party weights not vendored)",
    }
    return res.frames, metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS)
    ap.add_argument("--landmarks", type=Path, default=DEFAULT_LANDMARKS)
    ap.add_argument("--model", type=Path, required=True, help="licence-gated SMPL-X .npz")
    ap.add_argument("--out", type=Path, default=REPO / "artifacts/m7a/demo")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="0 = native video rate, so both arms play at the same speed as the "
        "landmark shard. Resampling one arm and not the other would double one "
        "side's playback rate and rig the comparison.",
    )
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--width", type=int, default=360)
    ap.add_argument("--height", type=int, default=520)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fast", action="store_true", help="fewer clips for a smoke run")
    args = ap.parse_args()
    if args.fast:
        args.limit = min(args.limit, 1)

    model = load_smplx(args.model)
    rest = model_rest_pose(model)
    n_betas = int(np.asarray(model["shapedirs"]).shape[-1])
    args.out.mkdir(parents=True, exist_ok=True)

    videos = sorted(args.videos.glob("*.mp4"))[: args.limit]
    if not videos:
        print(f"no videos in {args.videos}")
        return 2
    entries = []
    out_fps = args.fps if args.fps > 0 else 24.0

    def write_manifest(status: str, done: int = 0, **extra) -> None:
        """Publish progress, not just the finished result.

        The manifest used to be written once at the end, so for the ~15 minutes this takes
        on the GPU the page read "No clips. Run `make demo-avatar`" - which is exactly what
        it says to someone who never ran it. Writing it up front makes "still working" and
        "never run" distinguishable on screen.
        """
        m = {
            "status": status,
            "done": done,
            "total": len(videos),
            "purpose": "M7 arm comparison. Arm A = SMPLer-X regression; Arm B = the "
            "project's own MediaPipe landmark retargeting.",
            "no_ratings": "This produces stimuli and measurements only. No human "
            "preference result is produced, simulated or implied anywhere in this run.",
            "render": {"width": args.width, "height": args.height, "fps": out_fps},
            "model": str(args.model),
            "smplerx": {"checkpoint": "smpler_x_b32", "third_party": True, "vendored": False},
            "clips": list(entries),
            **extra,
        }
        (args.out / "manifest.json").write_text(json.dumps(m, indent=2))

    write_manifest("running", 0)
    for i, vid in enumerate(videos, 1):
        uid = vid.stem
        print(f"[{i}/{len(videos)}] {uid} ...", flush=True)
        write_manifest("running", i - 1, in_progress=uid)
        npz = args.landmarks / f"{uid}.npz"
        entry: dict = {"utterance_id": uid, "video": str(vid)}

        # Arm A
        arm_fps = args.fps if args.fps > 0 else None
        try:
            frames_a, met_a = smplerx_arm(vid, model, arm_fps, args.batch, args.out / f".{uid}")
            betas = np.zeros(n_betas)
            b = np.asarray(met_a["betas"])
            betas[: min(len(b), n_betas)] = b[:n_betas]
            seq_a = smplx_mesh(frames_a, model, betas=betas, use_posedirs=True)
            entry["arm_a_smplerx"] = {**met_a, "mesh": describe_mesh(seq_a)}
            mesh_a = seq_a
            # A real skinned GLB with an animation clip. `mesh.export_glb` writes one
            # static mesh per frame - 40 nodes, no skin, no animation - which opens fine
            # and never moves. Verified against smplx_mesh before it ships.
            glb = export_animated_glb(
                frames_a, model, args.out / f"{uid}_A.glb", betas=betas, fps=out_fps
            )
            entry["arm_a_smplerx"]["glb"] = glb.name
            entry["arm_a_smplerx"]["glb_check"] = verify_glb_animation(
                glb, frames_a, model, frame=0, betas=betas
            )
        except Exception as exc:
            print(f"    arm A FAILED: {type(exc).__name__}: {exc}")
            entry["arm_a_smplerx"] = {"error": f"{type(exc).__name__}: {exc}"}
            mesh_a = None

        # Arm B
        try:
            if not npz.exists():
                raise FileNotFoundError(f"no landmarks shard for {uid}")
            mf = args.max_frames or None
            frames_b, met_b = landmark_arm(npz, model, rest, mf)
            seq_b = smplx_mesh(frames_b, model, betas=None, use_posedirs=True)
            entry["arm_b_landmarks"] = {**met_b, "mesh": describe_mesh(seq_b)}
            mesh_b = seq_b
        except Exception as exc:
            print(f"    arm B FAILED: {type(exc).__name__}: {exc}")
            entry["arm_b_landmarks"] = {"error": f"{type(exc).__name__}: {exc}"}
            mesh_b = None

        if mesh_a is not None and mesh_b is not None:
            n = min(len(mesh_a.vertices), len(mesh_b.vertices))
            # One camera for both arms. Fitting each separately gave 5.92 m vs 4.28 m on
            # clip 1372, so the arms appeared at different scales - and a viewer asked
            # which looks better will answer "the bigger one". Same distance, same
            # lighting, same everything except the pose source.
            cam = _fit_camera(
                list(mesh_a.vertices) + list(mesh_b.vertices), args.width, args.height
            )
            entry.setdefault("camera", {})["shared_distance_m"] = round(cam.distance_m, 3)
            for key in ("arm_a_smplerx", "arm_b_landmarks"):
                entry[key]["camera_distance_m"] = round(cam.distance_m, 3)
            imgs_a = [
                render_frame(v, mesh_a.faces, width=args.width, height=args.height, camera=cam)
                for v in mesh_a.vertices
            ]
            imgs_b = [
                render_frame(v, mesh_b.faces, width=args.width, height=args.height, camera=cam)
                for v in mesh_b.vertices
            ]
            _write_mp4(imgs_a, args.out / f"{uid}_A.mp4", out_fps)
            _write_mp4(imgs_b, args.out / f"{uid}_B.mp4", out_fps)
            _write_mp4(_side_by_side(imgs_a[:n], imgs_b[:n]), args.out / f"{uid}_AB.mp4", out_fps)
            entry["side_by_side"] = f"{uid}_AB.mp4"
        entries.append(entry)
        # One arm surviving is still worth showing; a comparison is not, so the manifest
        # records the failure rather than quietly shipping a single-panel "comparison".
        if mesh_a is None or mesh_b is None:
            solo = mesh_a if mesh_a is not None else mesh_b
            if solo is not None:
                tag = "A" if mesh_a is not None else "B"
                cam = _fit_camera(list(solo.vertices), args.width, args.height)
                imgs = [
                    render_frame(v, solo.faces, width=args.width, height=args.height, camera=cam)
                    for v in solo.vertices
                ]
                _write_mp4(imgs, args.out / f"{uid}_{tag}.mp4", out_fps)
            entry["comparison"] = "unavailable: one arm failed; see per-arm error"
            print(f"    solo {uid} written; no side-by-side (one arm failed)")
        else:
            entry["comparison"] = "side_by_side"
            print(f"    wrote {uid}_A.mp4 / {uid}_B.mp4 / {uid}_AB.mp4")

    manifest = {
        "status": "done",
        "done": len(entries),
        "total": len(videos),
        "purpose": "M7 arm comparison. Arm A = SMPLer-X regression; Arm B = the project's "
        "own MediaPipe landmark retargeting. Both are real existing code.",
        "baseline_rationale": "Arm B is this project's previous front end, so any "
        "difference is attributable to the perception swap rather "
        "than to a baseline built to lose. No synthetic stub.",
        "no_ratings": "This produces stimuli and measurements only. No human preference "
        "result is produced, simulated or implied anywhere in this run.",
        "render": {"width": args.width, "height": args.height, "fps": out_fps},
        "temporal_parity": "Both arms are written at the same frame rate and the "
        "side-by-side truncates to the shorter one. Resampling only one "
        "arm would silently double the other's playback speed.",
        "model": str(args.model),
        "smplerx": {"checkpoint": "smpler_x_b32", "third_party": True, "vendored": False},
        "clips": entries,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {args.out / 'manifest.json'} ({len(entries)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
