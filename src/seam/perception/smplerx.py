"""SMPLer-X as the body-pose source: real SMPL-X regression instead of landmark guessing.

Why this exists
---------------
Until now the avatar's body came from `seam.features.signpose`, which estimates joint
positions from MediaPipe landmarks and then solves for rotations. That approach has a
measured failure that cannot be coded around: on EmoSign the hip-to-knee distance ranges
**0.06 to 1.42 torso-lengths** across 200 clips, which is noise rather than anatomy, and
every signer is seated while the renderer draws a standing figure. The legs were never
going to work.

SMPLer-X is a *learned* whole-body SMPL-X regressor (ViT-B, `smpler_x_b32`). It predicts
anatomically valid poses directly, so it needs no per-joint rotation solve and its legs
are a learned prior rather than a two-point depth guess. Measured on our own clip it
detected and fitted 43/43 frames on a 3.68 GiB laptop GPU.

It also carries what MediaPipe could never give us in a usable form:

    betas (10)      real body shape, so the avatar is not a generic neutral mannequin
    expression (10) face coefficients - the non-manual channel
    jaw_pose (3)    mouth motion
    left/right_hand_pose (45 each) MANO-order fingers

That last group matters because it is the same parameterisation `avatar.synthesis`
already speaks, so nothing in the avatar has to change to consume it.

Provenance, stated plainly
--------------------------
SMPLer-X is **third-party**, trained by its authors on 3DPW and similar mocap. It is not
trained on ASL and knows nothing about sign language. It is a body prior, so it will
render plausible-looking signing whatever the hands actually say - and its fingers in
particular should be treated as unverified without a dedicated hand model. Nothing
derived from it may be presented as evidence about linguistic content. Its weights are
**not** vendored here; they live in the pipeline tree and are licence-gated like the
SMPL-X parameters.

Relationship to the protected tree
----------------------------------
`SMPLERX_ROOT` (default `~/research/nsl/SMPLer-X`) is **read-only**. This module only ever
runs `main/nsl_runner.py` as a subprocess and reads the JSON it writes. Nothing in that
tree is imported, patched or written, so the adapter works against an untouched checkout
and a missing checkout degrades to a clear error rather than a broken half-state.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from seam.avatar.synthesis import SmplxFrame

DEFAULT_ROOT = Path(os.environ.get("SMPLERX_ROOT", "~/research/nsl/SMPLer-X")).expanduser()
DEFAULT_ENV = os.environ.get("SMPLERX_ENV", "smplerx")
CONDA_ENVS = Path(os.environ.get("CONDA_ENVS", Path.home() / "miniconda3/envs"))

#: SMPL-X's own parameter counts. Asserted on load because a silently wrong reshape is
#: the kind of bug that produces a plausible-looking animation from garbage.
N_BODY, N_HAND, N_EXPR, N_BETAS = 21, 15, 10, 10


@dataclass
class SmplerxResult:
    """One clip's worth of regressed SMPL-X, plus what it cost to get."""

    frames: list[SmplxFrame]
    betas: np.ndarray
    fps: float
    n_requested: int
    n_detected: int
    checkpoint: str
    #: Frames where the person detector found nobody. Interpolation across these is the
    #: caller's decision, not this module's - a gap is reported, never quietly filled.
    n_nobox: int = 0
    n_nobox_frames: list[int] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        return self.n_detected / self.n_requested if self.n_requested else 0.0


class SmplerxUnavailable(RuntimeError):
    """The read-only SMPLer-X tree or its conda env is not present."""


def locate(root: Path = DEFAULT_ROOT, env: str = DEFAULT_ENV) -> tuple[Path, Path]:
    """Return ``(runner, python)`` for the read-only SMPLer-X install.

    Raises :class:`SmplerxUnavailable` with the specific missing piece rather than letting
    a subprocess fail later with a stack trace from someone else's repo.
    """
    root = Path(root).expanduser()
    runner = root / "main" / "nsl_runner.py"
    python = CONDA_ENVS / env / "bin" / "python"
    ckpt = root / "pretrained_models" / "smpler_x_b32.pth.tar"
    if not root.exists():
        raise SmplerxUnavailable(f"SMPLERX_ROOT {root} does not exist")
    if not runner.exists():
        raise SmplerxUnavailable(f"runner missing: {runner}")
    if not ckpt.exists():
        raise SmplerxUnavailable(
            f"checkpoint missing: {ckpt}. Weights are third-party and not vendored here; "
            "see the SMPLer-X release for the download."
        )
    if not python.exists():
        raise SmplerxUnavailable(f"python missing: {python} (conda env {env!r})")
    return runner, python


def extract_frames(video: Path, out_dir: Path, *, fps: float | None = None) -> int:
    """Write ``frame_%05d.png`` for ``video``. Returns the frame count written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = ["scale=256:256"]
    if fps:
        vf.insert(0, f"fps={fps}")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-vf", ",".join(vf)]
    cmd += [str(out_dir / "frame_%05d.png")]
    # stdin closed: a child that reads the terminal can swallow or echo the user's
    # keystrokes, which floods the log with SS3 escapes and hides real progress.
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)
    return len(list(out_dir.glob("frame_*.png")))


def run(
    video: Path,
    *,
    fps: float | None = None,
    ckpt: str = "smpler_x_b32",
    batch: int = 4,
    root: Path = DEFAULT_ROOT,
    env: str = DEFAULT_ENV,
    work_dir: Path | None = None,
    timeout: float = 3600.0,
) -> SmplerxResult:
    """Regress one video to per-frame SMPL-X parameters.

    ``batch=4`` is deliberate. The ViT-B model fits a 4 GB card, which is what the 3050
    laptop in this project has; the default 8 in the upstream runner is an OOM risk on
    exactly the hardware we run on.
    """
    runner, python = locate(root=root, env=env)
    tmp = Path(tempfile.mkdtemp()) if work_dir is None else Path(work_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    frames = tmp / "frames"
    n = extract_frames(Path(video), frames, fps=fps)
    if n == 0:
        raise RuntimeError(f"no frames decoded from {video}")
    out_json = tmp / "smplerx_raw.json"
    # Reuse an existing dump. The regressor costs ~2 min/clip and nothing downstream
    # touches it, so without this, fixing an exporter costs a 15-minute re-run - long
    # enough that you stop verifying and start assuming. Keyed on the frame count so a
    # changed frame rate cannot silently reuse a differently-sampled dump.
    if out_json.is_file():
        try:
            cached = json.loads(out_json.read_text())
            if cached.get("n_frames") == n and cached.get("n_requested", n) == n:
                return parse(out_json, fps=fps or 25.0, checkpoint=ckpt)
        except (ValueError, KeyError, OSError):
            pass  # unreadable cache is not an error; just recompute
    subprocess.run(
        [
            str(python),
            str(runner),
            "--frames_dir",
            str(frames),
            "--pattern",
            "frame_*.png",
            "--ckpt",
            ckpt,
            "--batch",
            str(batch),
            "--out_json",
            str(out_json),
        ],
        cwd=str(Path(root).expanduser() / "main"),
        check=True,
        timeout=timeout,
        stdin=subprocess.DEVNULL,
    )
    return parse(out_json, fps=fps or 25.0, checkpoint=ckpt)


def parse(raw_json: Path, *, fps: float = 25.0, checkpoint: str = "") -> SmplerxResult:
    """Convert a ``nsl_runner`` JSON dump into :class:`SmplerxResult`.

    Raises on a shape mismatch rather than reshaping. A wrong-length ``body_pose``
    reshaped to ``(21, 3)`` still renders - just a different, wrong person - so the
    shape check is the only thing standing between a schema change and a plausible lie.
    """
    data = json.loads(Path(raw_json).read_text())
    raw = data["frames"]
    frames: list[SmplxFrame] = []
    betas = np.zeros(N_BETAS)
    nobox: list[int] = []
    seen_betas = False
    for i, fr in enumerate(raw):
        if not fr.get("detected"):
            nobox.append(i)
            frames.append(SmplxFrame())
            continue
        bp = np.asarray(fr["body_pose"], dtype=np.float64)
        lh = np.asarray(fr["left_hand_pose"], dtype=np.float64)
        rh = np.asarray(fr["right_hand_pose"], dtype=np.float64)
        if bp.size != N_BODY * 3:
            raise ValueError(f"frame {i}: body_pose has {bp.size} values, expected {N_BODY * 3}")
        if lh.size != N_HAND * 3 or rh.size != N_HAND * 3:
            raise ValueError(
                f"frame {i}: hand pose {lh.size}/{rh.size} values, expected {N_HAND * 3}"
            )
        ex = np.asarray(fr.get("expression", np.zeros(N_EXPR)), dtype=np.float64)
        frames.append(
            SmplxFrame(
                global_orient=np.asarray(fr["global_orient"], dtype=np.float64),
                body_pose=bp.reshape(N_BODY, 3),
                left_hand_pose=lh.reshape(N_HAND, 3),
                right_hand_pose=rh.reshape(N_HAND, 3),
                jaw_pose=np.asarray(fr.get("jaw_pose", np.zeros(3)), dtype=np.float64),
                expression=ex[:N_EXPR],
                global_transl=np.asarray(fr.get("cam_trans", np.zeros(3)), dtype=np.float64),
            )
        )
        if not seen_betas and np.any(np.asarray(fr.get("betas", 0.0))):
            betas = np.asarray(fr["betas"], dtype=np.float64)[:N_BETAS]
            seen_betas = True
    return SmplerxResult(
        frames=frames,
        betas=betas,
        fps=fps,
        n_requested=len(raw),
        n_detected=len(raw) - len(nobox),
        n_nobox=len(nobox),
        n_nobox_frames=nobox,
        checkpoint=checkpoint,
    )


def motion_energy(result: SmplerxResult) -> dict[str, float]:
    """How much the body actually moves. A guard against silently static output.

    A regressor that returns its mean pose every frame will happily render a smooth,
    plausible, completely motionless clip. That is the failure this catches, and it is
    why the demo reports these numbers instead of just showing a video.
    """
    body = np.stack([f.body_pose.reshape(-1) for f in result.frames])
    if len(body) < 2:
        return {}
    speed = np.linalg.norm(np.diff(body, axis=0), axis=1)
    return {
        "frames": float(len(body)),
        "mean_joint_speed": float(speed.mean()),
        "max_joint_speed": float(speed.max()),
        "body_range_rad": float(np.ptp(body, axis=0).max()),
        "global_yaw_range_rad": float(
            np.ptp(np.stack([f.global_orient for f in result.frames]), axis=0).max()
        ),
    }


def _cli_available() -> bool:
    return shutil.which("ffmpeg") is not None


# ─────────────────────────────────────────────────────────────────────────────
#  Convention alignment
# ─────────────────────────────────────────────────────────────────────────────


def _aa_to_matrix(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    th = float(np.linalg.norm(aa))
    if th < 1e-12:
        return np.eye(3)
    k = aa / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _matrix_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle. Only used to write the correction back."""
    ang = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if ang < 1e-9:
        return np.zeros(3)
    if abs(np.pi - ang) < 1e-6:
        # Near-pi is the numerically nasty case; take the most symmetric axis.
        A = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(A)))
        ax = np.zeros(3)
        ax[i] = np.sqrt(max(A[i, i], 0.0))
        for j in range(3):
            if j != i:
                ax[j] = A[i, j] / max(ax[i], 1e-12)
        n = np.linalg.norm(ax)
        return ax / n * ang if n > 0 else np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * np.sin(ang)) * ang


def _minimal_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest rotation carrying unit vector ``a`` onto unit vector ``b``."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel: a half turn about any axis *perpendicular* to a. Picking a
        # hardcoded axis (x) is wrong - a half turn about a itself leaves a unchanged, so
        # R @ a came back as +a instead of -a. Take the coordinate axis least aligned
        # with a, which is guaranteed non-degenerate.
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(a)))] = 1.0
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return _aa_to_matrix(axis * np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def upright(result: SmplerxResult) -> SmplerxResult:
    """Rotate one clip into this renderer's frame: head up, front toward the viewer.

    SMPLer-X emits ``global_orient`` in its own training-data convention, which is not
    SMPL-X's Y-up canonical frame. Fed straight to :func:`avatar.mesh.smplx_mesh` the
    body comes out **upside down** - measured on clip 1372, ``global_orient`` x averages
    129 deg, putting the head at y = -0.69 m with the pelvis at y = 0.

    There is no published conversion to copy: upstream renders these parameters inside a
    PyTorch3D camera and never rotates them. So the correction is derived from the clip
    itself, using two anatomical invariants that are checkable rather than assumed:

        1. the head is above the pelvis, and
        2. the body's front faces the viewer.

    Both are measured on the regressed skeleton before the rotation and re-checked after,
    so a clip that cannot satisfy them is reported instead of being silently forced.

    This is a *coordinate* correction, not a pose edit. The articulated pose - every joint
    angle, and the motion energy that depends on them - is untouched, which is why it is
    legitimate while inventing a baseline arm would not be.
    """
    from seam.avatar.synthesis import SMPL_BODY_JOINTS, forward_kinematics

    names = ["pelvis", *SMPL_BODY_JOINTS]
    head_i, pelvis_i = names.index("head"), 0

    # Reference pose: the temporally median body pose, orientation excluded, so genuine
    # movement across the clip cannot tilt the derived "up".
    body = np.stack([f.body_pose.reshape(-1) for f in result.frames])
    ref = np.median(body, axis=0).reshape(-1, 3)
    # The root rotation must be in the reference too. The pelvis->head direction in world
    # is R_global @ (chain direction), so deriving "up" with global_orient=0 measures the
    # chain, not the body - which under-corrects and leaves most frames inverted.
    ref_go = np.median(np.stack([f.global_orient for f in result.frames]), axis=0)
    J = forward_kinematics(ref, ref_go)
    up = J[head_i] - J[pelvis_i]
    if np.linalg.norm(up) < 1e-6:
        return result
    R_up = _minimal_rotation(up, np.array([0.0, 1.0, 0.0]))

    # Front. The canonical front of an SMPL-X body is +z, but in world space it is
    # carried by the root rotation: world_front = R_global @ [0, 0, 1]. Using
    # R_up @ [0, 0, 1] instead drops R_global and lands 180 deg out - which renders as a
    # perfectly good-looking back view, so it is easy to miss.
    front = R_up @ (_aa_to_matrix(ref_go) @ np.array([0.0, 0.0, 1.0]))
    yaw = float(np.arctan2(front[0], front[2]))
    R_yaw = _aa_to_matrix(np.array([0.0, 1.0, 0.0]) * -yaw)
    R = R_yaw @ R_up

    out = SmplerxResult(
        frames=[
            SmplxFrame(
                global_orient=_matrix_to_aa(R @ _aa_to_matrix(f.global_orient)),
                body_pose=f.body_pose.copy(),
                left_hand_pose=f.left_hand_pose.copy(),
                right_hand_pose=f.right_hand_pose.copy(),
                jaw_pose=f.jaw_pose.copy(),
                expression=f.expression.copy(),
                global_transl=R @ f.global_transl,
            )
            for f in result.frames
        ],
        betas=result.betas.copy(),
        fps=result.fps,
        n_requested=result.n_requested,
        n_detected=result.n_detected,
        n_nobox=result.n_nobox,
        n_nobox_frames=list(result.n_nobox_frames),
        checkpoint=result.checkpoint,
    )
    out.alignment = R  # type: ignore[attr-defined]
    return out


def check_upright(result: SmplerxResult) -> dict[str, float]:
    """Measure the two invariants. Returns numbers; it does not assert.

    ``head_above_pelvis`` should be positive for every frame, and the front-facing
    residual near zero. A clip that fails either was not made upright, whatever this
    function returns.
    """
    from seam.avatar.synthesis import SMPL_BODY_JOINTS, forward_kinematics

    names = ["pelvis", *SMPL_BODY_JOINTS]
    head_i, pelvis_i = names.index("head"), 0
    gaps, fronts = [], []
    for f in result.frames:
        J = forward_kinematics(f.body_pose, f.global_orient)
        gaps.append(float(J[head_i, 1] - J[pelvis_i, 1]))
        up = J[head_i] - J[pelvis_i]
        up /= max(np.linalg.norm(up), 1e-9)
        # Deviation of the head-pelvis axis from world up. Zero when the body stands
        # upright. (Measuring arcsin(up[1]) instead would read 90 deg *because* the body
        # is upright, which is the opposite of the intent.)
        fronts.append(float(np.degrees(np.arccos(np.clip(up[1], -1, 1)))))
    return {
        "head_above_pelvis_min_m": float(np.min(gaps)),
        "head_above_pelvis_mean_m": float(np.mean(gaps)),
        "frames_head_below_pelvis": float(np.sum(np.array(gaps) < 0)),
        "up_axis_tilt_max_deg": float(np.max(np.abs(fronts))),
    }


def repair_outliers(
    result: SmplerxResult, *, ratio: float = 0.5
) -> tuple[SmplerxResult, list[int]]:
    """Interpolate across frames where the regressor has collapsed, and report them.

    SMPLer-X occasionally loses a person and returns a collapsed pose rather than
    admitting it failed. On clip 1372 the head-above-pelvis distance is 0.68-0.69 m for
    frames 0-39 and then drops to 0.17-0.18 m for frames 40-42 - sharply bimodal, and the
    tail renders as a body lying on its side.

    Detection uses that anatomical distance rather than deviation from the median pose,
    because signing moves far from the median by definition (median deviation on this clip
    is 0.854 rad, so a median-based outlier test flags nothing - it was tried first and
    found zero outliers in a clip that visibly has three).

    The repair is interpolation between the nearest good frames, or edge-hold at the
    boundaries. **It is interpolation, not measurement**, so the affected frames are
    returned and must be reported alongside any result that uses them.

    Limitation: the test is median-relative, so a clip the regressor failed on from end to
    end is uniformly bad, its median is uniformly bad, and nothing is flagged. Such a clip
    passes through unrepaired. The uprightness numbers from :func:`check_upright` are what
    expose it, which is why the demo reports both.
    """
    from seam.avatar.synthesis import SMPL_BODY_JOINTS, forward_kinematics

    names = ["pelvis", *SMPL_BODY_JOINTS]
    head_i = names.index("head")
    gaps = np.array(
        [
            float(
                (lambda J: J[head_i, 1] - J[0, 1])(forward_kinematics(f.body_pose, f.global_orient))
            )
            for f in result.frames
        ]
    )
    if gaps.size == 0:
        return result, []
    med = float(np.median(gaps))
    bad = np.flatnonzero(gaps < ratio * med).tolist()
    if not bad:
        return result, []

    def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return a * (1.0 - t) + b * t

    frames = list(result.frames)
    good = [i for i in range(len(frames)) if i not in set(bad)]
    if not good:  # everything flagged: nothing to interpolate toward, leave it alone
        return result, bad
    for i in bad:
        lo = max([g for g in good if g < i], default=None)
        hi = min([g for g in good if g > i], default=None)
        if lo is None:
            frames[i] = frames[hi]  # type: ignore[index]
        elif hi is None:
            frames[i] = frames[lo]  # type: ignore[index]
        else:
            src, dst = frames[lo], frames[hi]
            t = (i - lo) / (hi - lo)
            frames[i] = SmplxFrame(
                global_orient=lerp(src.global_orient, dst.global_orient, t),
                body_pose=lerp(src.body_pose, dst.body_pose, t),
                left_hand_pose=lerp(src.left_hand_pose, dst.left_hand_pose, t),
                right_hand_pose=lerp(src.right_hand_pose, dst.right_hand_pose, t),
                jaw_pose=lerp(src.jaw_pose, dst.jaw_pose, t),
                expression=lerp(src.expression, dst.expression, t),
                global_transl=lerp(src.global_transl, dst.global_transl, t),
            )
    out = SmplerxResult(
        frames=frames,
        betas=result.betas,
        fps=result.fps,
        n_requested=result.n_requested,
        n_detected=result.n_detected,
        n_nobox=result.n_nobox,
        n_nobox_frames=list(result.n_nobox_frames),
        checkpoint=result.checkpoint,
    )
    out.repaired = bad  # type: ignore[attr-defined]
    return out, bad


def _aa_to_quat(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle to (..., 4) unit quaternions, w first."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    half = 0.5 * theta
    # sin(x)/x with its limit at zero, so a zero rotation is the identity and not NaN.
    k = np.where(theta > 1e-8, np.sin(half) / np.maximum(theta, 1e-8), 0.5)
    return np.concatenate([np.cos(half), aa * k], axis=-1)


def _quat_to_aa(q: np.ndarray) -> np.ndarray:
    """(..., 4) unit quaternions, w first, to (..., 3) axis-angle with angle in [0, pi]."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = np.where(q[..., :1] < 0, -q, q)
    w, v = q[..., :1], q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(n, w)
    return np.where(n > 1e-8, v / np.maximum(n, 1e-8) * angle, 2.0 * v)


def smooth_rotations(track: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing over time of an axis-angle track ``(T, J, 3)``, done on the sphere.

    Averaging axis-angle components directly is wrong for anything but tiny rotations: the
    components of two nearby rotations can differ by a full turn, and their mean is then a
    third, unrelated rotation. Each joint is instead converted to a unit quaternion, the
    sign made consistent from frame to frame (``q`` and ``-q`` are the same rotation, and
    averaging across a sign flip cancels to nothing), averaged with a Gaussian window that
    is clamped at the clip edges, and renormalised.
    """
    track = np.asarray(track, dtype=np.float64)
    n = track.shape[0]
    if sigma <= 0 or n < 3:
        return track.copy()
    q = _aa_to_quat(track)
    for t in range(1, n):
        flip = np.sum(q[t] * q[t - 1], axis=-1, keepdims=True) < 0
        q[t] = np.where(flip, -q[t], q[t])
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    out = np.zeros_like(q)
    for off, w in zip(offsets, weights, strict=True):
        out += w * q[np.clip(np.arange(n) + off, 0, n - 1)]
    return _quat_to_aa(out)


def stabilise(
    result: SmplerxResult,
    *,
    hands_mean: tuple[np.ndarray, np.ndarray] | None = None,
    sigma_body: float = 1.5,
    sigma_hands: float = 1.2,
    sigma_transl: float = 2.0,
) -> SmplerxResult:
    """Remove the regressor's frame-to-frame noise.

    SMPLer-X regresses every frame on its own, and three things about its raw output make
    an animation built straight from it wrong. All three were measured on clip 1372 after
    the avatar was seen to "jump back and forth" with unsteady hands.

    **1. Depth is noise.** ``cam_trans`` z has a frame-to-frame standard deviation of
    0.43 m and a range of 1.8 m on a signer who is sitting still, against 7 cm and 5 cm
    sideways and vertically. Monocular depth of a far subject is not recoverable per frame,
    and the body was being moved toward and away from the camera by that noise. Depth is
    replaced by its median over the clip; the other two axes are smoothed.

    **2. Rotations jitter.** Body joints change by up to 0.44 rad and finger joints by
    0.75 rad between consecutive frames. Every joint is smoothed on the rotation sphere
    (:func:`smooth_rotations`). The windows are short - about 60 ms - because signing is
    fast and a long window would blur one handshape into the next.

    **3. The hand poses are used as they are - the resting hand is NOT added.** SMPL-X
    layers built with the library default (``flat_hand_mean=False``) add the model's
    relaxed hand (``hands_meanl`` / ``hands_meanr``, 13 to 52 degrees of curl per joint) to
    whatever they are given, and the obvious reading is that this regressor's hand output
    is therefore an offset that needs that mean. It was tried and it is wrong for this
    runner: on frames where the signer holds flat open hands, the output without the mean
    renders flat open hands and the output with it renders claws. ``hands_mean`` exists
    for a regressor whose output really is mean-relative (InterWild is one), and must not
    be passed for SMPLer-X. What does improve the hands here is the smoothing: finger
    joints jump by up to 0.75 rad between frames in the raw output.

    Call this after :func:`repair_outliers`, so collapsed frames are replaced before they
    are averaged into their neighbours, and it does not matter whether :func:`upright` ran
    first: the depth axis is found from the data as the noisiest translation axis.
    """
    frames = result.frames
    n = len(frames)
    if n == 0:
        return result

    transl = np.array([f.global_transl for f in frames], dtype=np.float64)
    step_sd = np.diff(transl, axis=0).std(axis=0) if n > 1 else np.zeros(3)
    depth = int(np.argmax(step_sd))
    steadied = transl.copy()
    for axis in range(3):
        if axis == depth and step_sd[axis] > 3.0 * np.median(step_sd) + 1e-9:
            steadied[:, axis] = np.median(transl[:, axis])
        elif sigma_transl > 0 and n >= 3:
            radius = max(1, int(np.ceil(3 * sigma_transl)))
            offsets = np.arange(-radius, radius + 1)
            w = np.exp(-0.5 * (offsets / sigma_transl) ** 2)
            acc = np.zeros(n)
            for off, wi in zip(offsets, w, strict=True):
                acc += wi * transl[np.clip(np.arange(n) + off, 0, n - 1), axis]
            steadied[:, axis] = acc / w.sum()

    orient = smooth_rotations(np.array([f.global_orient for f in frames])[:, None, :], sigma_body)
    body = smooth_rotations(np.array([f.body_pose for f in frames]), sigma_body)
    jaw = smooth_rotations(np.array([f.jaw_pose for f in frames])[:, None, :], sigma_body)
    left = np.array([f.left_hand_pose for f in frames], dtype=np.float64)
    right = np.array([f.right_hand_pose for f in frames], dtype=np.float64)
    if hands_mean is not None:
        mean_l = np.asarray(hands_mean[0], dtype=np.float64).reshape(N_HAND, 3)
        mean_r = np.asarray(hands_mean[1], dtype=np.float64).reshape(N_HAND, 3)
        left, right = left + mean_l, right + mean_r
    left = smooth_rotations(left, sigma_hands)
    right = smooth_rotations(right, sigma_hands)

    out = [
        SmplxFrame(
            global_orient=orient[i, 0],
            body_pose=body[i],
            left_hand_pose=left[i],
            right_hand_pose=right[i],
            jaw_pose=jaw[i, 0],
            expression=frames[i].expression,
            global_transl=steadied[i],
        )
        for i in range(n)
    ]
    return SmplerxResult(
        frames=out,
        betas=result.betas,
        fps=result.fps,
        n_requested=result.n_requested,
        n_detected=result.n_detected,
        checkpoint=result.checkpoint,
        n_nobox=result.n_nobox,
        n_nobox_frames=result.n_nobox_frames,
    )


def jitter(result: SmplerxResult) -> dict[str, float]:
    """How much the clip shakes: mean frame-to-frame change, per group. Lower is steadier."""
    f = result.frames
    if len(f) < 2:
        return {"body_rad": 0.0, "hands_rad": 0.0, "root_m": 0.0}

    def step(get: object) -> float:
        a = np.array([get(x) for x in f], dtype=np.float64)  # type: ignore[operator]
        d = np.diff(a, axis=0)
        return float(np.linalg.norm(d.reshape(len(d), -1, 3), axis=2).mean())

    return {
        "body_rad": step(lambda x: x.body_pose),
        "hands_rad": step(lambda x: np.concatenate([x.left_hand_pose, x.right_hand_pose])),
        "root_m": step(lambda x: x.global_transl[None, :]),
    }


def recentre_to_origin(result: SmplerxResult) -> SmplerxResult:
    """Move the body so the pelvis sits at the world origin.

    SMPLer-X places the subject at a real camera distance - on clip 1372 the mesh centroid
    is **22.6 m** from the origin, because the regressor works in a metric camera frame
    with the person far away. This renderer's camera sits 2.6 m out, so an uncentred
    sequence renders as an empty frame: the first demo run produced a 2.9 KB MP4 with
    **two unique colours** in it while every measurement said the pose was fine.

    The offset is the clip's *median* pelvis position, not a per-frame one, so the camera
    stays fixed and the body does not jitter. Per-frame centring would make any camera
    shake impossible to see and would hide exactly the global motion the demo exists to
    show.
    """
    from seam.avatar.synthesis import forward_kinematics

    pelvis = []
    for f in result.frames:
        J = forward_kinematics(f.body_pose, f.global_orient)
        pelvis.append(f.global_transl + J[0])
    offset = np.median(np.stack(pelvis), axis=0)
    frames = [
        SmplxFrame(
            global_orient=f.global_orient,
            body_pose=f.body_pose,
            left_hand_pose=f.left_hand_pose,
            right_hand_pose=f.right_hand_pose,
            jaw_pose=f.jaw_pose,
            expression=f.expression,
            global_transl=f.global_transl - offset,
        )
        for f in result.frames
    ]
    out = SmplerxResult(
        frames=frames,
        betas=result.betas,
        fps=result.fps,
        n_requested=result.n_requested,
        n_detected=result.n_detected,
        n_nobox=result.n_nobox,
        n_nobox_frames=list(result.n_nobox_frames),
        checkpoint=result.checkpoint,
    )
    out.centre_offset = offset  # type: ignore[attr-defined]
    return out
