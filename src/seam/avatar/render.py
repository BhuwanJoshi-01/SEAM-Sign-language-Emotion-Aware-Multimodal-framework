"""Render a mesh sequence to image frames, for the M7 preference study.

**Why a software rasteriser rather than a GPU renderer.** The stimulus videos are
evidence. A renderer that needs a display, a GPU driver, or an OpenGL context produces
different output on different machines, silently, and the difference would be visible to
a rater. `trimesh.Scene.save_image` needs pyglet and a GL context; a headless CI box or a
locked-down Windows session does not have one. This module rasterises in numpy instead:
pure CPU, no display, no GPU, no driver. The same input gives byte-identical output on
any machine, which is the property a blinded study actually requires.

**What it draws.** Flat-shaded triangles with per-face Lambert lighting from a fixed
directional light, painter's-algorithm depth sorting, and a z-buffer so a nearer triangle
always wins. There is no texture, no shadow, no anti-aliasing. This is deliberate: the
study asks whether one body *moves* more naturally than another, and every rendering
flourish that differs between the two conditions would be a confound.

**Determinism is asserted, not assumed.** `render_sequence` is a pure function of its
inputs, and `tests/test_render.py` renders the same sequence twice and requires the
frames to be identical. A renderer with hidden state - a random seed, a frame counter, a
wall-clock timestamp - would make the two conditions non-comparable.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Camera: a fixed three-quarter view. Front-on hides depth, which is where a hand's
#: path shape shows; a pure side view hides the left/right hand separation, which is most
#: of what a signer reads. Three-quarter is the only view where both are legible.
#:
#: **The yaw is 210, not 30, and it matters.** At yaw 30 the camera looks at the body's
#: *back*: shoulder blades, the back of the head, heel-first feet. Verified by rendering
#: the rest pose at yaw 0/90/180/270 (`scripts/render_rest_yaws.py`) - yaw 0 shows a bare
#: back, yaw 180 shows a face with a chest and kneecaps. Measured independently
#: (`scripts/check_facing.py`): the nose is the most +z vertex on the whole body
#: (z = +0.142), so the body faces +z. A study whose raters judge facial and manual
#: expressiveness cannot be run against the back of a head. This is the old 30 turned
#: half a revolution, so the three-quarter offset and the depth cue are unchanged.
DEFAULT_YAW_DEG = 210.0
DEFAULT_PITCH_DEG = 8.0

#: Light direction in world space, normalised in `_shade`. A single fixed light, so
#: nothing about the lighting can differ between conditions.
LIGHT_DIR = np.array([-0.4, 0.8, 0.6], dtype=np.float64)

#: Background. Mid-grey, not black or white: against black a dark body silhouette loses
#: its limbs, and against white a pale body loses them. Grey is the only value where a
#: neutral body keeps a legible edge.
BACKGROUND = np.array([0.32, 0.34, 0.37], dtype=np.float64)


@dataclass(frozen=True)
class Camera:
    """A fixed camera looking at the origin."""

    yaw_deg: float = DEFAULT_YAW_DEG
    pitch_deg: float = DEFAULT_PITCH_DEG
    distance_m: float = 2.6
    fov_deg: float = 42.0

    def view_matrix(self) -> np.ndarray:
        """World -> camera rotation, (3, 3)."""
        y = np.radians(self.yaw_deg)
        p = np.radians(self.pitch_deg)
        ry = np.array([[np.cos(y), 0.0, np.sin(y)], [0.0, 1.0, 0.0], [-np.sin(y), 0.0, np.cos(y)]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(p), -np.sin(p)], [0.0, np.sin(p), np.cos(p)]])
        return rx @ ry


def _project(verts: np.ndarray, cam: Camera, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """World vertices -> screen (x, y) in pixels plus camera-space depth.

    Returns `(screen_xy, depth)` where depth is the camera-space z used for the z-buffer.
    Depth is returned rather than recomputed per triangle so the two cannot drift.
    """
    v = np.asarray(verts, dtype=np.float64)
    centred = v - np.array([0.0, -0.55, 0.0])  # aim at the torso, not the feet
    cam_space = centred @ cam.view_matrix().T
    cam_space[:, 2] += cam.distance_m  # push the body in front of the camera

    # Perspective divide. A vertex at or behind the camera is clamped rather than
    # skipped: skipping would drop whole triangles and make a limb vanish for a frame,
    # which a rater would read as a tracking fault that is not there.
    z = np.maximum(cam_space[:, 2], 1e-3)
    f = 1.0 / np.tan(np.radians(cam.fov_deg) / 2.0)
    aspect = w / float(h)
    ndc_x = (cam_space[:, 0] * f / aspect) / z
    ndc_y = (cam_space[:, 1] * f) / z

    # The camera looks down +z here with +y up, so +x in camera space is the viewer's
    # LEFT, not right - screen x has to be flipped. Without the flip every render was a
    # mirror image: a signer raising their right hand was drawn raising their left. It
    # went unnoticed because a mirrored body still faces the camera, stands upright and
    # moves plausibly, and the only check made was "is it facing us". Sign language has
    # handedness, so a mirrored stimulus is a different utterance.
    sx = (1.0 - (ndc_x * 0.5 + 0.5)) * (w - 1)
    sy = (1.0 - (ndc_y * 0.5 + 0.5)) * (h - 1)
    return np.stack([sx, sy], axis=1), z


def _shade(faces: np.ndarray, verts: np.ndarray) -> np.ndarray:
    """Per-face Lambert intensity from one fixed light, shape (F,).

    A face is lit by the absolute dot of its normal with the light, so a surface facing
    away is dark rather than black - a fully unlit limb edge would merge with the
    background and read as a missing limb.
    """
    tri = verts[faces]  # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(ln, 1e-12)
    light = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)
    lam = np.abs(n @ light)
    return 0.35 + 0.65 * lam  # ambient floor so no face is pure black


def _rasterise_triangle(
    img: np.ndarray,
    zbuf: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    z0: float,
    z1: float,
    z2: float,
    colour: np.ndarray,
) -> None:
    """Fill one triangle with a z-test, in place.

    Barycentric bounding-box rasterisation. Used rather than a scanline fill because it
    is branch-free over the box and therefore gives identical results regardless of
    triangle winding order - a winding-order bug would show as a random hole in the body.
    """
    h, w = zbuf.shape
    min_x = max(int(np.floor(min(p0[0], p1[0], p2[0]))), 0)
    max_x = min(int(np.ceil(max(p0[0], p1[0], p2[0]))), w - 1)
    min_y = max(int(np.floor(min(p0[1], p1[1], p2[1]))), 0)
    max_y = min(int(np.ceil(max(p0[1], p1[1], p2[1]))), h - 1)
    if min_x > max_x or min_y > max_y:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float64)
    ys = np.arange(min_y, max_y + 1, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)

    d = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(d) < 1e-12:
        return  # degenerate triangle, projects to a line
    l0 = ((p1[1] - p2[1]) * (gx - p2[0]) + (p2[0] - p1[0]) * (gy - p2[1])) / d
    l1 = ((p2[1] - p0[1]) * (gx - p2[0]) + (p0[0] - p2[0]) * (gy - p2[1])) / d
    l2 = 1.0 - l0 - l1
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    if not inside.any():
        return

    z = l0 * z0 + l1 * z1 + l2 * z2
    yy = (gy[inside]).astype(np.int64)
    xx = (gx[inside]).astype(np.int64)
    zz = z[inside]

    # Nearer is smaller z. Compare against the buffer only where the triangle covers the
    # pixel, so a far triangle cannot overwrite a near one that was drawn first.
    nearer = zz < zbuf[yy, xx]
    if not nearer.any():
        return
    yy, xx, zz = yy[nearer], xx[nearer], zz[nearer]
    zbuf[yy, xx] = zz
    # `colour` arrives as a flat (3,) triple. Expanding it to (1, 3) broadcasts it across
    # the N selected pixels; a bare (3,) assigns only when N == 1, which is why this
    # failed on the first triangle whose footprint was a single pixel.
    img[yy, xx] = np.asarray(colour, dtype=np.float64).reshape(1, 3)


def render_frame(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    width: int = 512,
    height: int = 640,
    camera: Camera | None = None,
) -> np.ndarray:
    """One frame as a uint8 RGB image, (height, width, 3).

    Painter's algorithm alone is not enough for a body - two hands cross in front of the
    torso constantly in sign language - so a z-buffer decides per pixel and the loop order
    cannot matter.
    """
    cam = camera or Camera()
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (V, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F, 3), got {f.shape}")

    img = np.broadcast_to(BACKGROUND, (height, width, 3)).copy()
    zbuf = np.full((height, width), np.inf)

    screen, depth = _project(v, cam, width, height)
    intensity = _shade(f, v)

    # Sort far-to-near so later draws overwrite earlier ones even without the z test;
    # the z test is the authority, the sort just reduces rejected work.
    order = np.argsort(-depth[f].mean(axis=1))
    for fi in order:
        a, b, c = f[fi]
        shade = float(intensity[fi])
        _rasterise_triangle(
            img,
            zbuf,
            screen[a],
            screen[b],
            screen[c],
            float(depth[a]),
            float(depth[b]),
            float(depth[c]),
            np.array([shade, shade, shade]),
        )
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_sequence(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    width: int = 512,
    height: int = 640,
    camera: Camera | None = None,
) -> list[np.ndarray]:
    """Every frame of a mesh sequence, as a list of uint8 RGB images.

    Pure: the output depends only on the arguments. No randomness, no timestamps, no
    counters, so two calls on the same input are byte-identical - which is what makes the
    two study conditions comparable.
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError(f"vertices must be (T, V, 3), got {verts.shape}")
    return [
        render_frame(verts[i], faces, width=width, height=height, camera=camera)
        for i in range(verts.shape[0])
    ]


def write_mp4(
    frames: Sequence[np.ndarray],
    out: Path,
    *,
    fps: int = 25,
    crf: int = 18,
) -> Path:
    """Encode frames to H.264 mp4 via ffmpeg.

    Frames go in through stdin as raw RGB rather than as a temporary PNG directory: a
    directory is thousands of files to write and clean up, and a stale frame left behind
    from a previous run would silently enter the next video.

    `-pix_fmt yuv420p` is required, not cosmetic - without it the result plays on the
    machine that made it and shows a black frame on most other players, which is the
    worst possible failure mode for a study whose raters are on their own machines.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    for i, fr in enumerate(frames):
        if fr.shape[:2] != (h, w):
            raise ValueError(
                f"frame {i} is {fr.shape[:2]} but frame 0 is {(h, w)}; ffmpeg needs a "
                "constant frame size and would otherwise fail partway through the encode"
            )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for fr in frames:
            assert proc.stdin is not None
            proc.stdin.write(np.ascontiguousarray(fr, dtype=np.uint8).tobytes())
    finally:
        if proc.stdin is not None:
            proc.stdin.close()
        err = proc.stderr.read() if proc.stderr is not None else b""
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}) writing {out}: {err.decode(errors='replace')}")
    if not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(
            f"ffmpeg reported success but {out} is missing or empty; a zero-byte stimulus "
            "would show as a broken video to a rater"
        )
    return out
