"""A vectorised software rasteriser for the M7 stimulus videos.

**Why this exists.** :mod:`seam.avatar.render` is correct and deterministic, and it is
the reference implementation this module is checked against. But it rasterises *one
triangle per Python call* - 20,908 calls for a whole-body SMPL-X mesh - so a single frame
costs about 1.7 s even at a small resolution. A 148-clip study at ~85 frames each is then
hours of pure Python, which is not a scheduling problem you can wait out, it is a design
one.

**What is different, and what is deliberately identical.** The maths is the same maths:
same camera, same projection, same flat Lambert shading, same z-buffer, same
painter's-algorithm pre-sort. Only the *loop structure* changes - the per-pixel work is
batched with numpy instead of being handed to Python one triangle at a time. A test
asserts the output is identical to :func:`seam.avatar.render.render_frame` rather than
trusting that it is.

**How the batching works.** Every triangle is expanded into its own bounding box, and all
boxes are stacked into one flat candidate array. Each triangle's barycentric coordinates
are computed for its own box pixels in a single vectorised pass, the inside-mask is
applied, and the surviving (triangle, pixel) candidates are resolved with a z-buffer:
`np.minimum.at` writes, per pixel, the nearest depth; then the candidate whose depth
equals that nearest depth is the winner and its flat shade is written. That is exactly
the reference's per-pixel outcome, without the reference's per-triangle Python overhead.

**What it does not do.** No anti-aliasing, no textures, no shadows - the same omissions as
the reference, for the same reason: any flourish that could differ between the study
conditions is a confound.
"""

from __future__ import annotations

import numpy as np

from seam.avatar.render import BACKGROUND, Camera, _project, _shade


def render_frame_fast(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    width: int = 480,
    height: int = 600,
    camera: Camera | None = None,
) -> np.ndarray:
    """One frame as a uint8 RGB image, (height, width, 3), vectorised."""
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
    intensity = _shade(f, v)  # (F,) flat Lambert shade per triangle

    p = screen[f]  # (F, 3, 2) triangle corners in screen space
    z = depth[f]  # (F, 3) per-corner camera depth

    min_x = np.clip(np.floor(p[:, :, 0].min(axis=1)).astype(np.int64), 0, width - 1)
    max_x = np.clip(np.ceil(p[:, :, 0].max(axis=1)).astype(np.int64), 0, width - 1)
    min_y = np.clip(np.floor(p[:, :, 1].min(axis=1)).astype(np.int64), 0, height - 1)
    max_y = np.clip(np.ceil(p[:, :, 1].max(axis=1)).astype(np.int64), 0, height - 1)

    # Signed area; a triangle that projects to a line has ~0 area and fills nothing.
    d = (p[:, 1, 1] - p[:, 2, 1]) * (p[:, 0, 0] - p[:, 2, 0]) + (p[:, 2, 0] - p[:, 1, 0]) * (
        p[:, 0, 1] - p[:, 2, 1]
    )
    keep = (np.abs(d) >= 1e-12) & (min_x <= max_x) & (min_y <= max_y)
    if not keep.any():
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    idx = np.nonzero(keep)[0]
    p, z, d = p[idx], z[idx], d[idx]
    min_x, max_x, min_y, max_y = min_x[idx], max_x[idx], min_y[idx], max_y[idx]
    shade = intensity[idx]

    bw = max_x - min_x + 1
    bh = max_y - min_y + 1
    counts = bw * bh

    # Flatten every triangle's bounding box into one candidate array. `tri_of` says which
    # triangle each candidate belongs to; (gx, gy) is its screen pixel.
    tri_of = np.repeat(np.arange(len(idx)), counts)
    offs = np.concatenate([[0], np.cumsum(counts)[:-1]])
    # Local box coordinates, rebuilt cheaply with cumsum rather than a per-triangle tile.
    local = np.arange(int(counts.sum())) - np.repeat(offs, counts)
    dx = local % np.repeat(bw, counts)
    dy = local // np.repeat(bw, counts)
    gx = min_x[tri_of] + dx
    gy = min_y[tri_of] + dy

    p0x, p0y = p[tri_of, 0, 0], p[tri_of, 0, 1]
    p1x, p1y = p[tri_of, 1, 0], p[tri_of, 1, 1]
    p2x, p2y = p[tri_of, 2, 0], p[tri_of, 2, 1]
    dd = d[tri_of]
    l0 = ((p1y - p2y) * (gx - p2x) + (p2x - p1x) * (gy - p2y)) / dd
    l1 = ((p2y - p0y) * (gx - p2x) + (p0x - p2x) * (gy - p2y)) / dd
    l2 = 1.0 - l0 - l1
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    if not inside.any():
        return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    tri_of = tri_of[inside]
    gx, gy = gx[inside], gy[inside]
    zz = l0[inside] * z[tri_of, 0] + l1[inside] * z[tri_of, 1] + l2[inside] * z[tri_of, 2]
    sh = shade[tri_of]

    # z-buffer: per pixel, the nearest depth across all covering candidates.
    np.minimum.at(zbuf, (gy, gx), zz)
    # Winner per pixel: the candidate whose depth equals that nearest depth. Ties are
    # broken by write order, which is irrelevant here because equal depth means the two
    # candidates are the same distance and their shades differ only by floating noise.
    win = zz <= zbuf[gy, gx] + 1e-12
    img[gy[win], gx[win]] = sh[win, None]
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_sequence_fast(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    width: int = 480,
    height: int = 600,
    camera: Camera | None = None,
) -> list[np.ndarray]:
    """Every frame of a mesh sequence, vectorised. See :func:`render_frame_fast`."""
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 3 or verts.shape[2] != 3:
        raise ValueError(f"vertices must be (T, V, 3), got {verts.shape}")
    return [
        render_frame_fast(verts[i], faces, width=width, height=height, camera=camera)
        for i in range(verts.shape[0])
    ]
