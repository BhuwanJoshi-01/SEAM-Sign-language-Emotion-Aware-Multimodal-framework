"""Tests for the vectorised rasteriser.

**The one property that matters here is equivalence.** `fast_render` exists only to make
the study render in minutes instead of hours; if it drew a body even slightly differently
from the reference `render.py`, the stimuli would no longer be the output of the module
the rest of the project documents, and a reader comparing the code to the videos would be
looking at two different renderers. So the central test asserts byte-identical output
against `render.render_frame` on several meshes, not just a similar-looking image.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.avatar.fast_render import render_frame_fast, render_sequence_fast
from seam.avatar.render import BACKGROUND, Camera, render_frame, render_sequence

TETRA_V = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
TETRA_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)

# A denser mesh than the tetrahedron, so the batching path is actually exercised: with 4
# triangles almost any implementation passes, and the real risk is in the flattening of
# many bounding boxes into one candidate array.
GRID_N = 9
_x, _y = np.meshgrid(np.linspace(-0.5, 0.5, GRID_N), np.linspace(-0.5, 0.5, GRID_N))
GRID_V = np.stack([_x.ravel(), _y.ravel(), 0.15 * _x.ravel()], axis=1)
_grid_faces = []
for _i in range(GRID_N - 1):
    for _j in range(GRID_N - 1):
        a = _i * GRID_N + _j
        _grid_faces += [[a, a + 1, a + GRID_N], [a + 1, a + GRID_N + 1, a + GRID_N]]
GRID_F = np.array(_grid_faces, dtype=np.int64)


@pytest.mark.parametrize("w,h", [(64, 80), (128, 160)])
def test_matches_the_reference_exactly(w: int, h: int) -> None:
    """Vectorised output must be byte-identical to the per-triangle reference."""
    ref = render_frame(GRID_V, GRID_F, width=w, height=h)
    fast = render_frame_fast(GRID_V, GRID_F, width=w, height=h)
    assert np.array_equal(ref, fast), (
        "fast_render differs from render.py; the study must not use a renderer that is "
        "only approximately the documented one"
    )


def test_matches_the_reference_on_a_tetrahedron() -> None:
    ref = render_frame(TETRA_V, TETRA_F, width=96, height=96)
    fast = render_frame_fast(TETRA_V, TETRA_F, width=96, height=96)
    assert np.array_equal(ref, fast)


def test_matches_the_reference_with_a_custom_camera() -> None:
    """The camera path must be shared, not re-implemented, or the two can drift."""
    cam = Camera(yaw_deg=-40.0, pitch_deg=20.0, distance_m=3.1, fov_deg=35.0)
    ref = render_frame(GRID_V, GRID_F, width=100, height=120, camera=cam)
    fast = render_frame_fast(GRID_V, GRID_F, width=100, height=120, camera=cam)
    assert np.array_equal(ref, fast)


def test_sequence_matches_the_reference() -> None:
    moving = np.stack([GRID_V + np.array([0.1 * i, 0.0, 0.0]) for i in range(4)])
    ref = render_sequence(moving, GRID_F, width=80, height=100)
    fast = render_sequence_fast(moving, GRID_F, width=80, height=100)
    assert len(ref) == len(fast) == 4
    assert all(np.array_equal(a, b) for a, b in zip(ref, fast))


def test_a_visible_mesh_covers_pixels() -> None:
    """A body in front of the camera must actually appear, not render as background."""
    img = render_frame_fast(
        TETRA_V, TETRA_F, width=128, height=128, camera=Camera(distance_m=3.0)
    )
    bg = (BACKGROUND * 255.0).astype(int)
    n_bg = int((np.abs(img.astype(int) - bg).sum(axis=2) <= 3).sum())
    assert n_bg < img.shape[0] * img.shape[1]


def test_is_deterministic() -> None:
    a = render_frame_fast(GRID_V, GRID_F, width=96, height=96)
    b = render_frame_fast(GRID_V, GRID_F, width=96, height=96)
    assert np.array_equal(a, b)


def test_a_mesh_entirely_off_screen_renders_background() -> None:
    """Geometry behind the camera must not crash or leave a stray streak."""
    behind = TETRA_V + np.array([0.0, 0.0, -50.0])
    img = render_frame_fast(behind, TETRA_F, width=64, height=64)
    bg = (BACKGROUND * 255.0).astype(int)
    assert int((np.abs(img.astype(int) - bg).sum(axis=2) <= 3).sum()) == img.shape[0] * img.shape[1]


def test_rejects_a_bad_vertex_shape() -> None:
    with pytest.raises(ValueError, match=r"\(V, 3\)"):
        render_frame_fast(np.zeros((10, 2)), TETRA_F)
    with pytest.raises(ValueError, match=r"\(T, V, 3\)"):
        render_sequence_fast(np.zeros((5, 10)), TETRA_F)


def test_rejects_a_bad_face_shape() -> None:
    with pytest.raises(ValueError, match=r"\(F, 3\)"):
        render_frame_fast(TETRA_V, np.zeros((4, 4), dtype=np.int64))
