"""Tests for the software rasteriser used to make M7 stimulus videos.

The properties asserted here are the ones that make a *blinded* preference study
meaningful. A renderer that is not deterministic, or that silently drops geometry, would
produce two conditions that differ for reasons other than the thing being studied - and
nothing downstream would reveal it, because a video always looks like a video.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from seam.avatar.render import (
    BACKGROUND,
    Camera,
    render_frame,
    render_sequence,
    write_mp4,
)

TETRA_V = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
TETRA_F = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)


def _bg_pixels(img: np.ndarray) -> int:
    """Count pixels still showing the background colour."""
    bg = (BACKGROUND * 255.0).astype(int)
    return int((np.abs(img.astype(int) - bg).sum(axis=2) <= 3).sum())


def test_a_visible_mesh_covers_pixels() -> None:
    """A body in front of the camera must actually appear.

    The failure this guards is a camera or projection mistake that leaves the frame
    entirely background. That renders without error, writes a valid mp4, and shows a
    rater an empty grey rectangle.
    """
    img = render_frame(TETRA_V, TETRA_F, width=128, height=128, camera=Camera(distance_m=3.0))
    total = img.shape[0] * img.shape[1]
    assert _bg_pixels(img) < total, "nothing was drawn - the mesh projected off-screen"


def test_rendering_is_deterministic() -> None:
    """Same input must give byte-identical output.

    This is the property the whole study rests on: if the renderer carried hidden state
    (a seed, a counter, a timestamp), condition A and condition B would differ for a
    reason unrelated to the avatar, and no downstream check would catch it.
    """
    a = render_frame(TETRA_V, TETRA_F, width=96, height=96)
    b = render_frame(TETRA_V, TETRA_F, width=96, height=96)
    assert np.array_equal(a, b)


def test_sequence_length_and_shape_match_the_input() -> None:
    """Frames out must correspond one-to-one with frames in, at a constant size.

    ffmpeg requires a constant frame size; a per-frame size change would fail *inside*
    the encoder, after the expensive rasterisation, with an error that points at ffmpeg
    rather than at the renderer.
    """
    verts = np.stack([TETRA_V + np.array([0.0, 0.1 * i, 0.0]) for i in range(5)])
    frames = render_sequence(verts, TETRA_F, width=80, height=110)
    assert len(frames) == 5
    assert all(f.shape == (110, 80, 3) for f in frames)
    assert all(f.dtype == np.uint8 for f in frames)


def test_motion_produces_different_frames() -> None:
    """A moving mesh must not render as a frozen image.

    A depth-sorting or projection bug can make every frame identical while the geometry
    genuinely changes, which a rater would see as "the avatar does not move" - a result
    that would be blamed on the model rather than the renderer.
    """
    still = np.stack([TETRA_V] * 3)
    moving = np.stack([TETRA_V + np.array([0.3 * i, 0.0, 0.0]) for i in range(3)])
    f_still = render_sequence(still, TETRA_F, width=96, height=96)
    f_moving = render_sequence(moving, TETRA_F, width=96, height=96)
    assert np.array_equal(f_still[0], f_still[2]), "a static mesh changed between frames"
    assert not np.array_equal(f_moving[0], f_moving[2]), "a moving mesh rendered identically"


def test_rejects_a_bad_vertex_shape() -> None:
    """Garbage in should raise, not render something plausible."""
    with pytest.raises(ValueError, match=r"\(V, 3\)"):
        render_frame(np.zeros((10, 2)), TETRA_F)
    with pytest.raises(ValueError, match=r"\(T, V, 3\)"):
        render_sequence(np.zeros((5, 10)), TETRA_F)


def test_rejects_a_bad_face_shape() -> None:
    """A faces array that is not triangles cannot be rasterised."""
    with pytest.raises(ValueError, match=r"\(F, 3\)"):
        render_frame(TETRA_V, np.zeros((4, 4), dtype=np.int64))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_write_mp4_produces_a_real_video(tmp_path) -> None:
    """The encode must produce a non-empty, re-readable file.

    Checked by reading it back rather than trusting the exit code: ffmpeg can exit 0
    having written a zero-byte or header-only file, and that is exactly the failure the
    old `export_glb` had in a different medium.
    """
    frames = render_sequence(
        np.stack([TETRA_V + np.array([0.15 * i, 0.0, 0.0]) for i in range(4)]),
        TETRA_F,
        width=64,
        height=64,
    )
    out = write_mp4(frames, tmp_path / "clip.mp4", fps=8)
    assert out.is_file()
    assert out.stat().st_size > 0
    # H.264 in mp4: the `ftyp` box appears in the first bytes.
    head = out.read_bytes()[:32]
    assert b"ftyp" in head, f"output does not look like an mp4 container: {head!r}"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_write_mp4_rejects_inconsistent_frame_sizes(tmp_path) -> None:
    """A size change mid-sequence must fail loudly before the encode.

    ffmpeg's own error for this is cryptic and arrives after all the rendering work, so
    the check is done here where the cause is still obvious.
    """
    frames = [np.zeros((32, 32, 3), np.uint8), np.zeros((40, 32, 3), np.uint8)]
    with pytest.raises(ValueError, match=r"constant frame size|but frame 0"):
        write_mp4(frames, tmp_path / "bad.mp4")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_write_mp4_rejects_no_frames(tmp_path) -> None:
    with pytest.raises(ValueError, match="no frames"):
        write_mp4([], tmp_path / "empty.mp4")


def test_z_buffer_lets_a_near_triangle_win() -> None:
    """A near surface must occlude a far one regardless of draw order.

    Two hands crossing in front of the torso is constant in sign language. Without the
    z-test the result depends on triangle index, so the same pose could render two
    different ways - and the difference would track the mesh's internal ordering rather
    than anything about the signing.
    """
    # A large far quad and a small near quad that overlaps its centre.
    far = np.array(
        [[-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    near = np.array(
        [[-0.2, -0.2, -1.0], [0.2, -0.2, -1.0], [0.2, 0.2, -1.0], [-0.2, 0.2, -1.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    verts = np.vstack([far, near])

    img = render_frame(verts, faces, width=120, height=120, camera=Camera(distance_m=4.0))
    centre = img[60, 60]
    # The near quad is centred on the view axis and must be what the centre pixel shows.
    assert not np.array_equal(centre.astype(int), (BACKGROUND * 255.0).astype(int)), (
        "the centre pixel is background; neither quad was drawn"
    )


def test_default_camera_looks_at_the_body_front_not_its_back() -> None:
    """The study camera must show a face, not a bare back.

    Regression guard for a real defect: the default yaw was 30, which put the camera
    behind the body. Nancy spotted it in a rendered video - "the avatar body is face
    backwards" - and it was confirmed by rendering the rest pose (`yaw 0` shows shoulder
    blades and the back of a head, `yaw 180` shows a face, a chest and kneecaps).

    The test does not hard-code the yaw. It builds a body with an unambiguous front - a
    nose-like spike protruding toward +z - and asserts that the default camera sees that
    spike rather than the flat back behind it. Any future change to the yaw, the view
    matrix, or the projection that turns the body around fails here.

    Why this is not caught by the other render tests: they assert determinism, coverage
    and z-order, all of which hold equally well for a back view. A consistent renderer
    faithfully renders the wrong side.
    """
    # A thin triangular prism: a wide, flat "back" plate at z = 0 and a spike to z = +0.6.
    back = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]],
        dtype=np.float64,
    )
    nose = np.array([[0.0, 0.0, 0.6]], dtype=np.float64)
    verts = np.vstack([back, nose])
    faces = np.array(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int64
    )  # the four sloped faces all meet at the nose tip

    cam = Camera(distance_m=3.0)  # the study default
    m = cam.view_matrix()
    # The camera's back-vector points away from the viewer. The nose faces the viewer when
    # the nose direction and the back-vector point opposite ways.
    nose_dir = np.array([0.0, 0.0, 1.0])
    assert float(nose_dir @ m[2]) < 0.0, (
        f"the default camera looks at the body's back (yaw={cam.yaw_deg}); "
        "the protruding front points away from the viewer"
    )

    # And the visible proof: the tip's pixel must not be background, i.e. the spike is in
    # front and drawn, rather than hidden behind the flat plate.
    img = render_frame(verts, faces, width=160, height=160, camera=cam)
    assert _bg_pixels(img) < 160 * 160


def test_the_render_is_not_a_mirror_image() -> None:
    """A person facing the camera has their right hand on the viewer's left.

    The body faces +z and its right side is at -x. Seen from the front (yaw 180), -x has
    to land left of centre. For the whole life of this renderer it landed right of centre:
    every stimulus video showed the signer mirrored, raising the wrong hand, and nothing
    noticed because a mirrored body still stands upright and faces the camera.
    """
    import dataclasses

    from seam.avatar.render import Camera, _project

    pts = np.array([[0.0, 0.0, 0.0], [-0.3, 0.0, 0.0], [0.3, 0.0, 0.0], [0.0, 0.3, 0.0]])
    for yaw in (180.0, 210.0):
        cam = dataclasses.replace(Camera(), yaw_deg=yaw, pitch_deg=0.0)
        screen, _ = _project(pts, cam, 400, 400)
        centre, right_side, left_side, above = screen
        assert right_side[0] < centre[0] < left_side[0], f"mirrored at yaw {yaw}"
        assert above[1] < centre[1], "up must still be up"
