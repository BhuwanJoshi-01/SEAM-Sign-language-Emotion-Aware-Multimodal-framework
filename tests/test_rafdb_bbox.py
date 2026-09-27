"""The 768-space face box must be mapped into image space before cropping.

Regression tests for a defect that was measured, not guessed. The upstream
``bbox_xyxy_768`` boxes live in a 768-pixel frame while the stored image bytes are
100x100 or 512x512. Cropping with the box used directly - or clipping it to the
image bounds without rescaling - produces a rectangle of plausible size that
usually is not the face. Nothing errors. The model trains, reaches 62-66%
accuracy, and the shortfall gets attributed to the model.

So the mapping is pinned here, and so is the property that makes it trustworthy:
the crop must contain a detectable face.
"""

from __future__ import annotations

import numpy as np
import pytest

from seam.data.rafdb import BBOX_REFERENCE_PX, MIN_BOX_PX, project_box


def test_project_box_scales_into_image_space() -> None:
    # A box that fills the 768 reference frame should fill any image.
    full = (0.0, 0.0, BBOX_REFERENCE_PX, BBOX_REFERENCE_PX)
    assert project_box(full, 100, 100) == pytest.approx((0.0, 0.0, 100.0, 100.0))
    assert project_box(full, 512, 512) == pytest.approx((0.0, 0.0, 512.0, 512.0))

    # A centred 384px face in the reference frame is the middle half of the image.
    centred = (192.0, 192.0, 576.0, 576.0)
    assert project_box(centred, 100, 100) == pytest.approx((25.0, 25.0, 75.0, 75.0))


def test_project_box_is_not_identity() -> None:
    """The bug's exact shape: using the box unscaled on a smaller image.

    ``project_box`` is a pure rescale and is deliberately allowed to overshoot the
    image - clipping is the caller's job, and a face box that hangs off the edge
    is normal. What must never happen is the unscaled value being used: a
    100x100 image cannot contain a box whose x1 is 900, so the unscaled read is
    either impossible or clipped into a wrong-but-plausible rectangle.
    """
    box = (100.0, 120.0, 900.0, 880.0)
    x0, _y0, x1, _y1 = project_box(box, 100, 100)

    # Rescaled: the true 768-space extent maps to 104.17 px, which then clips.
    assert (x1 - x0) == pytest.approx(800.0 * 100.0 / BBOX_REFERENCE_PX)
    assert x0 == pytest.approx(100.0 * 100.0 / BBOX_REFERENCE_PX)

    # The unscaled value is what must NOT be used, and it is unambiguously
    # out of range for a 100 px image.
    assert box[2] > 100.0

    # So the caller clips, and what survives is a real crop rather than a corner.
    clipped_x0, _ = max(x0, 0), None
    clipped_x1 = min(x1, 100.0)
    assert clipped_x1 - clipped_x0 > MIN_BOX_PX


def test_project_box_handles_non_square_images() -> None:
    x0, y0, x1, y1 = project_box((0.0, 0.0, 768.0, 768.0), 640, 480)
    assert (x0, y0) == (0.0, 0.0)
    assert (x1, y1) == pytest.approx((640.0, 480.0))


def test_real_shard_crops_contain_a_face() -> None:
    """The property that actually justifies the mapping.

    Decode real held-out rows and require that MediaPipe finds a face in the crop.
    A clipped-but-unscaled crop passes every shape check and fails this one, which
    is the whole reason this test exists.
    """
    pytest.importorskip("pyarrow")
    pytest.importorskip("PIL")
    pytest.importorskip("mediapipe")

    from seam.data.rafdb import test_shards
    from seam.perception.tasks_api import TaskLandmarker, process_frame

    shards = test_shards()
    if not shards:
        pytest.skip("no RAF-DB held-out shard present")

    import io

    import pyarrow.parquet as pq
    from PIL import Image

    from seam.data.rafdb import REQUIRED_COLUMNS, project_box  # noqa: F401

    table = pq.read_table(shards[0], columns=["image", "bbox_xyxy_768"]).slice(0, 200)
    images = table.column("image").to_pylist()
    boxes = table.column("bbox_xyxy_768").to_pylist()

    task = TaskLandmarker(parallel=False)
    hits = 0
    total = 0
    for i, (raw, box) in enumerate(zip(images[:60], boxes[:60], strict=True)):
        if not raw or not box or len(box) != 4:
            continue
        img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        w, h = img.size
        x0, y0, x1, y1 = project_box(box, w, h)
        px0, py0 = max(int(x0), 0), max(int(y0), 0)
        px1, py1 = min(int(x1), w), min(int(y1), h)
        if px1 - px0 < MIN_BOX_PX or py1 - py0 < MIN_BOX_PX:
            continue
        crop = np.asarray(img.crop((px0, py0, px1, py1)), dtype=np.uint8)
        total += 1
        if process_frame(crop, i, task=task).presence.get("face"):
            hits += 1

    if total < 10:
        pytest.skip("too few usable rows in the held-out shard")
    rate = hits / total
    assert rate >= 0.90, (
        f"only {rate:.0%} of projected crops contain a detectable face "
        f"({hits}/{total}); the 768-space mapping has regressed"
    )
