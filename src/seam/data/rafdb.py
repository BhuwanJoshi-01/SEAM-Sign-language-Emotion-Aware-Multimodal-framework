"""One decoder for the RAF-DB shards, shared by training and by export parity.

This module exists because of a real defect. M1's first FER models were trained by
one script and evaluated by another, and the two disagreed about crop geometry.
The models were not wrong, the *measurement* was: accuracy numbers were compared
across two different input pipelines, and the discrepancy was read as a model
result. Six hours of retraining followed.

So decoding lives here, once, and both the trainer and the ONNX parity harness
call it. There is exactly one place where a parquet row becomes a model tensor.

The parity harness in particular needs this. Parity was first measured on
synthetic noise, which is the worst possible input for the question being asked:
noise images push a FER model to near-uniform probabilities, so an argmax is
decided by a vanishing margin and flips on numerical noise alone. That produced a
95.8% argmax agreement for ``fer_cnn_a`` under INT8, which says almost nothing
about the quantized model. The same check on real held-out faces is the
meaningful one, and getting the real images requires this decoder.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from seam.eval.fer import CROP_SIZE, preprocess_faces
from seam.logging import get
from seam.paths import default_data_root

log = get(__name__)


def rafdb_dir() -> Path:
    """Where the RAF-DB shards live.

    Resolved from the data root rather than hardcoded, so the same code runs on a
    machine with a different mount. The previous hardcoded absolute path in the
    training script is what made that script unrunnable anywhere but this box.
    """
    return default_data_root() / "rafdb"


#: Columns required to reconstruct a crop. The bbox is the one MediaPipe already
#: computed at 768 px, so the crop matches ``seam.eval.fer.face_crop`` instead of
#: being a second, subtly different geometry.
REQUIRED_COLUMNS = ("image", "label", "bbox_xyxy_768")

#: A face box smaller than this in either axis is a detection artefact, not a face.
MIN_BOX_PX = 8

#: The reference frame the upstream ``bbox_xyxy_768`` boxes are expressed in.
#:
#: The boxes are in a 768-pixel coordinate space regardless of the resolution the
#: image bytes were stored at - the column name says so, and the stored images are
#: a mix of 100x100 and 512x512. Applying the box straight to the image, or
#: clipping it to the image bounds, is wrong in both cases, and the failure is
#: quiet: the crop is still a plausible rectangle of plausible size, so it passes
#: every size check and simply is not the face.
#:
#: Measured on 40 held-out rows per shard by running face detection on the crop
#: and counting hits:
#:
#:   mapping                    face detected
#:   clip to image bounds            52%
#:   scale by size/1000             82%
#:   scale by size/768             100%   <- this one
#:
#: So the 768 mapping is not an assumption fitted to the column name, it is the
#: only one of the candidates under which the crop reliably contains a face.
BBOX_REFERENCE_PX = 768.0


def project_box(box: Sequence[float], width: int, height: int) -> tuple[float, float, float, float]:
    """Map a 768-space box into the pixel space of a ``width`` x ``height`` image."""
    sx, sy = width / BBOX_REFERENCE_PX, height / BBOX_REFERENCE_PX
    x0, y0, x1, y1 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def decode_shard(
    parquet_path: Path, *, limit: int | None = None, size: int = CROP_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Load one parquet shard into ``(x, y)``.

    ``x`` is ``(N, 3, size, size)`` float32 through the shared front end, ``y`` is
    ``(N,)`` int64. Rows whose image bytes are missing, undecodable, or whose face
    box is degenerate are skipped rather than repaired - a silently padded face is
    worse than an absent one, and the skip count is logged so the gap is visible.
    """
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet_path, columns=list(REQUIRED_COLUMNS))
    images = table.column("image").to_pylist()
    labels = table.column("label").to_pylist()
    boxes = table.column("bbox_xyxy_768").to_pylist()
    n = len(images) if limit is None else min(limit, len(images))

    crops: list[np.ndarray] = []
    ys: list[int] = []
    # Skip reasons are counted separately, never as a single "skipped" number.
    # The three causes need different responses: missing bytes is a fetch problem
    # and is fixable, an undecodable image is a corrupt row, and a degenerate box
    # is a face-detection problem in the upstream dataset. Reporting them as one
    # number hides which one is happening, and an unlabelled drop rate cannot be
    # acted on - it can only be noted.
    reasons = {"missing_bytes": 0, "bad_box": 0, "undecodable": 0, "tiny_box": 0}
    for i in range(n):
        raw = images[i].get("bytes") if isinstance(images[i], dict) else None
        box = boxes[i]
        if not raw:
            reasons["missing_bytes"] += 1
            continue
        if not box or len(box) != 4:
            reasons["bad_box"] += 1
            continue
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            reasons["undecodable"] += 1
            continue
        w, h = img.size
        # 768-space -> image space, then clip. Clipping without the rescale is the
        # defect documented on BBOX_REFERENCE_PX.
        x0, y0, x1, y1 = project_box(box, w, h)
        px0, py0 = max(int(x0), 0), max(int(y0), 0)
        px1, py1 = min(int(x1), w), min(int(y1), h)
        if px1 - px0 < MIN_BOX_PX or py1 - py0 < MIN_BOX_PX:
            reasons["tiny_box"] += 1
            continue
        crops.append(np.asarray(img.crop((px0, py0, px1, py1)), dtype=np.uint8))
        ys.append(int(labels[i]))

    skipped = sum(reasons.values())
    if skipped:
        detail = ", ".join(f"{k}={v}" for k, v in reasons.items() if v)
        log.info("%s: skipped %d of %d rows (%s)", parquet_path.name, skipped, n, detail)
    if not crops:
        return np.zeros((0, 3, size, size), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return preprocess_faces(crops, size), np.asarray(ys, dtype=np.int64)


def test_shards() -> list[Path]:
    """Held-out shards, in a stable order."""
    d = rafdb_dir()
    found = sorted(d.glob("test*.parquet"))
    val = d / "val.parquet"
    return found or ([val] if val.is_file() else [])


def train_shards() -> list[Path]:
    return sorted(rafdb_dir().glob("train_*.parquet"))


def load_split(limit: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(x_train, y_train, x_test, y_test)`` from whatever shards exist.

    Streams shard by shard and concatenates at the end. The full set does not fit
    in this machine's memory budget alongside anything else, and reading a 400 MB
    parquet through ``to_pylist()`` is the single largest allocation in the
    project - so callers that only need a subsample pass ``limit``.
    """
    trains, tests = train_shards(), test_shards()
    if not trains:
        raise FileNotFoundError(f"no train shards under {rafdb_dir()}; run scripts/fetch_rafdb.py")
    if not tests:
        raise FileNotFoundError(f"no held-out shards under {rafdb_dir()}")

    per_shard = None if limit is None else max(1, limit // len(trains))
    xt, yt = [], []
    for f in trains:
        x, y = decode_shard(f, limit=per_shard, size=CROP_SIZE)
        if len(x):
            xt.append(x)
            yt.append(y)
        log.info("loaded %s -> %d images", f.name, len(x))

    xs, ys = [], []
    for f in tests:
        x, y = decode_shard(f, limit=limit, size=CROP_SIZE)
        if len(x):
            xs.append(x)
            ys.append(y)
        log.info("loaded %s -> %d images", f.name, len(x))

    if not xs or not xt:
        raise RuntimeError("no usable RAF-DB data")
    return np.concatenate(xt), np.concatenate(yt), np.concatenate(xs), np.concatenate(ys)


def parity_batches(
    size: int = CROP_SIZE, *, per_batch: int = 4, count: int = 24
) -> list[np.ndarray]:
    """Real held-out faces, batched, for ONNX-vs-PyTorch parity checking.

    Batch sizes rotate through 1, 2 and 4 so the exported graph's dynamic batch
    axis is actually exercised. A dynamic axis that was never run is a claim
    rather than a measurement.
    """
    xs: list[np.ndarray] = []
    for shard in test_shards():
        x, _y = decode_shard(shard, limit=count * per_batch, size=size)
        if len(x):
            xs.append(x[: count * per_batch])
            break
    if not xs:
        raise RuntimeError("no held-out RAF-DB faces available for parity checking")
    pool = np.concatenate(xs)
    out: list[np.ndarray] = []
    i = 0
    step = 0
    # Cycle the batch size rather than iterating the tuple once. ``for b in
    # (1, 2, 4)`` runs three times and stops, which silently yields three batches
    # however many were asked for - a parity claim on 7 images that reads like a
    # claim on 24.
    while len(out) < count and i < len(pool):
        b = (1, 2, 4)[step % 3]
        step += 1
        if i + b > len(pool):
            # Not enough left for the next size: finish with what is available
            # rather than dropping the remainder, so the sample is as large as the
            # data allows.
            out.append(pool[i:])
            break
        out.append(pool[i : i + b])
        i += b
    if not out:
        out = [pool[:1]]
    return out
