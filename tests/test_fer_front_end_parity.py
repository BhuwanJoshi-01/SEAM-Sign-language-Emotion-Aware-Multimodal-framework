"""Rule 16: the FER fitting path and the FER inference path share one front end.

`plan.md` section 0 rule 16 requires that this be asserted by test, and until now it was
only true by inspection. The two paths are:

* **fit** - `seam.data.rafdb.decode_shard`, which turns RAF-DB parquet rows into the
  tensors the baselines are trained and parity-checked on;
* **inference** - `seam.eval.fer_audit.score_clip_faces`, which turns sign-video frames
  into the tensors those baselines are scored on in the confound audit.

The first M1 audit was invalid because these disagreed (tight uint8 boxes at fit, [0, 1]
float crops at inference), and all three models read lighting instead of faces. The tests
below fail if either path grows a normalisation of its own.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from seam.data import rafdb
from seam.eval import fer
from seam.eval import fer_audit as A

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
Image = pytest.importorskip("PIL.Image")
cv2 = pytest.importorskip("cv2")
torch = pytest.importorskip("torch")

FACE = slice(75, 543)


def _shard(path: Path, pixels: np.ndarray) -> Path:
    """One RAF-DB-shaped row whose box covers the whole image, stored losslessly."""
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    table = pa.table(
        {
            "image": [{"bytes": buf.getvalue(), "path": "x.png"}],
            "label": [0],
            "bbox_xyxy_768": [[0.0, 0.0, rafdb.BBOX_REFERENCE_PX, rafdb.BBOX_REFERENCE_PX]],
        }
    )
    pq.write_table(table, path)
    return path


def _clip(path: Path, n_frames: int = 4) -> tuple[Path, np.ndarray, np.ndarray]:
    """A short video plus landmarks that put a face box in the middle of every frame."""
    rng = np.random.default_rng(0)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (320, 240))
    assert writer.isOpened(), "cv2 could not open an mp4v writer"
    for _ in range(n_frames):
        writer.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    writer.release()

    landmarks = np.zeros((n_frames, 553, 3), dtype=np.float32)
    k = np.arange(FACE.start, FACE.stop)
    landmarks[:, FACE, 0] = 0.40 + 0.2 * ((k % 30) / 30.0)
    landmarks[:, FACE, 1] = 0.35 + 0.3 * ((k % 25) / 25.0)
    presence = np.ones((n_frames, 4), dtype=bool)
    return path, landmarks, presence


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    """Record every crop that reaches `fer.normalize_face`, without changing it."""
    seen: list[np.ndarray] = []
    real = fer.normalize_face

    def recording(crop: np.ndarray, size: int = fer.CROP_SIZE) -> np.ndarray:
        seen.append(np.array(crop, copy=True))
        return real(crop, size)

    monkeypatch.setattr(fer, "normalize_face", recording)
    return seen


def test_both_paths_call_the_same_function_object() -> None:
    """A re-implementation with the same name in either module would break this."""
    assert rafdb.preprocess_faces is fer.preprocess_faces
    assert A.fer is fer


def test_every_sample_on_both_paths_goes_through_the_shared_front_end(
    tmp_path: Path, spy: list[np.ndarray]
) -> None:
    pixels = np.random.default_rng(1).integers(0, 255, (96, 96, 3), dtype=np.uint8)
    x_fit, _ = rafdb.decode_shard(_shard(tmp_path / "s.parquet", pixels))
    assert len(x_fit) == 1
    assert len(spy) == 1, "the fitting path produced a tensor without the shared front end"
    np.testing.assert_array_equal(spy[0], pixels)

    spy.clear()
    video, landmarks, presence = _clip(tmp_path / "c.mp4")
    model = fer.build_model(fer.FerModelSpec("t", seed=0, width=8, blocks=2))
    model.eval()  # type: ignore[attr-defined]
    probs = A.score_clip_faces(video, landmarks, presence, model)
    assert len(probs) == len(landmarks)
    assert len(spy) == len(landmarks), (
        "the inference path scored a frame that did not pass through the shared front end"
    )
    for crop in spy:
        assert crop.dtype == np.uint8, "the two paths must hand over the same dtype"


def test_the_same_pixels_give_the_same_tensor_on_both_paths(
    tmp_path: Path, spy: list[np.ndarray]
) -> None:
    """The statement rule 16 actually makes: identical input, identical model input.

    The crop the inference path cut from a video frame is replayed through the fitting
    path as a lossless RAF-DB row. If the two decoded differently - colour order, dtype,
    range, resize, equalisation - the tensors would differ, and a model fitted on one
    would be scored on the other.
    """
    video, landmarks, presence = _clip(tmp_path / "c.mp4", n_frames=2)
    model = fer.build_model(fer.FerModelSpec("t", seed=0, width=8, blocks=2))
    model.eval()  # type: ignore[attr-defined]
    A.score_clip_faces(video, landmarks, presence, model)
    crop = spy[0]
    assert crop.shape == (fer.CROP_SIZE, fer.CROP_SIZE, 3)

    x_infer = fer.preprocess_faces([crop])
    x_fit, _ = rafdb.decode_shard(_shard(tmp_path / "s.parquet", crop))
    np.testing.assert_array_equal(x_fit, x_infer)


def test_the_parity_would_fail_if_the_fit_path_skipped_equalisation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check above can say no: a front end that differs on one path is detected."""
    crop = np.random.default_rng(2).integers(0, 255, (fer.CROP_SIZE, fer.CROP_SIZE, 3), np.uint8)
    x_infer = fer.preprocess_faces([crop])

    def raw_scaled(crops: list[np.ndarray], size: int = fer.CROP_SIZE) -> np.ndarray:
        return np.stack([np.transpose(c, (2, 0, 1)).astype(np.float32) / 255.0 for c in crops])

    monkeypatch.setattr(rafdb, "preprocess_faces", raw_scaled)
    x_fit, _ = rafdb.decode_shard(_shard(tmp_path / "s.parquet", crop))
    assert not np.array_equal(x_fit, x_infer)
