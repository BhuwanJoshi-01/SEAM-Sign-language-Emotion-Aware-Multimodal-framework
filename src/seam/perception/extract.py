"""Video -> landmark record, as resumable ``.npz`` shards plus JSON sidecars.

Design decisions that come from measurements rather than preference:

* **Store everything.** All 553 points and all 52 blendshapes are persisted, and
  the point subset is chosen at load time. Changing the feature set must never
  require re-extraction, which on this machine costs hours.
* **float16 for landmarks, float32 for blendshapes.** Landmarks are normalized
  coordinates in [-1, 1]; float16 resolves them to ~5e-4, well below the noise
  floor of the detector. Blendshape coefficients are small differences between
  competing hypotheses, so they keep full precision.
* **Resumable at the clip level.** A 200-clip run that dies at clip 180 must not
  redo 180 clips. Presence of a valid sidecar is the resume marker, and a
  sidecar whose frame count disagrees with the array is rejected and re-extracted.
* **Presence is stored, not imputed.** A frame with no detected hand has zeros
  there and ``False`` in the mask. Filling it in would make "the hand was not
  visible" indistinguishable from "the hand was still", which is exactly the
  distinction the prosody channel depends on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from seam.logging import get
from seam.perception.tasks_api import (
    FACE_BLENDSHAPES,
    PART_SLICES,
    TOTAL_POINTS,
    Frame,
    process_frame,
)

log = get(__name__)

PART_NAMES = tuple(PART_SLICES)


class ExtractionError(RuntimeError):
    """Raised when a clip cannot be decoded at all."""


@dataclass(slots=True)
class ClipMeta:
    """The JSON sidecar. Everything needed to interpret the arrays."""

    source_path: str
    utterance_id: str
    native_fps: float
    source_width: int
    source_height: int
    frame_count: int
    #: fraction of frames in which each part was detected, per part
    detection_rates: dict[str, float]
    landmark_order: list[dict[str, object]] = field(default_factory=list)
    blendshape_order: list[str] = field(default_factory=list)
    extractor: str = "mediapipe-tasks"
    schema_version: int = 1

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def default_order() -> tuple[list[dict[str, object]], list[str]]:
    order = [
        {"part": name, "start": start, "count": end - start}
        for name, (start, end) in PART_SLICES.items()
    ]
    from seam.perception.tasks_api import BLENDSHAPE_CATEGORIES

    return order, list(BLENDSHAPE_CATEGORIES)


def extract_clip(
    video_path: Path,
    *,
    utterance_id: str | None = None,
    max_frames: int | None = None,
    every_n: int = 1,
) -> tuple[dict[str, np.ndarray], ClipMeta]:
    """Decode a video and run perception over it.

    Returns the arrays dict and its metadata. ``every_n`` subsamples frames, used
    by the 12 fps path in M1; the extractor itself does not resample so that the
    resampling decision stays in ``preprocess`` where it can be tested.
    """
    import cv2

    if not video_path.is_file():
        raise ExtractionError(f"no such video: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ExtractionError(f"OpenCV could not open {video_path} (truncated or corrupt?)")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    landmarks: list[np.ndarray] = []
    blendshapes: list[np.ndarray] = []
    presence: list[list[bool]] = []
    head_rotations: list[np.ndarray] = []
    gaze: list[np.ndarray] = []

    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index % every_n == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame: Frame = process_frame(rgb, index)
            landmarks.append(frame.landmarks.astype(np.float16))
            blendshapes.append(frame.blendshapes.astype(np.float32))
            presence.append([frame.presence[p] for p in PART_NAMES])
            if frame.head_rotation is not None:
                head_rotations.append(frame.head_rotation)
            if frame.gaze is not None:
                gaze.append(frame.gaze)
        index += 1
        if max_frames is not None and len(landmarks) >= max_frames:
            break
    cap.release()

    if not landmarks:
        raise ExtractionError(f"{video_path} decoded zero frames")

    pres = np.asarray(presence, dtype=bool)
    arrays: dict[str, np.ndarray] = {
        "landmarks": np.stack(landmarks),
        "blendshapes": np.stack(blendshapes),
        "presence": pres,
    }
    if head_rotations:
        # Pad so the array is rectangular; frames with no face contribute identity.
        stack = np.zeros((len(blendshapes), 4, 4), dtype=np.float32)
        for i in range(len(blendshapes)):
            stack[i] = np.eye(4, dtype=np.float32)
        stack[: len(head_rotations)] = np.stack(head_rotations)
        arrays["head_rotation"] = stack
    if gaze:
        stack = np.zeros((len(blendshapes), 2), dtype=np.float32)
        stack[: len(gaze)] = np.stack(gaze)
        arrays["gaze"] = stack

    order, shape_order = default_order()
    meta = ClipMeta(
        source_path=str(video_path),
        utterance_id=utterance_id or video_path.stem,
        native_fps=native_fps,
        source_width=width,
        source_height=height,
        frame_count=len(blendshapes),
        detection_rates={p: float(pres[:, i].mean()) for i, p in enumerate(PART_NAMES)},
        landmark_order=order,
        blendshape_order=shape_order,
    )
    return arrays, meta


def shard_path(root: Path, utterance_id: str) -> Path:
    return root / f"{utterance_id}.npz"


def is_extracted(root: Path, utterance_id: str) -> bool:
    """Whether a valid shard already exists for this clip.

    Validity means both files exist *and* the sidecar's frame count matches the
    array. A shard truncated by a full disk is the failure this catches.
    """
    npz = shard_path(root, utterance_id)
    side = root / f"{utterance_id}.json"
    if not (npz.is_file() and side.is_file()):
        return False
    try:
        meta = json.loads(side.read_text(encoding="utf-8"))
        with np.load(npz) as data:
            return int(data["blendshapes"].shape[0]) == int(meta["frame_count"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def save_shard(root: Path, arrays: dict[str, np.ndarray], meta: ClipMeta) -> Path:
    """Write a shard atomically, then its sidecar."""
    root.mkdir(parents=True, exist_ok=True)
    npz = shard_path(root, meta.utterance_id)
    tmp = npz.with_suffix(".npz.part")
    # np.savez's overloads do not model a file object plus **kwargs; the call is
    # correct, the type stub is not.
    with tmp.open("wb") as fh:
        np.savez_compressed(fh, **arrays)  # type: ignore[arg-type]
    tmp.replace(npz)
    meta.save(root / f"{meta.utterance_id}.json")
    return npz


def load_shard(path: Path) -> tuple[dict[str, np.ndarray], ClipMeta]:
    """Load a shard and its sidecar, validating the contract."""
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}

    side = path.with_suffix(".json")
    raw = json.loads(side.read_text(encoding="utf-8"))
    fields = set(ClipMeta.__dataclass_fields__)
    meta = ClipMeta(**{k: v for k, v in raw.items() if k in fields})

    n = arrays["blendshapes"].shape[0]
    if arrays["landmarks"].shape != (n, TOTAL_POINTS, 3):
        raise ExtractionError(
            f"{path.name}: landmarks {arrays['landmarks'].shape} inconsistent with "
            f"{n} frames x {TOTAL_POINTS} points"
        )
    if arrays["blendshapes"].shape[1] != FACE_BLENDSHAPES:
        raise ExtractionError(
            f"{path.name}: blendshapes width {arrays['blendshapes'].shape[1]}, "
            f"expected {FACE_BLENDSHAPES}"
        )
    if meta.blendshape_order and tuple(meta.blendshape_order) != tuple(
        raw.get("blendshape_order", ())
    ):
        raise ExtractionError(f"{path.name}: sidecar blendshape order is self-inconsistent")
    return arrays, meta
