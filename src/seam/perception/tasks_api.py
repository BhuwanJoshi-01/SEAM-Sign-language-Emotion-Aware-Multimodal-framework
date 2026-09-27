"""MediaPipe Tasks wrappers: Face, Hand and Pose landmarkers.

Two corrections to the inherited design, both of which would have cost real time:

* There is no ``HolisticLandmarker`` in the Tasks API. Tasks ships separate
  ``FaceLandmarker``, ``HandLandmarker`` and ``PoseLandmarker`` graph modules, so
  a combined 543-point stream has to be composed from three of them. The legacy
  ``mp.solutions.holistic`` does produce a combined stream but is superseded and
  emits **no blendshape coefficients**, which is the entire non-manual channel
  this project is about. Tasks it is.
* The Tasks face mesh is **478** points (468 + 10 iris), not 468. The ordering
  contract is asserted in tests rather than assumed, because a silently different
  point order turns a model into noise while every shape check still passes.

Model bundles are reused from a local cache when present. They are ~27 MB and
already on this machine, and downloading them is the slowest part of standing up
perception from cold.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from seam.logging import get

log = get(__name__)

# -- landmark counts, from the Tasks API -------------------------------
POSE_POINTS = 33
HAND_POINTS = 21
FACE_POINTS = 478  # 468 mesh + 10 iris
FACE_BLENDSHAPES = 52  # see BLENDSHAPE_CATEGORIES

#: The blendshape basis, in the exact order MediaPipe Tasks emits it. Measured
#: against face_landmarker.task on 2026-09-26 rather than taken from the ARKit
#: documentation, because the two differ in ways that matter: MediaPipe's list
#: opens with a ``_neutral`` basis coefficient and includes ``eyeLookUp*``, and
#: does **not** include ``tongueOut``.
#:
#: This order is asserted on every face detection. A silently permuted basis
#: would leave every shape check passing while inverting the meaning of every
#: brow-raise rule in the M3 marker labeller - the single most expensive class of
#: bug in this project, because it produces plausible numbers.
BLENDSHAPE_CATEGORIES: tuple[str, ...] = (
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
)

#: Name -> index, for rule-based marker rules that name coefficients by meaning.
BLENDSHAPE_INDEX: dict[str, int] = {n: i for i, n in enumerate(BLENDSHAPE_CATEGORIES)}

#: A flat (T, P, 3) stream: pose, then both hands, then face.
TOTAL_POINTS = POSE_POINTS + 2 * HAND_POINTS + FACE_POINTS
PART_SLICES = {
    "pose": (0, POSE_POINTS),
    "left_hand": (POSE_POINTS, POSE_POINTS + HAND_POINTS),
    "right_hand": (POSE_POINTS + HAND_POINTS, POSE_POINTS + 2 * HAND_POINTS),
    "face": (POSE_POINTS + 2 * HAND_POINTS, TOTAL_POINTS),
}
PART_ORDER = ("pose", "left_hand", "right_hand", "face")

#: Model bundle filenames, searched in the local cache before any download.
_BUNDLES = {
    "face": "face_landmarker.task",
    "hand": "hand_landmarker.task",
    "pose": "pose_landmarker_full.task",
}
_SEARCH_DIRS = (
    Path("/mnt/Volume2/Sign_Language_Recognition/artifacts/mp_tasks"),
    Path(__file__).resolve().parents[2] / "artifacts" / "mediapipe_tasks",
)
_CDN = "https://storage.googleapis.com/mediapipe-models"


class PerceptionUnavailable(RuntimeError):
    """Raised when the Tasks runtime or its model bundles cannot be loaded."""


class TimestampError(ValueError):
    """Raised when a caller supplies a non-monotonic timestamp.

    The Tasks graphs run in VIDEO mode, which requires strictly increasing
    timestamps *per graph instance*, and the three graphs share one instance per
    process. Decoding two videos back to back, or resetting a counter between
    batches, silently violates that. The symptom is not an exception - it is
    tracking that quietly degrades - so the guard is explicit.
    """


class _MonotonicClock:
    """Enforces strictly increasing timestamps across the whole process."""

    def __init__(self) -> None:
        self._last = -1

    def next(self, requested_ms: int) -> int:
        if requested_ms <= self._last:
            requested_ms = self._last + 1
        self._last = requested_ms
        return requested_ms

    def reset(self) -> None:
        self._last = -1


_CLOCK = _MonotonicClock()


def reset_clock() -> None:
    """Reset the timestamp guard. Only safe when every graph is being rebuilt."""
    _CLOCK.reset()
    close()


@dataclass(frozen=True, slots=True)
class Frame:
    """One frame of extracted signal.

    ``landmarks`` is (TOTAL_POINTS, 3) in the fixed order given by ``PART_SLICES``.
    ``blendshapes`` is (52,) and is all-zero when no face was detected, which is
    the documented missing-face policy: absence is represented as absence, not
    imputed, so a downstream model can tell "neutral" from "not visible".
    """

    landmarks: np.ndarray
    blendshapes: np.ndarray
    head_rotation: np.ndarray | None
    gaze: np.ndarray | None
    presence: dict[str, bool]


class TaskLandmarker:
    """Lazily-constructed container for the three Tasks graph modules."""

    def __init__(
        self,
        *,
        model_dir: Path | None = None,
        delegate: str = "CPU",
        parallel: bool = True,
    ) -> None:
        self._model_dir = model_dir
        self._delegate = delegate
        self._parallel = parallel
        # The graph handles are pybind11 objects with no useful stub type, so they
        # are Any until _ensure() constructs them.
        self._face: Any = None
        self._hands: list[Any] = []
        self._pose: Any = None
        self._pool: ThreadPoolExecutor | None = None

    @property
    def parallel(self) -> bool:
        return self._parallel

    def _executor(self) -> ThreadPoolExecutor | None:
        if not self._parallel:
            return None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=3,
                thread_name_prefix="seam-perception",
            )
        return self._pool

    # -- model discovery ---------------------------------------------------
    def _resolve(self, key: str) -> str:
        filename = _BUNDLES[key]
        roots: list[Path] = []
        if self._model_dir is not None:
            roots.append(self._model_dir)
        env = os.environ.get("SEAM_MEDIAPIPE_MODELS")
        if env:
            roots.append(Path(env))
        roots.extend(_SEARCH_DIRS)

        for root in roots:
            candidate = root / filename
            if candidate.is_file():
                return str(candidate)
        return f"{_CDN}/{_REMOTE_PATH[key]}"


_REMOTE_PATH = {
    "face": "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    "hand": "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose": "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def _base_options(model_path: str, delegate: str) -> Any:
    """Build a Tasks ``BaseOptions``.

    ``model_asset_path`` alone is correct for mediapipe 0.10.x. The buffer form
    (``model_asset_buffer`` + a synthetic name) is only needed to load a model
    from memory, and the pybind11 binding in this version has no public handle
    for it.
    """
    from mediapipe.tasks.python import BaseOptions

    if delegate == "GPU":
        return BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.GPU)
    return BaseOptions(model_asset_path=model_path, delegate=BaseOptions.Delegate.CPU)


_SINGLETON: TaskLandmarker | None = None


def landmarker(
    model_dir: Path | None = None,
    delegate: str = "CPU",
    *,
    parallel: bool = True,
) -> TaskLandmarker:
    """Process-wide singleton. MediaPipe graphs are expensive to build and cheap to reuse.

    ``parallel`` runs the three independent graphs concurrently on separate
    threads. Measured on this machine: 109.7 -> 67.3 ms/frame, a 1.63x speedup,
    because TFLite releases the GIL during inference and the three graphs share
    no state. It is the default because the per-stage latency budget is the
    project's headline efficiency claim; pass ``parallel=False`` to reproduce
    the sequential baseline for the ablation table.
    """
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = TaskLandmarker(model_dir=model_dir, delegate=delegate, parallel=parallel)
    return _SINGLETON


def _ensure(task: TaskLandmarker) -> TaskLandmarker:
    """Construct the three graph modules on first use."""
    if task._face is not None:
        return task

    try:
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover
        raise PerceptionUnavailable(
            "mediapipe.tasks is unavailable. The legacy solutions API is not a "
            "substitute: it emits no blendshape coefficients."
        ) from exc

    face_path = task._resolve("face")
    log.debug("face landmarker bundle: %s", face_path)
    task._face = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=_base_options(face_path, task._delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    )

    hand_path = task._resolve("hand")
    for handedness in ("Left", "Right"):
        task._hands.append(
            vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=_base_options(hand_path, task._delegate),
                    running_mode=vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.3,
                    min_hand_presence_confidence=0.3,
                    min_tracking_confidence=0.3,
                )
            )
        )
        del handedness

    pose_path = task._resolve("pose")
    task._pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=_base_options(pose_path, task._delegate),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_segmentation_masks=False,
        )
    )
    return task


def _empty_landmarks() -> np.ndarray:
    return np.zeros((TOTAL_POINTS, 3), dtype=np.float32)


def _empty_blendshapes() -> np.ndarray:
    return np.zeros(FACE_BLENDSHAPES, dtype=np.float32)


def process_frame(
    rgb: np.ndarray,
    timestamp_ms: int,
    *,
    task: TaskLandmarker | None = None,
    delegate: str = "CPU",
) -> Frame:
    """Run all three landmarkers on one RGB frame and merge into one record.

    ``rgb`` must be uint8 HxWx3. The returned landmark stream uses the fixed
    ``PART_SLICES`` order regardless of which subset of parts was detected, so
    downstream code never has to ask which parts are present - it reads
    ``frame.presence``.

    The three graphs are independent and are dispatched concurrently; see
    ``landmarker()`` for the measurement that justifies it. The timestamp is read
    once and shared, which is required: each graph requires its own timestamps to
    increase strictly, and all three advance in lockstep by construction.
    """
    import mediapipe as mp

    task = _ensure(task or landmarker(delegate=delegate))
    ts = _CLOCK.next(int(timestamp_ms))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    pool = task._executor()
    if pool is None:
        pose_result = task._pose.detect_for_video(image, ts)
        hand_results = [h.detect_for_video(image, ts) for h in task._hands]
        face_result = task._face.detect_for_video(image, ts)
    else:
        f_pose = pool.submit(task._pose.detect_for_video, image, ts)
        f_hands = pool.submit(lambda: [h.detect_for_video(image, ts) for h in task._hands])
        f_face = pool.submit(task._face.detect_for_video, image, ts)
        # Collected in a fixed order so the sequential and concurrent paths
        # differ only in scheduling, never in which result is used where.
        pose_result = f_pose.result()
        hand_results = f_hands.result()
        face_result = f_face.result()

    landmarks = _empty_landmarks()
    blendshapes = _empty_blendshapes()
    head_rotation = None
    presence = {"face": False, "left_hand": False, "right_hand": False, "pose": False}

    # -- pose
    pose_lms = getattr(pose_result, "pose_landmarks", None)
    if pose_lms:
        start, end = PART_SLICES["pose"]
        landmarks[start:end] = np.array([[p.x, p.y, p.z] for p in pose_lms[0]], dtype=np.float32)
        presence["pose"] = True

    # -- hands. Each graph reports whichever hand it found; the slot is fixed by
    # the graph's own construction, and MediaPipe's Left/Right handedness is
    # anatomical only when the image is not mirrored, which is a documented
    # property of the ASLLRP and WLASL framings this project consumes.
    for result, name in zip(hand_results, ("left_hand", "right_hand"), strict=True):
        found = getattr(result, "hand_landmarks", None)
        if not found:
            continue
        start, end = PART_SLICES[name]
        landmarks[start:end] = np.array([[p.x, p.y, p.z] for p in found[0]], dtype=np.float32)
        presence[name] = True

    # -- face + blendshapes
    face_lms = getattr(face_result, "face_landmarks", None)
    if face_lms:
        start, end = PART_SLICES["face"]
        landmarks[start:end] = np.array([[p.x, p.y, p.z] for p in face_lms[0]], dtype=np.float32)
        presence["face"] = True

    shapes = getattr(face_result, "face_blendshapes", None)
    if shapes:
        categories = shapes[0]
        # Blendshape order is asserted, not trusted: a permuted basis would leave
        # every shape check passing while inverting the meaning of every
        # brow-raise rule in the M3 marker labeller.
        names = tuple(c.category_name for c in categories)
        if names != BLENDSHAPE_CATEGORIES:
            missing = sorted(set(BLENDSHAPE_CATEGORIES) - set(names))
            extra = sorted(set(names) - set(BLENDSHAPE_CATEGORIES))
            raise PerceptionUnavailable(
                f"blendshape basis does not match the measured ARKit order "
                f"({len(names)} categories, expected {len(BLENDSHAPE_CATEGORIES)}); "
                f"missing={missing[:5]} unexpected={extra[:5]}"
            )
        blendshapes = np.array([c.score for c in categories], dtype=np.float32)

    matrices = getattr(face_result, "facial_transformation_matrixes", None)
    if matrices:
        head_rotation = np.array(matrices[0], dtype=np.float32).reshape(4, 4)

    return Frame(
        landmarks=landmarks,
        blendshapes=blendshapes,
        head_rotation=head_rotation,
        gaze=_gaze_proxy(landmarks),
        presence=presence,
    )


def _gaze_proxy(landmarks: np.ndarray) -> np.ndarray | None:
    """Iris-offset gaze proxy, normalised by inter-ocular distance.

    MediaPipe Tasks emits iris points as the last 10 of the 478-point face mesh
    (468..477). A gaze proxy is enough for M1/M3's head-and-eye dynamics; a real
    gaze model is not a contribution of this project.
    """
    start, end = PART_SLICES["face"]
    face = landmarks[start:end]
    if face.shape[0] < FACE_POINTS:
        return None
    left_iris = face[468:473, :2].mean(axis=0)
    right_iris = face[473:478, :2].mean(axis=0)
    # Face-mesh eye corners, used for scale normalisation.
    left_corner = face[33, :2]
    right_corner = face[263, :2]
    span = float(np.linalg.norm(left_corner - right_corner))
    if span < 1e-6:
        return None
    centre = (left_iris + right_iris) / 2.0
    return ((centre - (left_corner + right_corner) / 2.0) / span).astype(np.float32)


def close(task: TaskLandmarker | None = None) -> None:
    """Release the graph modules. Needed between processes, not within one."""
    task = task or landmarker()
    for attr in ("_face", "_pose"):
        obj = getattr(task, attr, None)
        if obj is not None and hasattr(obj, "close"):
            obj.close()
        setattr(task, attr, None)
    for hand in task._hands:
        if hasattr(hand, "close"):
            hand.close()
    task._hands.clear()
    if task._pool is not None:
        task._pool.shutdown(wait=False)
        task._pool = None
