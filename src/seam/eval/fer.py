"""Non-signer facial-emotion-recognition baselines, and the audit harness.

The claim under test is that a facial-emotion model trained on non-signers
reads sign language the way a hearing non-signer does: it cannot tell a
grammatical brow raise from an emotional one. Testing that needs a model whose
training distribution is *known* to be non-signers, so these are trained here on
RAF-DB - posed, in-the-wild expressions of hearing adults - and their accuracy
on that distribution is reported alongside the audit result. A model nobody can
score on its own training distribution cannot be used to accuse anybody of
anything.

Design choices that matter for the audit rather than for accuracy:

* **Face crops, not landmarks.** The models see pixels, because that is what a
  hearing non-signer and every published MLLM baseline sees. A landmark-based
  model would be a weaker and less faithful stand-in for the thing being audited.
* **The same preprocessing at fit and at inference.** A FER model evaluated on
  differently-cropped input measures the crop, not the bias. One function,
  :func:`face_crop`, produces both.
* **Probability mass, not argmax.** The audit needs a continuous
  "how negative does this look" quantity. An argmax discards exactly the
  information the claim is about.
* **Negative mass is the read-out, defined once.** ``negative_mass`` is the total
  probability on the four unambiguously negative classes. Aggregating a
  continuous score is what makes the paired test well defined; per-class
  confusion matrices are reported alongside so the aggregation is never the only
  evidence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from seam.logging import get
from seam.perception.tasks_api import PART_SLICES

log = get(__name__)

#: RAF-DB's seven classes, in the dataset's own declared order.
EMOSIGN_LABELS: tuple[str, ...] = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
)

#: Classes that count as unambiguously negative affect for the read-out.
NEGATIVE_LABELS: tuple[str, ...] = ("anger", "disgust", "fear", "sadness")
#: Classes that count as positive. ``neutral`` is in neither: the audit asks
#: whether a marker pushes a model *toward* negative affect, and folding neutral
#: into either side would answer a different question.
POSITIVE_LABELS: tuple[str, ...] = ("happiness", "surprise")

NEG_IDX = np.array([EMOSIGN_LABELS.index(n) for n in NEGATIVE_LABELS])
POS_IDX = np.array([EMOSIGN_LABELS.index(n) for n in POSITIVE_LABELS])
NEUTRAL_IDX = EMOSIGN_LABELS.index("neutral")

#: Face crop geometry. Small on purpose: the audit runs a FER forward pass per
#: window per clip, and this has to fit inside the 4 GB / <400 ms envelope.
CROP_SIZE = 112


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def face_bbox(landmarks: np.ndarray, present: bool, *, margin: float = 0.35) -> tuple | None:
    """Axis-aligned face bounding box from the 478-point face mesh.

    Returns ``None`` when the face was not detected, so the caller records a
    missing face rather than cropping the whole frame - a full-frame crop fed to
    a FER model produces confident nonsense, which is a different bug from the
    one being audited.
    """
    if not present:
        return None
    start, end = PART_SLICES["face"]
    face = np.asarray(landmarks[start:end, :2], dtype=np.float64)
    if face.size == 0 or not np.isfinite(face).all():
        return None
    lo = face.min(axis=0)
    hi = face.max(axis=0)
    size = hi - lo
    if not np.isfinite(size).all() or (size <= 1e-4).any():
        return None
    # ASL signing keeps the hands near the face, so a tight crop is not enough
    # context for the model to be a fair stand-in for a viewer.
    pad = size * margin
    lo = np.maximum(lo - pad, 0.0)
    hi = np.minimum(hi + pad, 1.0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def face_crop(
    frame_rgb: np.ndarray,
    landmarks: np.ndarray,
    present: bool,
    size: int = CROP_SIZE,
) -> np.ndarray | None:
    """Crop and resize the face region of one RGB frame, as **uint8**.

    Shared by fitting and inference on purpose, and uint8 on purpose: a FER
    model scored on a differently-preprocessed input is measuring the
    preprocessing, and the first version of this pipeline returned float32 in
    [0, 1] here while the fitting path handed over uint8 from PIL - so
    ``equalizeHist`` raised on the inference path only. One dtype, one code
    path, asserted in ``normalize_face``.
    """
    box = face_bbox(landmarks, present)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    h, w = frame_rgb.shape[:2]
    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = max(int(x1 * w), px0 + 1), max(int(y1 * h), py0 + 1)

    import cv2

    crop = frame_rgb[py0:py1, px0:px1]
    if crop.size == 0:
        return None
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def normalize_face(crop: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    """Standard FER preprocessing: grayscale, histogram equalisation, z-score.

    This function exists because the first version of the audit produced a
    meaningless result and the reason was diagnosable. Trained on RAF-DB's tight
    MediaPipe face boxes and applied to sign-video face crops, all three FER
    models collapsed to a **single predicted class across all 200 EmoSign clips**
    (sadness 200/200, fear 200/200, anger 200/200) with Spearman correlation to
    true sentiment of +0.03, -0.09 and +0.16. Per-frame probability spread
    *within* a clip was 0.0002 while the spread *across* clips was 0.283 - one
    output per clip. A model that emits one answer per clip is reading lighting,
    background and resolution, not a face.

    Grayscale plus histogram equalisation plus per-image standardisation removes
    exactly those nuisance factors and is the conventional FER front end. It is
    applied by **one function on both the fitting and the inference path**: a
    model scored on differently-preprocessed input is measuring the
    preprocessing, which is the bug the prior ISLR system lost 19 accuracy points
    to.
    """
    import cv2

    crop = np.asarray(crop)
    if crop.dtype != np.uint8:
        raise TypeError(
            f"normalize_face expects a uint8 crop, got {crop.dtype}. The fitting "
            f"and inference paths must hand over the same dtype; converting here "
            f"would hide a real train/serve skew."
        )
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    crop = cv2.equalizeHist(crop)
    arr = crop.astype(np.float32)
    arr -= arr.mean()
    sd = arr.std()
    if sd > 1e-6:
        arr /= sd
    # Three identical channels: the model keeps a 3-channel stem, and the
    # evidence says colour is exactly the nuisance we are removing.
    return np.repeat(arr[None, :, :], 3, axis=0)


def preprocess_faces(crops: Iterable[np.ndarray], size: int = CROP_SIZE) -> np.ndarray:
    """Stack crops into a (N, 3, size, size) normalized tensor."""
    out = [normalize_face(c, size) for c in crops]
    if not out:
        return np.zeros((0, 3, size, size), dtype=np.float32)
    return np.ascontiguousarray(np.stack(out))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FerModelSpec:
    """A named FER baseline."""

    name: str
    seed: int
    width: int = 32
    blocks: int = 3
    dropout: float = 0.3


def build_model(spec: FerModelSpec, n_classes: int = len(EMOSIGN_LABELS)) -> object:
    """A compact CNN, written here rather than imported.

    No pretrained backbone: ImageNet weights are neither signer nor non-signer
    data, and importing one would make "trained only on non-signers" untrue in a
    way that is very hard to see in the results. The architecture is small on
    purpose - the audit's claim is about the training distribution, not about
    squeezing the last point out of RAF-DB.
    """
    from torch import nn

    layers: list[nn.Module] = []
    in_ch = 3
    for _ in range(spec.blocks):
        layers += [
            nn.Conv2d(in_ch, spec.width, 3, padding=1, bias=False),
            nn.BatchNorm2d(spec.width),
            nn.ReLU(inplace=True),
            nn.Conv2d(spec.width, spec.width, 3, padding=1, bias=False),
            nn.BatchNorm2d(spec.width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        ]
        in_ch = spec.width
    return nn.Sequential(
        *layers,
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(spec.dropout),
        nn.Linear(in_ch, n_classes),
    )


# ---------------------------------------------------------------------------
# Read-out
# ---------------------------------------------------------------------------


def negative_mass(probs: np.ndarray) -> np.ndarray:
    """Total probability on the four negative classes.

    ``probs`` is (..., n_classes). This is the audit's read-out: a continuous
    "how negative does a non-signer model think this looks", which is what makes
    a paired test well defined. Per-class breakdowns are reported alongside so
    the aggregation is never the only evidence.
    """
    return np.asarray(probs)[..., NEG_IDX].sum(axis=-1)


def positive_mass(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs)[..., POS_IDX].sum(axis=-1)


def neutral_mass(probs: np.ndarray) -> np.ndarray:
    return np.asarray(probs)[..., NEUTRAL_IDX]


def read_outs(probs: np.ndarray) -> dict[str, np.ndarray]:
    """All three read-outs plus the argmax label, for one batch."""
    probs = np.asarray(probs)
    return {
        "negative": negative_mass(probs),
        "positive": positive_mass(probs),
        "neutral": neutral_mass(probs),
        "argmax": probs.argmax(axis=-1),
    }


def describe() -> dict[str, object]:
    """Machine-readable description of the audited models, for the appendix."""
    return {
        "labels": list(EMOSIGN_LABELS),
        "negative_labels": list(NEGATIVE_LABELS),
        "positive_labels": list(POSITIVE_LABELS),
        "neutral_excluded": True,
        "training_data": "RAF-DB (posed, in-the-wild expressions of hearing adults)",
        "training_data_gated": False,
        "pretrained_backbone": False,
        "crop_size": CROP_SIZE,
        "crop_margin": 0.35,
        "input": "RGB face crop, [0, 1]",
        "readout": "negative = total probability on anger+disgust+fear+sadness",
        "non_signer_only": True,
    }
