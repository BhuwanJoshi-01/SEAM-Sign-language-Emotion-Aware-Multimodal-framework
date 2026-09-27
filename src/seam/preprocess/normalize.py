"""The single shared normalization path.

This module exists because of a specific, measured failure. In the prior ISLR
system the serving path called the shared normalizer with ``mask=None`` instead
of the presence mask, and whole-clip top-1 accuracy fell from 52.1% to 33.6% with
no error raised anywhere. The number was plausible, the pipeline was green, and
nothing was wrong except one argument.

So: one normalizer, one signature, used by training, extraction, evaluation and
serving alike, and a test that asserts the mask is actually threaded through.

Every function here is pure numpy with no torch dependency, because the browser
path and the serving path both need it and neither should be forced to import a
deep-learning framework to smooth a landmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: MediaPipe pose indices used for the normalization frame.
POSE_NOSE = 0
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12


class NormalizeError(ValueError):
    """Raised when a sequence cannot be normalized at all."""


@dataclass(frozen=True, slots=True)
class NormParams:
    """The transform applied to a sequence, kept so it can be inverted or logged.

    ``center`` and ``scale`` are per-frame (T, 2) and (T,), not scalars: a signer
    who moves toward or away from the camera gets a different shoulder span on
    every frame, and collapsing that to one number reintroduces exactly the
    scale dependence the normalization exists to remove.
    """

    center: np.ndarray
    scale: np.ndarray


def interpolate_gaps(
    landmarks: np.ndarray,
    presence: np.ndarray,
    *,
    max_gap: int = 12,
    part_slices: dict[str, tuple[int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill short runs of undetected points, in place of the zeros.

    ``presence`` is per-frame-per-part, so it says which of the four landmark
    blocks were detected, not which individual joints are missing. The convention
    here is therefore: a block that was never detected is left at zero rather
    than interpolated, and inside a detected block a short run of degenerate
    points is filled.

    Interpolating a never-detected block would fabricate a hand in mid-air, and
    the prosody channel would then measure the speed of that fabrication. Long
    gaps are also left alone: beyond ``max_gap`` frames the motion between the
    endpoints is no longer a reasonable stand-in.

    ``part_slices`` maps a presence column to its landmark row range. It is
    required, and the reason is a bug this signature exists to prevent: presence
    column 1 is the left hand but its landmark rows are 33..54, so indexing the
    landmark array with the presence column fills the wrong block while every
    shape assertion still passes.

    Returns the filled array and the per-point validity actually used, so a
    caller can tell repaired points from real ones.
    """
    if landmarks.ndim != 3 or landmarks.shape[-1] != 3:
        raise NormalizeError(f"expected (T, P, 3), got {landmarks.shape}")
    if part_slices is None:
        from seam.perception.tasks_api import PART_SLICES

        part_slices = PART_SLICES

    out = landmarks.astype(np.float32, copy=True)
    n_frames = out.shape[0]

    valid = presence.astype(bool)
    if valid.shape[0] != n_frames:
        raise NormalizeError(f"presence has {valid.shape[0]} frames, landmarks has {n_frames}")
    if valid.shape[1] != len(part_slices):
        raise NormalizeError(
            f"presence has {valid.shape[1]} parts, part_slices has {len(part_slices)}"
        )

    for column, (_name, (start, end)) in enumerate(part_slices.items()):
        mask = valid[:, column]
        if not mask.any():
            continue

        block = out[:, start:end, :]
        # A point is present when it is not the degenerate zero that means "not
        # measured". Inside a detected block, individual joints can still be
        # missing, and those are the ones worth interpolating.
        finite = np.isfinite(block).all(axis=-1) & (np.abs(block).sum(axis=-1) > 0)
        usable = mask[:, None] & finite
        if not usable.any():
            continue

        span = np.flatnonzero(usable.any(axis=1))
        for coord in range(3):
            values = block[:, :, coord]
            per_point = np.empty((n_frames, end - start), dtype=np.float32)
            for j in range(end - start):
                col = np.flatnonzero(usable[:, j])
                if len(col) == 0:
                    per_point[:, j] = 0.0
                elif len(col) == 1:
                    per_point[:, j] = values[col[0], j]
                else:
                    per_point[:, j] = np.interp(np.arange(n_frames), col, values[col, j])
            block[:, :, coord] = per_point

        # Reject fills spanning a gap longer than max_gap, and never extend
        # beyond the detected span.
        for lo, hi in _runs(~mask):
            if lo > 0 and hi < n_frames - 1 and (hi - lo) > max_gap:
                block[lo:hi, :, :] = 0.0
        block[: span[0], :, :] = 0.0
        block[span[-1] + 1 :, :, :] = 0.0

        out[:, start:end, :] = block

    valid_points = (np.abs(out).sum(axis=-1) > 0) & np.isfinite(out).all(axis=-1)
    out[~np.isfinite(out)] = 0.0
    return out, valid_points


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as half-open ``[start, end)`` spans."""
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def normalize_pose(
    landmarks: np.ndarray,
    pose_slice: tuple[int, int],
    *,
    rotate: bool = False,
) -> tuple[np.ndarray, NormParams]:
    """Center on the shoulder midpoint and scale by shoulder width.

    This is what makes the features comparable across signers and across
    camera distances: a signer close to the lens and one far from it produce the
    same normalized trajectory, and a signer who shifts in frame produces the
    same one as a signer who does not.

    Degenerate cases are handled rather than guarded against by an exception:
    a frame where the shoulders are not both detected falls back to the nose, and
    a frame where the shoulder span collapses to nothing keeps a scale of 1
    rather than dividing by ~0 and producing inf.
    """
    start, end = pose_slice
    pose = landmarks[:, start:end, :2].astype(np.float64)

    left = pose[:, POSE_LEFT_SHOULDER, :]
    right = pose[:, POSE_RIGHT_SHOULDER, :]
    nose = pose[:, POSE_NOSE, :]

    both = (left.sum(axis=1) > 0) & (right.sum(axis=1) > 0)
    center = np.where(
        both[:, None],
        (left + right) / 2.0,
        np.where((nose.sum(axis=1) > 0)[:, None], nose, np.nan),
    )

    # A frame with no usable anchor is carried by the nearest frame that has one.
    if np.isnan(center).any():
        good = ~np.isnan(center[:, 0])
        if not good.any():
            raise NormalizeError("no frame has a usable pose anchor")
        idx = np.arange(len(center))
        for c in range(2):
            center[:, c] = np.interp(idx, idx[good], center[good, c])

    width = np.linalg.norm(left - right, axis=1)
    scale = np.where(both, width, np.nan)
    if np.isnan(scale).all():
        scale = np.ones(len(landmarks))
    else:
        good = ~np.isnan(scale)
        idx = np.arange(len(scale))
        scale = np.interp(idx, idx[good], scale[good])
    # A collapsed span means the pose was not really detected.
    scale = np.where(scale < 1e-6, np.nan, scale)
    if np.isnan(scale).any():
        good = ~np.isnan(scale)
        idx = np.arange(len(scale))
        scale = np.interp(idx, idx[good], scale[good])
    scale = np.where(scale < 1e-6, 1.0, scale)

    params = NormParams(center=center, scale=scale)
    out = landmarks.astype(np.float32, copy=True)
    out[:, :, :2] = (landmarks[:, :, :2] - center[:, None, :]) / scale[:, None, None]
    if rotate:
        out[:, :, :2] = _align_tilt(out[:, start:end, :2])
    return out, params


def _align_tilt(pose_xy: np.ndarray) -> np.ndarray:
    """Rotate each frame so the shoulder line is horizontal.

    Optional, and off by default: it removes a nuisance variable (lean) but also
    removes a real one (torso tilt is communicative), so enabling it is an
    ablation and not a default.
    """
    left = pose_xy[:, POSE_LEFT_SHOULDER, :]
    right = pose_xy[:, POSE_RIGHT_SHOULDER, :]
    vec = right - left
    angle = np.arctan2(vec[:, 1], vec[:, 0])
    c, s = np.cos(-angle), np.sin(-angle)
    out = pose_xy.copy()
    out[..., 0] = pose_xy[..., 0] * c[:, None] - pose_xy[..., 1] * s[:, None]
    out[..., 1] = pose_xy[..., 0] * s[:, None] + pose_xy[..., 1] * c[:, None]
    return out


def one_euro_filter(
    x: np.ndarray,
    fps: float,
    *,
    min_cutoff: float = 1.0,
    beta: float = 0.007,
    d_cutoff: float = 1.0,
) -> np.ndarray:
    """Causal One Euro filter over the leading axis of ``x``.

    Chosen over a moving average because it trades jitter for lag adaptively: at
    low speed it smooths hard, at high speed it backs off, so a fast sign is not
    smeared while a still signer is not noisy. Casiez et al., CHI 2012.

    Strictly causal - frame ``t`` depends only on frames ``<= t`` - which is what
    makes it safe for the live path. A centred smoother would peek at the future
    and quietly break any latency claim.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[0] < 2:
        return arr.astype(np.float32)

    if fps <= 0:
        raise NormalizeError("fps must be positive")

    tau_d = 1.0 / (2 * np.pi * d_cutoff)
    dt = 1.0 / fps
    alpha_d = 1.0 / (1.0 + tau_d / dt)

    def alpha_for(cutoff: float) -> float:
        """One Euro's exponential smoothing weight for an instantaneous cutoff."""
        t = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + t / dt)

    flat = arr.reshape(arr.shape[0], -1)
    out = np.empty_like(flat)
    prev = flat[0].copy()
    out[0] = prev
    prev_deriv = np.zeros_like(prev)

    for i in range(1, flat.shape[0]):
        deriv = (flat[i] - prev) * fps
        # Smooth the derivative first, then let its magnitude raise the cutoff.
        smooth_deriv = alpha_d * deriv + (1.0 - alpha_d) * prev_deriv
        cutoff = min_cutoff + beta * np.abs(smooth_deriv)
        alpha = alpha_for(cutoff)
        out[i] = alpha * flat[i] + (1.0 - alpha) * prev
        prev = out[i]
        prev_deriv = smooth_deriv

    return out.reshape(arr.shape).astype(np.float32)


def resample_indices(
    n_frames: int,
    src_fps: float,
    dst_fps: float,
    *,
    max_frames: int | None = None,
) -> np.ndarray:
    """Frame indices that resample ``src_fps`` to ``dst_fps`` by nearest neighbour.

    12 fps is the project's default because encoder self-attention is O(n^2):
    halving the rate quarters that cost, and the published compact-SLT recipe
    measures the quality cost as BLEU-4 10.06 at 24 fps against 9.53 at 12.
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=np.int64)
    if src_fps <= 0 or dst_fps <= 0:
        raise NormalizeError("frame rates must be positive")

    duration = n_frames / src_fps
    n_out = max(1, round(duration * dst_fps))
    if max_frames is not None:
        n_out = min(n_out, max_frames)
    times = np.arange(n_out) / dst_fps
    idx = np.round(times * src_fps).astype(np.int64)
    return np.clip(idx, 0, n_frames - 1)


def window_indices(n_frames: int, window: int, stride: int) -> np.ndarray:
    """Left edges of sliding windows, or an empty array when the clip is too short."""
    if n_frames < window:
        return np.zeros(0, dtype=np.int64)
    return np.arange(0, n_frames - window + 1, stride, dtype=np.int64)
