"""The prosody channel ``P``: how a sign is *produced*, not what it denotes.

Why these belong to affect rather than to grammar: the EmoSign Deaf annotators
named them themselves. Their free-text cue columns describe sign size, speed,
repetition, emphatic fingerspelling, pauses and overall intensity, and the eJSL
study independently found that adding hand motion improves emotion recognition
in signers. A brow raise is grammar; a sign delivered twice as fast and twice as
large is affect. Keeping them in a separate channel from the non-manual one is
what makes that separation testable rather than asserted.

Everything here is computed from the manual channel (hands and upper body) in
normalized coordinates, so the units are shoulder-widths per second and
shoulder-widths cubed - not pixels. A signer close to the lens and one far from
it must produce the same numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from seam.preprocess.normalize import NormalizeError

#: MediaPipe pose indices bounding the upper body used for the centroid.
POSE_LEFT_WRIST = 15
POSE_RIGHT_WRIST = 16
POSE_LEFT_SHOULDER = 11
POSE_RIGHT_SHOULDER = 12


@dataclass(frozen=True, slots=True)
class Prosody:
    """One clip's prosodic summary.

    All fields are scalar summaries over the whole clip, so two clips of the
    same gloss and different affect are comparable. The windowed values used by
    the audit live in :func:`windowed`.
    """

    speed: float
    peak_speed: float
    amplitude: float
    volume: float
    repetition: float
    pause_fraction: float
    pause_mean: float
    jerk: float
    sign_count: float
    active_fraction: float
    n_frames: int
    seconds: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def as_vector(self) -> np.ndarray:
        """Fixed-order vector, for the ``P`` input to the affect encoder."""
        return np.array(
            [
                self.speed,
                self.peak_speed,
                self.amplitude,
                self.volume,
                self.repetition,
                self.pause_fraction,
                self.pause_mean,
                self.jerk,
                self.sign_count,
                self.active_fraction,
            ],
            dtype=np.float32,
        )


PROSODY_FIELDS = (
    "speed",
    "peak_speed",
    "amplitude",
    "volume",
    "repetition",
    "pause_fraction",
    "pause_mean",
    "jerk",
    "sign_count",
    "active_fraction",
)


def hand_centroids(landmarks: np.ndarray, pose_slice: tuple[int, int]) -> np.ndarray:
    """Per-frame midpoint of the detected hands, in normalized coordinates.

    Returns shape (T, 2). Frames with no detected hand are NaN rather than zero:
    a centroid at the origin means "hands at the body centre", which is a real
    and different pose from "no hands detected", and conflating them would put a
    phantom pause in every clip.
    """
    start, end = pose_slice
    pose = landmarks[:, start:end, :2]
    out = np.full((landmarks.shape[0], 2), np.nan, dtype=np.float64)
    for wrist in (POSE_LEFT_WRIST, POSE_RIGHT_WRIST):
        pt = pose[:, wrist, :]
        ok = np.linalg.norm(pt, axis=1) > 1e-6
        out[ok] = 0.5 * (out[ok] + pt[ok]) if np.isfinite(out[ok]).any() else pt[ok]
    return out


def _fill_nan(x: np.ndarray) -> np.ndarray:
    """Forward/backward fill a (T, C) array, leaving all-NaN columns as zeros."""
    out = x.copy()
    for c in range(out.shape[1]):
        col = out[:, c]
        good = ~np.isnan(col)
        if not good.any():
            out[:, c] = 0.0
            continue
        idx = np.arange(len(col))
        out[:, c] = np.interp(idx, idx[good], col[good])
    return out


def per_frame_speed(centroids: np.ndarray, fps: float) -> np.ndarray:
    """Per-frame centroid displacement. NaN where the hands were not detected."""
    if fps <= 0:
        raise NormalizeError("fps must be positive")
    c = _fill_nan(centroids)
    if len(c) < 2:
        return np.zeros(len(c))
    step = np.linalg.norm(np.diff(c, axis=0), axis=1)
    return np.concatenate([[0.0], step]) * fps


def amplitude(centroids: np.ndarray) -> float:
    """Peak-to-peak excursion of the signing hand, in shoulder-widths.

    The Deaf annotators' "sign size". Bounding-box diagonal is used rather than
    a radius so that a long horizontal reach and a tall vertical reach both count
    as large.
    """
    c = _fill_nan(centroids)
    if len(c) == 0:
        return 0.0
    extent = c.max(axis=0) - c.min(axis=0)
    return float(np.linalg.norm(extent))


def volume(centroids: np.ndarray) -> float:
    """Axis-aligned volume of the signing-space bounding box, shoulder-widths^3."""
    c = _fill_nan(centroids)
    if len(c) == 0:
        return 0.0
    extent = np.clip(c.max(axis=0) - c.min(axis=0), 0.0, None)
    return float(np.prod(extent))


def repetition(centroids: np.ndarray, fps: float, *, min_period: float = 0.25) -> float:
    """Repetition count from autocorrelation peaks of the speed signal.

    The annotators' "repeated movements" and "emphasatic fingerspelling". The
    speed signal is mean-removed and autocorrelation is taken over it, so a slow
    steady sign scores 0 and a sign that cycles three times scores about 3.

    Peaks are required to clear a prominence threshold and be separated by at
    least ``min_period`` seconds, because the autocorrelation of a noisy
    derivative has many small local maxima that would otherwise each count as a
    repetition.
    """
    speed = per_frame_speed(centroids, fps)
    if len(speed) < 4:
        return 0.0
    signal = speed - speed.mean()
    denom = float((signal * signal).sum())
    if denom < 1e-12:
        return 0.0

    ac = np.correlate(signal, signal, mode="full")[len(signal) - 1 :] / denom
    min_lag = max(1, round(min_period * fps))
    if min_lag >= len(ac):
        return 0.0

    window = ac[min_lag:]
    if len(window) < 2:
        return 0.0

    # Local maxima above a fraction of the zero-lag value.
    thresh = 0.15
    count = 0
    for i in range(1, len(window) - 1):
        if window[i] > thresh and window[i] >= window[i - 1] and window[i] > window[i + 1]:
            count += 1
    return float(count)


def pause_stats(
    centroids: np.ndarray,
    fps: float,
    *,
    rel_threshold: float = 0.15,
    abs_floor: float = 0.02,
) -> tuple[float, float, np.ndarray]:
    """Return ``(pause_fraction, mean_pause_seconds, active_mask)``.

    A pause is a run of frames whose speed falls below
    ``max(rel_threshold * peak, abs_floor)``.

    Both terms are needed, and the property test that forced this is worth
    recording. A purely *relative* threshold is scale-free, which is what we
    wanted, but on a motionless clip the "peak" is tracker noise, so
    ``0.15 * noise_peak`` sits below the noise and every frame counts as active:
    a signer standing perfectly still measured 6.7% paused. The absolute floor,
    in shoulder-widths per second, is what makes stillness read as stillness.
    """
    speed = per_frame_speed(centroids, fps)
    if len(speed) == 0:
        return 0.0, 0.0, np.zeros(0, dtype=bool)
    peak = float(speed.max())
    if peak < 1e-9:
        return 1.0, len(speed) / fps, np.zeros(len(speed), dtype=bool)

    active = speed > max(rel_threshold * peak, abs_floor)
    n_pause = int((~active).sum())
    fraction = n_pause / len(speed)

    durations: list[float] = []
    run = 0
    for flag in active:
        if flag:
            if run:
                durations.append(run / fps)
            run = 0
        else:
            run += 1
    if run:
        durations.append(run / fps)
    mean_pause = float(np.mean(durations)) if durations else 0.0
    return fraction, mean_pause, active


def jerk(centroids: np.ndarray, fps: float, *, smooth: bool = True) -> float:
    """Mean magnitude of the third derivative of the hand centroid.

    Jerk is the standard smoothness measure in motor control: a sharp, angular
    movement has high jerk, and ASL affective prosody is documented to shorten and
    sharpen the movement path under anger. Reported as a mean rather than a max
    so one tracking glitch does not define the clip.

    **The trajectory is smoothed first, and that is not optional.** The third
    derivative amplifies noise by 1/dt^3, so on raw landmarks a linear ramp
    scored *higher* jerk than a square wave - the property test that checks
    "sharp motion has more jerk than smooth motion" failed with 4.67 against
    6.78, entirely because the ramp's third derivative was pure detector noise.
    Third-derivative features are meaningless without pre-smoothing.
    """
    c = _fill_nan(centroids)
    if len(c) < 4 or fps <= 0:
        return 0.0
    if smooth:
        from seam.preprocess.normalize import one_euro_filter

        # Explicit bandwidth, deliberately higher than the interactive default.
        # Sign movement lives in roughly the 0-5 Hz band; smoothing at the
        # default min_cutoff of 1 Hz attenuates a real 3 Hz repetition as hard as
        # it attenuates detector noise, and the feature then measures the filter
        # instead of the movement.
        smoothed = one_euro_filter(c.astype(np.float32), fps, min_cutoff=6.0, beta=0.02)
        c = _fill_nan(np.asarray(smoothed, dtype=np.float64))
        if len(c) < 4:
            return 0.0
    dt = 1.0 / fps
    d1 = np.diff(c, axis=0) / dt
    d2 = np.diff(d1, axis=0) / dt
    d3 = np.diff(d2, axis=0) / dt
    if len(d3) == 0:
        return 0.0
    return float(np.linalg.norm(d3, axis=1).mean())


def summarize(
    landmarks: np.ndarray,
    pose_slice: tuple[int, int],
    fps: float,
) -> Prosody:
    """Compute the full prosodic summary for one clip."""
    if fps <= 0:
        raise NormalizeError("fps must be positive")
    c = hand_centroids(landmarks, pose_slice)
    speed = per_frame_speed(c, fps)
    frac, mean_pause, active = pause_stats(c, fps)
    return Prosody(
        speed=float(speed.mean()) if len(speed) else 0.0,
        peak_speed=float(speed.max()) if len(speed) else 0.0,
        amplitude=amplitude(c),
        volume=volume(c),
        repetition=repetition(c, fps),
        pause_fraction=frac,
        pause_mean=mean_pause,
        jerk=jerk(c, fps),
        sign_count=float(_count_runs(active)),
        active_fraction=float(active.mean()) if len(active) else 0.0,
        n_frames=int(landmarks.shape[0]),
        seconds=landmarks.shape[0] / fps,
    )


def _count_runs(active: np.ndarray) -> int:
    if len(active) == 0:
        return 0
    return int(np.sum(active[1:] & ~active[:-1]) + (1 if active[0] else 0))


def windowed(
    landmarks: np.ndarray,
    pose_slice: tuple[int, int],
    fps: float,
    window: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window prosody.

    Returns ``(vectors, edges)`` where ``vectors`` is (n_windows, 10) and
    ``edges`` holds each window's left frame index. The audit needs the
    per-window values because the confound is a *segment-level* effect: the
    question is whether a grammatical marker shifts a model's output on the
    frames where the marker is visible, not on the clip as a whole.
    """
    from seam.preprocess.normalize import window_indices

    n = landmarks.shape[0]
    edges = window_indices(n, window, stride)
    if len(edges) == 0:
        return np.zeros((0, len(PROSODY_FIELDS)), dtype=np.float32), edges

    vectors = np.empty((len(edges), len(PROSODY_FIELDS)), dtype=np.float32)
    for i, start in enumerate(edges):
        chunk = landmarks[start : start + window]
        vectors[i] = summarize(chunk, pose_slice, fps).as_vector()
    return vectors, edges
