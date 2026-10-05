"""The confound audit: does a non-signer FER model read grammar as negative affect?

The design, and why each piece is there:

**Within-clip matched pairs, not between-clip comparison.** Marker-bearing
windows are paired with marker-free windows *from the same clip*, matched on
prosodic amplitude and speed. A between-clip comparison would confound the
marker with everything else that differs between signs - signer, handshape,
movement energy, background. Holding the clip fixed removes all of it, because
the two windows differ only in whether the marker is present.

**Negative probability mass, not argmax.** The claim is about a *direction* of
misreading, so the read-out has to be continuous. An argmax throws away exactly
the evidence.

**Bootstrap over clips, not over windows.** Windows inside one clip are strongly
correlated; treating 4,000 windows as 4,000 independent observations would
shrink the confidence interval by roughly the square root of the windows per
clip and manufacture significance. The resampling unit is the clip.

**A control marker.** ``mouth_positive`` is included in the marker set and
expected to behave *differently*. If every marker shifted the model the same way
the effect would be "any facial movement reads as negative", which is a much
weaker and less interesting claim, and the control is what distinguishes the two.

**A null model.** A uniform-random classifier is scored on the same windows. It
must show no bias by construction, which is the check that the pipeline does not
manufacture a shift out of the matching itself.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from seam.eval import fer
from seam.features import markers as M
from seam.features import prosody as P
from seam.logging import get
from seam.perception.extract import ClipMeta
from seam.perception.tasks_api import PART_SLICES
from seam.preprocess import normalize as N

log = get(__name__)

#: Window length and stride, in frames *at* ``TARGET_FPS``.
#:
#: These are not free parameters and the first guess was wrong. T=64 was carried
#: over from the continuous-SLT design, where a window is a phrase. WLASL is
#: *isolated* signs: measured over 400 clips, the median is 73 native frames, and
#: at 12 fps that is **33 frames**. T=64 therefore produced **zero** windows on
#: every clip in the corpus, and the audit reported an empty table rather than
#: failing. T=24 is 2.0 s, long enough to contain a brow raise or a head shake
#: with context and short enough that a typical sign yields more than one window -
#: which is what the within-clip contrast requires. Coverage: T=24 gives 76% of
#: clips at least one full window and T=16 gives 94%.
WINDOW = 24
STRIDE = 8
TARGET_FPS = 12.0

#: Below this many distinct clips a bootstrap interval is not reported. The
#: resampling unit is the clip, so a handful of clips cannot support an interval
#: at all, and reporting the degenerate one is worse than reporting nothing.
MIN_CLUSTERS = 10
POSE_SLICE = PART_SLICES["pose"]


@dataclass(slots=True)
class Window:
    """One analysis unit: a window of a clip, with everything the audit needs."""

    clip: str
    index: int
    start: int
    n_frames: int
    #: bool per MARKERS entry
    markers: list[bool]
    #: prosody vector, aligned with prosody.PROSODY_FIELDS
    prosody: list[float]
    #: FER class probabilities, or None when no face was visible
    probs: list[float] | None
    #: True when the FER read-out for this window is usable at all
    scored: bool
    #: What each entry of ``markers`` refers to. The heuristic visual markers by default;
    #: the human-annotated ones when the windows were built from SignStream events.
    names: tuple[str, ...] = M.MARKERS
    #: Markers this window is neither clearly bearing nor clearly free of - a human
    #: event that covers part of the window. Such a window is on neither side of the
    #: contrast for that marker. Always empty for heuristic markers.
    partial: tuple[str, ...] = ()

    def marker(self, name: str) -> bool:
        return bool(self.markers[self.names.index(name)])

    def free_of(self, name: str) -> bool:
        """Usable as the marker-free side of a pair: not bearing it, and not ambiguous."""
        return not self.marker(name) and name not in self.partial

    def prob_vector(self) -> np.ndarray:
        """The window's FER distribution, or zeros when it is not scorable.

        Callers that reached this point through ``scored`` already hold a real
        vector; the zero fallback exists so that a downstream aggregation over a
        mixed batch cannot raise. Zeros contribute nothing to a mean of real
        probabilities, so the fallback is harmless rather than a silent
        substitution - and it is only reachable when ``scored`` is False, which
        the matching and bootstrap both filter on.
        """
        if self.probs is None:
            return np.zeros(len(fer.EMOSIGN_LABELS), dtype=np.float64)
        return np.asarray(self.probs, dtype=np.float64)


@dataclass(slots=True)
class AuditStats:
    """Paired statistics for one marker and one model."""

    marker: str
    model: str
    n_pairs: int
    mean_shift: float
    ci_low: float
    ci_high: float
    effect_size: float
    p_value: float
    per_class_shift: dict[str, float] = field(default_factory=dict)
    #: Distinct clips contributing pairs. The bootstrap's effective sample size.
    n_clips: int = 0
    #: Smallest |shift| this design could detect at 80% power, alpha 0.05, two
    #: sided. Reported with every result because a null without its power is not a
    #: finding, it is an absence of measurement.
    mde: float = float("nan")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# FER scoring over a clip's video
# ---------------------------------------------------------------------------


def score_clip_faces(
    video_path: Path,
    landmarks: np.ndarray,
    presence: np.ndarray,
    model: Any,
    *,
    every_n: int = 1,
) -> np.ndarray:
    """Per-frame FER probabilities for one clip, shape (T, n_classes).

    A second decode pass over the video, not over the landmarks. The shards store
    geometry, not pixels, and a FER model has to see pixels to be a fair
    stand-in for a hearing non-signer. Decoding is roughly seven times cheaper
    than landmark extraction, so the pass is affordable, and only the resulting
    probabilities are kept.
    """
    import cv2
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((0, len(fer.EMOSIGN_LABELS)), dtype=np.float32)

    # The shard is a prefix of the video - extraction caps frames per clip - so
    # the loop is bounded by whichever runs out first. Reading past the landmarks
    # is not merely wasteful, it raises.
    n_available = min(len(landmarks), len(presence))
    out: list[np.ndarray] = []
    index = 0
    while index < n_available:
        ok, bgr = cap.read()
        if not ok:
            break
        if index % every_n == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            crop = fer.face_crop(
                rgb,
                np.asarray(landmarks[index]),
                bool(presence[index][3]),
            )
            if crop is not None:
                out.append(crop)
            else:
                out.append(None)  # type: ignore[arg-type]
        index += 1
    cap.release()

    if not out:
        return np.zeros((0, len(fer.EMOSIGN_LABELS)), dtype=np.float32)

    keep = [(i, c) for i, c in enumerate(out) if c is not None]
    probs = np.zeros((len(out), len(fer.EMOSIGN_LABELS)), dtype=np.float32)
    if keep:
        x = fer.preprocess_faces([c for _, c in keep])
        with torch.no_grad():
            logits = model(torch.from_numpy(x))
            p = torch.softmax(logits, dim=1).numpy()
        for (i, _), row in zip(keep, p, strict=True):
            probs[i] = row
    return probs


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------


def build_windows(
    clip_key: str,
    arrays: dict[str, np.ndarray],
    meta: ClipMeta,
    face_probs: np.ndarray,
    *,
    thresholds: M.MarkerThresholds = M.DEFAULT_THRESHOLDS,
    window: int = WINDOW,
    stride: int = STRIDE,
    frame_markers: Mapping[str, np.ndarray] | None = None,
    bearing_min: float = 0.5,
) -> list[Window]:
    """Turn one clip's shard plus its FER probabilities into analysis windows.

    The shard is resampled to ``TARGET_FPS`` first, because window length is a
    property of the analysis rate: a 64-frame window is 2.7 s at 24 fps and 5.3 s
    at 12 fps, and mixing the two would compare different spans of signing.

    ``frame_markers`` replaces the heuristic visual markers with per-frame boolean
    tracks on the clip's *native* frames - in practice the human SignStream
    annotations, already mapped to the clip's frame rate. A window then bears a marker
    when the track covers at least ``bearing_min`` of it, is free of the marker when the
    track does not touch it at all, and is ``partial`` - on neither side of the contrast
    - in between. Everything else about the window (resampling, normalisation, prosody,
    the scorability rule) is the same code, so a result on continuous signing and one on
    isolated signs differ in the marker source and the window length and nothing else.
    """
    landmarks = arrays["landmarks"].astype(np.float32)
    blendshapes = arrays["blendshapes"].astype(np.float64)
    presence = arrays["presence"].astype(bool)
    rotation = arrays.get("head_rotation")

    n = landmarks.shape[0]
    src_fps = meta.native_fps or 25.0
    idx = N.resample_indices(n, src_fps, TARGET_FPS, max_frames=None)
    landmarks, blendshapes = landmarks[idx], blendshapes[idx]
    presence = presence[idx]
    if rotation is not None:
        rotation = rotation[idx]

    landmarks, _ = N.interpolate_gaps(landmarks, presence, max_gap=6)
    landmarks, _ = N.normalize_pose(landmarks, POSE_SLICE)
    probs = (
        face_probs[idx]
        if len(face_probs)
        else np.zeros((len(idx), len(fer.EMOSIGN_LABELS)), np.float32)
    )

    names: tuple[str, ...] = M.MARKERS
    tracks: dict[str, np.ndarray] = {}
    if frame_markers is not None:
        names = tuple(frame_markers)
        for name, track in frame_markers.items():
            track = np.asarray(track, dtype=bool)
            if len(track) != n:
                raise ValueError(
                    f"{clip_key}: marker track {name!r} has {len(track)} frames, the clip has {n}"
                )
            tracks[name] = track[idx]
    sig = M.signals(blendshapes, rotation, fps=TARGET_FPS, thresholds=thresholds)
    edges = N.window_indices(len(idx), window, stride)
    if len(edges) == 0:
        # A clip shorter than one window is its own window. For an isolated sign
        # that is the natural unit anyway, and dropping the clip would bias the
        # corpus toward long signs - which is exactly where a marker is least
        # likely to appear and disappear within the clip.
        edges = np.zeros(1, dtype=np.int64)

    windows: list[Window] = []
    for i, start in enumerate(edges):
        sl = slice(int(start), min(int(start) + window, len(idx)))
        width = sl.stop - sl.start
        chunk = landmarks[sl]
        # The centroid statistic needs hands, not a filled-in guess, so the
        # window's presence is required rather than assumed.
        hands_ok = bool(presence[sl, 1].any() or presence[sl, 2].any())
        pv = (
            P.summarize(chunk, POSE_SLICE, TARGET_FPS).as_vector()
            if hands_ok
            else np.zeros(len(P.PROSODY_FIELDS), dtype=np.float32)
        )
        wp = probs[sl]
        face_frac = float(presence[sl, 3].mean())
        # A window is scorable when the face was visible for most of it. A FER
        # probability averaged over a window where the face appeared in 20% of
        # frames is a measurement of the other 80%.
        scored = face_frac >= 0.5
        row = wp.mean(axis=0) if scored else None
        if frame_markers is None:
            fired = [bool(x) for x in M.window_fired(sig, edges[i : i + 1], window, thresholds)[0]]
            partial: tuple[str, ...] = ()
        else:
            cover = {name: float(tracks[name][sl].mean()) for name in names}
            fired = [cover[name] >= bearing_min for name in names]
            partial = tuple(name for name in names if 0.0 < cover[name] < bearing_min)
        windows.append(
            Window(
                clip=clip_key,
                index=i,
                start=int(start),
                n_frames=width,
                markers=fired,
                prosody=[float(v) for v in pv],
                probs=None if row is None else [float(v) for v in row],
                scored=scored and hands_ok,
                names=names,
                partial=partial,
            )
        )
    return windows


# ---------------------------------------------------------------------------
# Matching and statistics
# ---------------------------------------------------------------------------


def match_windows(
    windows: list[Window],
    marker: str,
    *,
    tolerance: float = 1.0,
) -> list[tuple[Window, Window]]:
    """Pair each marker-bearing window with the closest marker-free window in the same clip.

    Matching is on amplitude and speed, the two prosodic dimensions most likely
    to co-occur with a facial marker and therefore the ones that could produce
    the effect with no marker present at all. Anything matched more loosely
    leaves that path open.

    The tolerance is in units of the **pooled** standard deviation across all
    scorable windows, not a per-clip one. A clip-local scale makes the tolerance
    mean a different thing in every clip - a clip with three marker-free windows
    has a scale estimated from three samples, so its "0.5 sd" is noise - and it
    silently drops most pairs in exactly the clips with fewest windows. Pooling
    gives a stable yardstick estimated from thousands of windows, so the same
    tolerance applies everywhere.

    A marker-bearing window with no partner inside ``tolerance`` is dropped
    rather than paired with a bad match: an unpaired window is missing data, and
    forcing a pair would bias the difference toward zero. The default of 1.0
    pooled sd on the 2-D distance is deliberately strict; how many windows it
    drops is reported by the caller, because a matching rate that collapses is
    itself a finding about whether the marker is confounded with prosody.
    """
    amp_i, spd_i = P.PROSODY_FIELDS.index("amplitude"), P.PROSODY_FIELDS.index("speed")

    scorable = [w for w in windows if w.scored]
    if not scorable:
        return []
    all_amp = np.array([w.prosody[amp_i] for w in scorable])
    all_spd = np.array([w.prosody[spd_i] for w in scorable])
    scale = np.array([max(all_amp.std(), 1e-9), max(all_spd.std(), 1e-9)])

    by_clip: dict[str, list[Window]] = {}
    for w in scorable:
        by_clip.setdefault(w.clip, []).append(w)

    pairs: list[tuple[Window, Window]] = []
    for group in by_clip.values():
        marked = [w for w in group if w.marker(marker)]
        clean = [w for w in group if w.free_of(marker)]
        if not marked or not clean:
            continue
        amp = np.array([w.prosody[amp_i] for w in clean])[None, :]
        spd = np.array([w.prosody[spd_i] for w in clean])[None, :]
        for m in marked:
            # (n_clean, 2): standardized distance on each covariate.
            d = np.stack(
                [
                    (m.prosody[amp_i] - amp[0]) / scale[0],
                    (m.prosody[spd_i] - spd[0]) / scale[1],
                ],
                axis=1,
            )
            dist = np.linalg.norm(d, axis=1)
            j = int(np.argmin(dist))
            if dist[j] <= tolerance:
                pairs.append((m, clean[j]))
    return pairs


def bootstrap_shift(
    pairs: list[tuple[Window, Window]],
    label: str,
    *,
    n_boot: int = 4000,
    seed: int = 0,
) -> AuditStats:
    """Paired mean shift in negative probability mass, with a cluster bootstrap CI.

    Resampling is over *clips*, not pairs: windows within a clip are correlated
    and treating them as independent would shrink the interval by roughly
    sqrt(windows per clip) and turn nothing into significance.
    """
    if not pairs:
        return AuditStats(label, "n/a", 0, 0.0, 0.0, 0.0, 0.0, 1.0)

    diffs = np.array(
        [
            float(fer.negative_mass(a.prob_vector()) - fer.negative_mass(b.prob_vector()))
            for a, b in pairs
        ]
    )
    clips = np.array([a.clip for a, _ in pairs])
    uniq = np.unique(clips)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.flatnonzero(clips == c) for c in pick])
        boots[b] = diffs[idx].mean()

    point = float(diffs.mean())
    n_clusters = len(uniq)

    if n_clusters < MIN_CLUSTERS:
        # A cluster bootstrap over one or two clips has no variance to estimate:
        # every resample draws the same cluster, so the interval collapses onto
        # the point estimate and a *uniform* null model comes out "significant".
        # That is not a finding, it is a degenerate interval, and reporting it as
        # one is how a null result gets mistaken for a positive one.
        return AuditStats(
            marker=label,
            model="all",
            n_pairs=len(pairs),
            mean_shift=point,
            ci_low=float("nan"),
            ci_high=float("nan"),
            effect_size=float("nan"),
            p_value=float("nan"),
            per_class_shift={},
            n_clips=n_clusters,
        )

    lo, hi = (float(v) for v in np.quantile(boots, [0.025, 0.975]))
    sd = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
    effect = point / sd if sd > 0 else 0.0
    # Normal approximation with the clip as the unit: 2.8 sigma is the classic
    # 80%-power, two-sided, alpha=0.05 factor. Slightly conservative because the
    # true statistic is a cluster bootstrap, but it is reported in the same units
    # as the effect so the two can be compared directly.
    mde = float(2.8 * sd / np.sqrt(n_clusters)) if n_clusters else float("nan")
    # Two-sided bootstrap p-value: how often a resample lands on the other side
    # of zero. With 4,000 draws the floor is 2.5e-4, so that is also the smallest
    # p-value this method can report, and it is reported as such.
    p = float(2.0 * min((boots <= 0).mean(), (boots >= 0).mean()))
    p = max(p, 1.0 / n_boot)

    per_class = {}
    for k, cls in enumerate(fer.EMOSIGN_LABELS):
        per_class[cls] = float(
            np.mean([a.prob_vector()[k] for a, _ in pairs])
            - np.mean([b.prob_vector()[k] for _, b in pairs])
        )

    return AuditStats(
        marker=label,
        model="all",
        n_pairs=len(pairs),
        mean_shift=point,
        ci_low=lo,
        ci_high=hi,
        effect_size=effect,
        p_value=p,
        per_class_shift=per_class,
        n_clips=n_clusters,
        mde=mde,
    )


def marker_emotion_table(windows: Iterable[Window]) -> dict[str, dict[str, float]]:
    """Mean FER probability per class, split by whether each marker is present.

    The confusable, reviewer-facing view: a brow raise is supposed to be grammar,
    so if the "grammar" rows and the "affect" rows look alike in this table, that
    is the confound stated in numbers.
    """
    out: dict[str, dict[str, float]] = {}
    windows = list(windows)
    names = windows[0].names if windows else M.MARKERS
    for marker in names:
        on = [w for w in windows if w.scored and w.marker(marker)]
        off = [w for w in windows if w.scored and w.free_of(marker)]
        if not on or not off:
            continue
        out[marker] = {
            **{
                f"p_{c}": float(np.mean([w.prob_vector()[k] for w in on]))
                for k, c in enumerate(fer.EMOSIGN_LABELS)
            },
            "n_on": len(on),
            **{
                f"p_{c}_off": float(np.mean([w.prob_vector()[k] for w in off]))
                for k, c in enumerate(fer.EMOSIGN_LABELS)
            },
            "n_off": len(off),
        }
    return out


def render_table(stats: list[AuditStats]) -> str:
    """Terminal table.

    Carries the **minimum detectable effect** next to every result. A null
    without its power is an absence of measurement, and a reader who cannot see
    the MDE cannot tell "we ruled out a large bias" from "we could not have seen
    one". It also carries the number of clips, because the clip is the
    bootstrap's resampling unit and therefore the effective sample size - the pair
    count is not.
    """
    head = (
        f"{'marker':<15} {'pairs':>5} {'clips':>5} {'shift':>8} {'95% CI':>21} "
        f"{'MDE':>7} {'d':>7} {'p':>8}"
    )
    lines = [head, "-" * len(head)]
    for s in stats:
        if np.isnan(s.ci_low):
            interval, mde = "too few clips", "n/a"
        else:
            interval = f"[{s.ci_low:+.4f}, {s.ci_high:+.4f}]"
            mde = "n/a" if np.isnan(s.mde) else f"{s.mde:.4f}"
        p = "n/a" if np.isnan(s.p_value) else f"{s.p_value:.4f}"
        d = "n/a" if np.isnan(s.effect_size) else f"{s.effect_size:+.3f}"
        lines.append(
            f"{s.marker:<15} {s.n_pairs:>5} {s.n_clips:>5} {s.mean_shift:>+8.4f} "
            f"{interval:>21} {mde:>7} {d:>7} {p:>8}"
        )
    return "\n".join(lines)


def save(windows: list[Window], stats: list[AuditStats], table: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "window": WINDOW,
        "stride": STRIDE,
        "target_fps": TARGET_FPS,
        "n_windows": len(windows),
        "n_scored": sum(1 for w in windows if w.scored),
        "marker_provenance": M.describe(),
        "fer": fer.describe(),
        "stats": [s.as_dict() for s in stats],
        "marker_emotion_table": table,
        "windows": [asdict(w) for w in windows],
    }
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
