#!/usr/bin/env python
"""Can the non-manual markers be read off the face at all? Against human frame labels.

Every marker in this project has so far been a heuristic over blendshapes and head pose,
and none was ever checked against an annotator. M3 found four of six "degenerate" or
"blind" at clip level and stopped there, because there was nothing to check them against.
There is now: the SignStream annotators marked brow raises, brow furrows, head shakes and
head nods with frame bounds on all 200 EmoSign utterances.

Two questions, per marker, at frame level:

1. **How good is the heuristic?** AUC of the detector's own per-frame evidence against
   the human track. No fitting, so no split is needed, but it is reported per held-out
   signer to sit beside (2).
2. **How good could a detector be from the same inputs?** A logistic regression on the
   52 blendshapes and head pose, with short-range dynamics, trained on three signers and
   scored on the fourth.

Design, fixed before the first run
----------------------------------
* **Features per frame:** 52 blendshapes; head roll, pitch, yaw; their rate of change;
  and their rolling standard deviation over a centred 0.5 s window, because a shake or a
  nod is an oscillation and no single frame shows one. Blendshapes and angles are
  centred on the clip's own median, which removes the signer's resting face and the
  camera angle without reading any label.
* **Model:** logistic regression, C=1, balanced class weights, features standardised
  with training-fold statistics.
* **Evaluation:** leave one signer out. Two AUCs per fold: over all test frames, and the
  mean of per-clip AUCs, which cannot be earned by telling clips apart.
* **Gate:** a marker is **validated** when the supervised detector's within-clip AUC is at
  least 0.80 on each of the three large folds (Cory, Jonathan, Rachel), **usable with
  caution** at 0.70, and **not validated** below that. Ben's seven clips are reported and
  do not decide the verdict.

Second design, added 2026-10-05 after the head-axis finding
-----------------------------------------------------------
The first run scored both head markers at 0.50, heuristic and fitted. The cause was not the
face: `seam.features.markers` names its head angles by an aerospace convention and so reads
a head *tilt* for `head_shake` and a head *turn* for `head_nod`
(`seam.features.head_motion` has the measurement). The 61-feature model had all three
angles and still failed, because a 0.5 s rolling sd of a raw angle is dominated by slow
posture changes and 52 clip-centred blendshapes do not transfer across signers.

Everything above is kept and still reported, as the baseline. Three rows are added:

3. **The live page's detector, unchanged.** `docs/index.html` was written before any label
   was read: brows as a blendshape mean rescaled by a running percentile, head shake and
   nod as two reversals of at least 5 degrees inside 1.3 s, on the correct axes. Ported
   line for line and scored causally. Nothing is fitted, so this is a pure test of what
   the page shows.
   For the two head markers a **revised page detector** is scored as well: the same idea
   made amplitude-free and causal (`head_motion.live_oscillation`), its three constants
   chosen from `LIVE_GRID` on the training signers. Its level threshold is set there too,
   at the score unmarked frames exceed a quarter of the time; the page's *event* also
   needs two reversals of 2 degrees, and the hit and false-alarm rates of that event are
   reported beside those of the event the page fired before. This is what the page runs
   after this change.
4. **A tuned detector.** One signal chosen from a small declared family (`CANDIDATES`),
   the choice made on the three training signers by mean within-clip AUC and scored on
   the fourth. The choice per fold is written to the artifact.
5. **A compact fitted detector.** Logistic regression on a handful of physically meant
   features (`compact_features`): brow blendshapes and brow geometry for the brow
   markers, band-limited motion of the three head axes for the head markers.

The gate is unchanged. An exploratory pass over all four signers was looked at before this
design was fixed (which axis is which; that an amplitude-free band energy beats a 5 degree
reversal count; that no single brow signal fixes the furrow). It decided what went into
`CANDIDATES`; no number from it is reported, and every number below comes from a detector
whose free choice was made without the held-out signer.

Writes ``artifacts/m3/marker_validation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from seam.data import emosign as em
from seam.data.signstream import marker_frame_mask, parse_directory
from seam.eval.probes import auc
from seam.features import head_motion as HM
from seam.features import markers as VM
from seam.paths import artifacts_root, default_data_root
from seam.perception import extract as extract_mod
from seam.provenance import KEY as PROVENANCE_KEY
from seam.provenance import stamp

MARKERS = ("brow_raise", "brow_furrow", "head_shake", "head_nod")
DECIDING_FOLDS = ("Cory", "Jonathan", "Rachel")
VALIDATED, CAUTION = 0.80, 0.70
ROLL_SECONDS = 0.5


def rolling_std(x: np.ndarray, half: int) -> np.ndarray:
    """Centred rolling standard deviation along axis 0, shrinking at the clip edges."""
    out = np.empty_like(x, dtype=np.float64)
    n = len(x)
    for i in range(n):
        out[i] = x[max(i - half, 0) : min(i + half + 1, n)].std(axis=0)
    return out


def frame_features(blendshapes: np.ndarray, rotation: np.ndarray, fps: float) -> np.ndarray:
    """(T, 61) per-frame features: the face, the head, and how the head is moving."""
    bs = np.asarray(blendshapes, dtype=np.float64)
    euler = np.unwrap(VM._euler_from_matrix(np.asarray(rotation, dtype=np.float64)), axis=0)
    rate = np.gradient(euler, axis=0) * fps if len(euler) > 1 else np.zeros_like(euler)
    half = max(round(ROLL_SECONDS * fps / 2), 1)
    wobble = rolling_std(euler, half)
    bs = bs - np.median(bs, axis=0, keepdims=True)
    euler = euler - np.median(euler, axis=0, keepdims=True)
    return np.hstack([bs, euler, rate, wobble])


FACE = slice(75, 543)
SMOOTH_SECONDS = 0.25
#: The live page's `Adaptive`: baseline and range over this many frames, refreshed every 8.
ADAPTIVE_BUFFER, ADAPTIVE_EVERY, ADAPTIVE_FLOOR = 600, 8, 0.1

Signal = Callable[[dict], np.ndarray]


def adaptive_level(x: np.ndarray) -> np.ndarray:
    """The live page's brow level: the signal over the viewer's own running baseline.

    `Adaptive` in `docs/index.html`, ported: the baseline is the 20th percentile of the
    last 600 samples and the range runs from there to the 95th, both refreshed every 8th
    frame, and the level is clipped to [0, 1.5]. A clip is scored from a cold start, which
    is the harder case: the page has usually seen the face for a while.
    """
    out = np.zeros(len(x), dtype=np.float64)
    base, span = 0.0, ADAPTIVE_FLOOR
    for i in range(len(x)):
        if i % ADAPTIVE_EVERY == 0:
            buf = np.sort(x[max(i + 1 - ADAPTIVE_BUFFER, 0) : i + 1])
            base = float(buf[int(len(buf) * 0.2)])
            span = max(float(buf[int(len(buf) * 0.95)]) - base, ADAPTIVE_FLOOR)
        out[i] = min(max((float(x[i]) - base) / span, 0.0), 1.5)
    return out


def _smoothed(x: np.ndarray, fps: float) -> np.ndarray:
    return HM._moving_average(x, max(round(SMOOTH_SECONDS * fps), 1))


def _bs(clip: dict, names: tuple[str, ...]) -> np.ndarray:
    return np.asarray(VM._mean(clip["bs"], names), dtype=np.float64)


def _reversals(axis: int, min_deg: float, window_s: float) -> Signal:
    return lambda c: HM.reversal_score(
        c["angles"][:, axis], c["fps"], min_deg=min_deg, window_s=window_s
    )


def _energy(axis: int, window_s: float) -> Signal:
    return lambda c: HM.band_energy(c["angles"][:, axis], c["fps"], window_s=window_s)


def _head_family(axis: int, other: int) -> dict[str, Signal]:
    name = HM.AXES[axis]
    fam: dict[str, Signal] = {}
    for min_deg in (1.5, 2.0, 3.0, 5.0):
        for window_s in (0.8, 1.3):
            fam[f"{name} reversals >= {min_deg:g} deg in {window_s:g} s"] = _reversals(
                axis, min_deg, window_s
            )
    for window_s in (0.5, 0.8, 1.2):
        fam[f"{name} band energy over {window_s:g} s"] = _energy(axis, window_s)
    fam[f"{name} minus {HM.AXES[other]} band energy over 0.8 s"] = lambda c: (
        HM.band_energy(c["angles"][:, axis], c["fps"])
        - HM.band_energy(c["angles"][:, other], c["fps"])
    )
    return fam


#: The declared family each tuned detector is chosen from. Nothing here reads a label.
CANDIDATES: dict[str, dict[str, Signal]] = {
    "brow_raise": {
        "brow-raise blendshapes": lambda c: _bs(c, VM.BROW_RAISE),
        "brow-raise blendshapes, smoothed": lambda c: _smoothed(_bs(c, VM.BROW_RAISE), c["fps"]),
        "brow-raise minus brow-down blendshapes": lambda c: (
            _bs(c, VM.BROW_RAISE) - _bs(c, VM.BROW_FURROW)
        ),
        "mid-brow height (landmarks)": lambda c: c["geometry"]["mid_height"],
    },
    "brow_furrow": {
        "brow-down blendshapes": lambda c: _bs(c, VM.BROW_FURROW),
        "brow-down blendshapes, smoothed": lambda c: _smoothed(_bs(c, VM.BROW_FURROW), c["fps"]),
        "inner-brow-up blendshape, negated": lambda c: -_bs(c, ("browInnerUp",)),
        "inner-brow gap (landmarks), negated": lambda c: -c["geometry"]["inner_gap"],
        "inner-brow height (landmarks), negated": lambda c: -c["geometry"]["inner_height"],
        "outer-brow height (landmarks), negated": lambda c: -c["geometry"]["outer_height"],
    },
    "head_shake": _head_family(HM.TURN, HM.NOD),
    "head_nod": _head_family(HM.NOD, HM.TURN),
}

#: What `docs/index.html` computed before any label was read.
BROWSER: dict[str, Signal] = {
    "brow_raise": lambda c: adaptive_level(_bs(c, VM.BROW_RAISE)),
    "brow_furrow": lambda c: adaptive_level(_bs(c, VM.BROW_FURROW)),
    "head_shake": _reversals(HM.TURN, HM.BROWSER_MIN_DEG, HM.BROWSER_WINDOW_S),
    "head_nod": _reversals(HM.NOD, HM.BROWSER_MIN_DEG, HM.BROWSER_WINDOW_S),
}


def compact_features(clip: dict, marker: str) -> np.ndarray:
    """A few features that mean something physically, for the fitted detector.

    Brow markers: four blendshape groups and four landmark measurements, centred on the
    clip's median, each also smoothed over a quarter second (16 columns). Head markers:
    band energy of the three axes at two window lengths and reversal counts of turn and
    nod at two amplitudes (10 columns); these are already motion, so nothing is centred.
    """
    fps = clip["fps"]
    if marker in ("brow_raise", "brow_furrow"):
        cols = [
            _bs(clip, VM.BROW_RAISE),
            _bs(clip, VM.BROW_FURROW),
            _bs(clip, ("browInnerUp",)),
            _bs(clip, ("eyeSquintLeft", "eyeSquintRight")),
            *(
                clip["geometry"][k]
                for k in ("inner_gap", "inner_height", "mid_height", "outer_height")
            ),
        ]
        base = np.stack(cols, axis=1)
        base = base - np.median(base, axis=0, keepdims=True)
        smooth = np.stack([_smoothed(base[:, j], fps) for j in range(base.shape[1])], axis=1)
        return np.hstack([base, smooth])
    a = clip["angles"]
    head = [HM.band_energy(a[:, ax], fps, window_s=w) for ax in range(3) for w in (0.5, 1.0)]
    head += [
        HM.reversal_score(a[:, ax], fps, min_deg=d, window_s=1.3)
        for ax in (HM.TURN, HM.NOD)
        for d in (2.0, 4.0)
    ]
    return np.stack(head, axis=1)


def face_geometry(landmarks: np.ndarray, presence: np.ndarray | None) -> dict[str, np.ndarray]:
    """Brow geometry, with frames that have no face set to the clip's own median."""
    geo = HM.brow_geometry(np.asarray(landmarks[:, FACE, :], dtype=np.float64))
    seen = np.abs(np.asarray(landmarks[:, FACE, :2], dtype=np.float64)).sum(axis=(1, 2)) > 0
    if presence is not None and len(presence) == len(seen):
        seen &= np.asarray(presence)[:, 0].astype(bool)
    for k, v in geo.items():
        fill = float(np.median(v[seen])) if seen.any() else 0.0
        geo[k] = np.where(seen & np.isfinite(v), v, fill)
    return geo


def pose_proxies(landmarks: np.ndarray) -> dict[str, np.ndarray]:
    """Three things the head can do, measured on the face landmarks with no rotation matrix.

    Used only to settle which angle is which: the nose tip moves sideways across the face
    when the head turns and up or down it when the head nods, and the line through the
    outer eye corners rotates when the head tilts.
    """
    lm = np.asarray(landmarks[:, FACE, :2], dtype=np.float64)
    width = np.linalg.norm(lm[:, 454] - lm[:, 234], axis=1) + 1e-6
    eyes = lm[:, 263] - lm[:, 33]
    return {
        "nose sideways (a turn)": (lm[:, 1, 0] - 0.5 * (lm[:, 234, 0] + lm[:, 454, 0])) / width,
        "nose up or down (a nod)": (lm[:, 1, 1] - 0.5 * (lm[:, 10, 1] + lm[:, 152, 1])) / width,
        "eye line rotates (a tilt)": np.degrees(np.arctan2(eyes[:, 1], eyes[:, 0])),
    }


def axis_check(clips: list[dict]) -> dict[str, object]:
    """Which angle follows which motion: median per-clip |correlation| with the landmarks.

    This is the evidence that `markers._euler_from_matrix` names its columns for a vehicle
    and not a face. Signs are dropped because image and camera axes differ in handedness;
    the question is only which column moves.
    """
    columns = {
        "markers.roll": lambda c: c["legacy_euler"][:, 0],
        "markers.pitch": lambda c: c["legacy_euler"][:, 1],
        "markers.yaw (read as head_shake)": lambda c: c["legacy_euler"][:, 2],
        "head_motion.turn": lambda c: c["angles"][:, HM.TURN],
        "head_motion.nod": lambda c: c["angles"][:, HM.NOD],
        "head_motion.tilt": lambda c: c["angles"][:, HM.TILT],
    }
    table: dict[str, dict[str, float]] = {}
    for motion in next(iter(clips))["pose_proxies"]:
        row: dict[str, float] = {}
        for name, column in columns.items():
            rs = []
            for c in clips:
                a, b = c["pose_proxies"][motion], column(c)
                ok = np.isfinite(a) & np.isfinite(b) & c["face_seen"]
                if ok.sum() >= 12 and a[ok].std() > 1e-9 and b[ok].std() > 1e-9:
                    rs.append(abs(float(np.corrcoef(a[ok], b[ok])[0, 1])))
            row[name] = round(float(np.median(rs)), 3) if rs else float("nan")
        table[motion] = row
    return {
        "what": "median over clips of |Pearson r| between a landmark measurement of one head "
        "motion and each angle column",
        "median_abs_correlation": table,
    }


def mean_within_clip(clips: list[dict], marker: str, signal: Signal) -> float:
    vals = [safe_auc(c["y"][marker], signal(c)) for c in clips]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def choose_candidate(
    train: list[dict], marker: str, family: dict[str, Signal] | None = None
) -> str:
    """The family member with the best mean within-clip AUC on the training signers."""
    family = CANDIDATES[marker] if family is None else family
    scored = {name: mean_within_clip(train, marker, fn) for name, fn in family.items()}
    usable = {k: v for k, v in scored.items() if v == v}
    if not usable:
        return next(iter(family))
    return max(usable, key=lambda k: usable[k])


#: The revised live detector is chosen from this causal family, the same way.
LIVE_GRID = [
    {"contrast": contrast, "slow_s": slow_s, "hold_s": hold_s}
    for contrast in (0.0, 0.5, 1.0)
    for slow_s in (0.4, 0.7)
    for hold_s in (0.4, 0.8)
]
LIVE_AXES = {"head_shake": (HM.TURN, HM.NOD), "head_nod": (HM.NOD, HM.TURN)}
#: The page's level threshold: the score that unmarked frames exceed this often. The page's
#: *event* also needs the swing gate below, which removes most of those false alarms.
LIVE_FALSE_ALARM = 0.25
#: The event needs the angle to have gone there and back: this many reversals of this size.
SWING_MIN_DEG = VM.DEFAULT_THRESHOLDS.swing_deg
SWING_REVERSALS = VM.DEFAULT_THRESHOLDS.min_reversals


def live_name(params: dict[str, float]) -> str:
    return (
        f"contrast {params['contrast']:g}, slow {params['slow_s']:g} s, hold {params['hold_s']:g} s"
    )


def live_signal(marker: str, params: dict[str, float]) -> Signal:
    axis, other = LIVE_AXES[marker]
    return lambda c: HM.live_oscillation(
        c["angles"][:, axis], c["angles"][:, other], c["fps"], **params
    )


def live_family(marker: str) -> dict[str, Signal]:
    return {live_name(p): live_signal(marker, p) for p in LIVE_GRID}


LIVE_PARAMS = {live_name(p): p for p in LIVE_GRID}


def threshold_at_false_alarm(clips: list[dict], marker: str, signal: Signal) -> float:
    """The score that frames the annotators did not mark exceed `LIVE_FALSE_ALARM` of the time."""
    quiet = np.concatenate([signal(c)[~c["y"][marker].astype(bool)] for c in clips])
    return float(np.quantile(quiet, 1.0 - LIVE_FALSE_ALARM)) if len(quiet) else float("nan")


def _rates(on: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "hit_rate": round(float(on[y].mean()), 4) if y.any() else float("nan"),
        "false_alarm_rate": round(float(on[~y].mean()), 4) if (~y).any() else float("nan"),
    }


def swing_gate(clip: dict, marker: str) -> np.ndarray:
    """True where the angle has reversed `SWING_REVERSALS` times by `SWING_MIN_DEG` recently."""
    score = HM.reversal_score(
        clip["angles"][:, LIVE_AXES[marker][0]], clip["fps"], min_deg=SWING_MIN_DEG
    )
    return np.asarray(score >= SWING_REVERSALS / HM.BROWSER_FULL_SCALE - 1e-9)


def old_page_event(clip: dict, marker: str) -> np.ndarray:
    """The page's event before this change: two reversals of 5 degrees inside 1.3 s."""
    return np.asarray(BROWSER[marker](clip) >= 0.6)


def operating_point(
    clips: list[dict], marker: str, signal: Signal, threshold: float
) -> dict[str, object]:
    """What fraction of marked, and of unmarked, frames each on/off rule calls active.

    AUC says how well the level ranks frames; the page also has to say yes or no. Three
    rules are scored on the same frames: the level against its threshold, the event the
    page fires (level and swing gate together), and the event the page fired before.
    """
    y = np.concatenate([c["y"][marker] for c in clips]).astype(bool)
    level = np.concatenate([signal(c) for c in clips]) >= threshold
    gate = np.concatenate([swing_gate(c, marker) for c in clips])
    old = np.concatenate([old_page_event(c, marker) for c in clips])
    return {
        "threshold_deg_rms": round(threshold, 3),
        "level_only": _rates(level, y),
        "event_with_swing_gate": _rates(level & gate, y),
        "old_page_event": _rates(old, y),
    }


def revised_page_row(train: list[dict], test: list[dict], marker: str) -> dict[str, object]:
    """Choose the causal detector and its threshold on the training signers; score the rest."""
    family = live_family(marker)
    chosen = choose_candidate(train, marker, family)
    signal = family[chosen]
    threshold = threshold_at_false_alarm(train, marker, signal)
    return {
        "chosen_on_training_signers": chosen,
        **two_aucs(test, marker, [signal(c) for c in test]),
        "operating_point": operating_point(test, marker, signal, threshold),
    }


def compact_scores(train: list[dict], test: list[dict], marker: str) -> list[np.ndarray] | None:
    from sklearn.linear_model import LogisticRegression

    x_tr = np.vstack([compact_features(c, marker) for c in train])
    y_tr = np.concatenate([c["y"][marker] for c in train]).astype(int)
    if y_tr.min() == y_tr.max():
        return None
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    model.fit((x_tr - mu) / sd, y_tr)
    return [model.decision_function((compact_features(c, marker) - mu) / sd) for c in test]


def load(xml_dir: Path) -> list[dict]:
    root = default_data_root()
    utterances, _ = parse_directory(xml_dir)
    by_id = {u.utterance_id: u for u in utterances}
    shard_root = root / "emosign" / "landmarks"
    clips: list[dict] = []
    for rec in em.load(root):
        u = by_id.get(rec.utterance_id)
        shard = shard_root / f"{rec.utterance_id}.npz"
        if u is None or not shard.is_file():
            continue
        try:
            arrays, meta = extract_mod.load_shard(shard)
        except extract_mod.ExtractionError:
            continue
        fps = float(meta.native_fps)
        bs, rot = arrays["blendshapes"], arrays["head_rotation"]
        n = len(bs)
        sig = VM.signals(bs.astype(np.float64), rot.astype(np.float64), fps=fps)
        evidence = VM.clip_evidence(sig)
        clips.append(
            {
                "id": rec.utterance_id,
                "signer": rec.signer,
                "fps": fps,
                "bs": bs.astype(np.float64),
                "angles": HM.head_angles(rot.astype(np.float64)),
                "legacy_euler": VM._euler_from_matrix(rot.astype(np.float64)),
                "pose_proxies": pose_proxies(arrays["landmarks"]),
                "face_seen": np.asarray(arrays["presence"])[:, 0].astype(bool),
                "geometry": face_geometry(arrays["landmarks"], arrays.get("presence")),
                "x": frame_features(bs, rot, fps),
                "y": {m: marker_frame_mask(u, m, n, fps) for m in MARKERS},
                "heuristic_raw": {
                    m: np.asarray(getattr(sig, m), dtype=np.float64) for m in MARKERS
                },
                "heuristic_evidence": {
                    m: np.asarray(evidence[m], dtype=np.float64) for m in MARKERS
                },
            }
        )
    return clips


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(auc(y, s))


def two_aucs(clips: list[dict], marker: str, scores: list[np.ndarray]) -> dict[str, object]:
    """AUC over all frames, and the mean of per-clip AUCs."""
    y_all = np.concatenate([c["y"][marker] for c in clips])
    s_all = np.concatenate(scores)
    per_clip = [safe_auc(c["y"][marker], s) for c, s in zip(clips, scores, strict=True)]
    per_clip = [v for v in per_clip if v == v]
    return {
        "auc_all_frames": round(safe_auc(y_all, s_all), 4),
        "auc_within_clip": round(float(np.mean(per_clip)), 4) if per_clip else float("nan"),
        "n_clips_with_both_classes": len(per_clip),
        "n_frames": len(y_all),
        "positive_fraction": round(float(y_all.mean()), 4),
    }


def supervised_scores(train: list[dict], test: list[dict], marker: str) -> list[np.ndarray] | None:
    from sklearn.linear_model import LogisticRegression

    x_tr = np.vstack([c["x"] for c in train])
    y_tr = np.concatenate([c["y"][marker] for c in train]).astype(int)
    if y_tr.min() == y_tr.max():
        return None
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    model.fit((x_tr - mu) / sd, y_tr)
    return [model.decision_function((c["x"] - mu) / sd) for c in test]


def verdict(within_clip: dict[str, float]) -> str:
    vals = [within_clip.get(f, float("nan")) for f in DECIDING_FOLDS]
    if any(v != v for v in vals):
        return "no verdict: a deciding fold has no clip with both classes"
    worst = min(vals)
    if worst >= VALIDATED:
        return "validated"
    if worst >= CAUTION:
        return "usable with caution"
    return "not validated"


#: The constants the live page and `seam.features.markers` both run. Read from the module,
#: never retyped here, so this script reports on the detector that is actually deployed.
DEPLOYED = {"contrast": 0.0, "slow_s": VM.HEAD_SLOW_S, "hold_s": VM.HEAD_HOLD_S}
DEPLOYED_THRESHOLD = {
    "head_shake": VM.DEFAULT_THRESHOLDS.shake_rms_deg,
    "head_nod": VM.DEFAULT_THRESHOLDS.nod_rms_deg,
}


def revised_summary(
    clips: list[dict], folds: dict[str, dict], signers: list[str], marker: str
) -> dict[str, object]:
    """Held-out scores of the revised page detector, and a check on the one that ships.

    The held-out rows are what is claimed: in each fold the detector's constants and its
    threshold come from the other three signers. What ships is one fixed member of the
    same grid, the same in the page and in `seam.features.markers`. Several members score
    within a few thousandths of each other, and which is best flips between two extraction
    runs of the same videos, so the shipped constants are fixed and checked rather than
    re-chosen: this block reports how far the shipped member is from the best one, and how
    far its threshold is from the quantile it was set at. Tests hold both to a tolerance.
    """
    if marker not in LIVE_AXES:
        return {}
    within = {f: folds[f]["revised_page_detector"]["auc_within_clip"] for f in signers}
    family = live_family(marker)
    best = choose_candidate(clips, marker, family)
    shipped = live_name(DEPLOYED)
    signal = family[shipped]
    by_fold = {
        f: round(mean_within_clip([c for c in clips if c["signer"] == f], marker, signal), 4)
        for f in signers
    }
    return {
        "revised_page_within_clip_auc_by_fold": within,
        "verdict_revised_page": verdict(within),
        "revised_page_choice_by_fold": {
            f: folds[f]["revised_page_detector"]["chosen_on_training_signers"] for f in signers
        },
        "shipped_in_page": {
            "name": shipped,
            **DEPLOYED,
            "fast_s": VM.HEAD_FAST_S,
            "threshold_deg_rms": DEPLOYED_THRESHOLD[marker],
            "threshold_at_false_alarm_target_now": round(
                threshold_at_false_alarm(clips, marker, signal), 2
            ),
            "false_alarm_target": LIVE_FALSE_ALARM,
            "swing_min_deg": VM.DEFAULT_THRESHOLDS.swing_deg,
            "swing_reversals": VM.DEFAULT_THRESHOLDS.min_reversals,
            "within_clip_auc_by_signer": by_fold,
            "mean_within_clip_auc_all_signers": round(mean_within_clip(clips, marker, signal), 4),
            "best_member_on_all_signers": {
                "name": best,
                "mean_within_clip_auc": round(mean_within_clip(clips, marker, family[best]), 4),
            },
            "operating_point_all_signers": operating_point(
                clips, marker, signal, DEPLOYED_THRESHOLD[marker]
            ),
            "note": "fixed constants, chosen on all four signers on 2026-10-05, so its own "
            "scores here are not held-out results; revised_page_within_clip_auc_by_fold is",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xml", default=str(default_data_root() / "asllrp_signstream_xml" / "raw"))
    ap.add_argument("--out", default=str(artifacts_root() / "m3" / "marker_validation.json"))
    args = ap.parse_args()

    clips = load(Path(args.xml))
    if not clips:
        print("no clips with landmarks and SignStream annotations")
        return 1
    signers = sorted({c["signer"] for c in clips})

    results: dict[str, dict] = {}
    for marker in MARKERS:
        folds: dict[str, dict] = {}
        for held in signers:
            train = [c for c in clips if c["signer"] != held]
            test = [c for c in clips if c["signer"] == held]
            sup = supervised_scores(train, test, marker)
            chosen = choose_candidate(train, marker)
            compact = compact_scores(train, test, marker)
            folds[held] = {
                "n_clips": len(test),
                "browser_detector": two_aucs(test, marker, [BROWSER[marker](c) for c in test]),
                "tuned_detector": {
                    "chosen_on_training_signers": chosen,
                    **two_aucs(test, marker, [CANDIDATES[marker][chosen](c) for c in test]),
                },
                "supervised_compact": (
                    two_aucs(test, marker, compact) if compact is not None else None
                ),
                "revised_page_detector": (
                    revised_page_row(train, test, marker) if marker in LIVE_AXES else None
                ),
                "heuristic_raw_signal": two_aucs(
                    test, marker, [c["heuristic_raw"][marker] for c in test]
                ),
                "heuristic_clip_evidence": two_aucs(
                    test, marker, [c["heuristic_evidence"][marker] for c in test]
                ),
                "supervised": two_aucs(test, marker, sup) if sup is not None else None,
            }
        within = {
            f: folds[f]["supervised"]["auc_within_clip"]
            for f in signers
            if folds[f]["supervised"] is not None
        }
        heur_within = {f: folds[f]["heuristic_raw_signal"]["auc_within_clip"] for f in signers}
        browser_within = {f: folds[f]["browser_detector"]["auc_within_clip"] for f in signers}
        tuned_within = {f: folds[f]["tuned_detector"]["auc_within_clip"] for f in signers}
        compact_within = {
            f: folds[f]["supervised_compact"]["auc_within_clip"]
            for f in signers
            if folds[f]["supervised_compact"] is not None
        }
        results[marker] = {
            "folds": folds,
            "browser_within_clip_auc_by_fold": browser_within,
            "tuned_within_clip_auc_by_fold": tuned_within,
            "compact_within_clip_auc_by_fold": compact_within,
            "tuned_choice_by_fold": {
                f: folds[f]["tuned_detector"]["chosen_on_training_signers"] for f in signers
            },
            "verdict_browser": verdict(browser_within),
            "verdict_tuned": verdict(tuned_within),
            "verdict_compact": verdict(compact_within),
            **revised_summary(clips, folds, signers, marker),
            "supervised_within_clip_auc_by_fold": within,
            "heuristic_within_clip_auc_by_fold": heur_within,
            "verdict_supervised": verdict(within),
            "verdict_heuristic": verdict(heur_within),
        }

    checked: dict = axis_check(clips)
    out = {
        "question": "are the four form markers readable from blendshapes and head pose, "
        "frame by frame, against human annotation?",
        "design": {
            "features": "52 blendshapes + head roll/pitch/yaw, their rate, and their rolling "
            f"sd over {ROLL_SECONDS} s; blendshapes and angles centred on the clip median",
            "model": "logistic regression, C=1, balanced class weights, train-fold standardisation",
            "evaluation": "leave one signer out; AUC over all frames and mean per-clip AUC",
            "gate": f"validated when within-clip AUC >= {VALIDATED} on each of "
            f"{', '.join(DECIDING_FOLDS)}; usable with caution at {CAUTION}",
            "labels": "human SignStream annotation, frame-aligned at the clip's fps",
            "fixed_before_first_run": True,
            "second_design": {
                "why": "the first run read head tilt for head_shake and head turn for head_nod; "
                "see seam.features.head_motion",
                "browser_detector": "docs/index.html ported unchanged: brows as a blendshape mean "
                "over a running-percentile baseline, head as >= 2 reversals of "
                f"{HM.BROWSER_MIN_DEG:g} degrees inside {HM.BROWSER_WINDOW_S:g} s; nothing fitted",
                "tuned_detector": "one member of a declared family per marker, chosen by mean "
                "within-clip AUC on the three training signers",
                "candidates": {m: list(fam) for m, fam in CANDIDATES.items()},
                "supervised_compact": "logistic regression, C=1, balanced, on 16 brow or 10 "
                "head-motion features",
                "revised_page_detector": "head markers only: a causal band-energy detector "
                "(seam.features.head_motion.live_oscillation) chosen from LIVE_GRID on the "
                "training signers, with its level threshold set there at a "
                f"{LIVE_FALSE_ALARM:.0%} false-alarm rate on unmarked frames; the page's event "
                f"also needs {SWING_REVERSALS} reversals of {SWING_MIN_DEG:g} degrees",
                "live_grid": LIVE_GRID,
                "exploration_disclosed": "axis identity and the candidate family were decided "
                "after an exploratory look at all four signers; no number from that look is "
                "reported",
            },
        },
        "n_clips": len(clips),
        "axis_check": checked,
        "markers": results,
        PROVENANCE_KEY: stamp(__file__),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print("\nwhich angle follows which head motion (median |r| over clips):")
    check: dict[str, dict[str, float]] = checked["median_abs_correlation"]
    for motion, row in check.items():
        cells = "  ".join(f"{k.split('.')[1][:5]} {v:.2f}" for k, v in row.items())
        print(f"  {motion:<28}{cells}")
    for marker, r in results.items():
        print(
            f"\n{marker}: supervised {r['verdict_supervised']}; heuristic {r['verdict_heuristic']}"
        )
        print(
            f"  second design: page detector {r['verdict_browser']}; tuned {r['verdict_tuned']}; "
            f"compact fitted {r['verdict_compact']}"
        )
        if "shipped_in_page" in r:
            print(f"  revised page detector: {r['verdict_revised_page']}")
            print(f"  the page ships: {r['shipped_in_page']}")
            for f, d in r["folds"].items():
                v = d["revised_page_detector"]
                new, old = (
                    v["operating_point"]["event_with_swing_gate"],
                    v["operating_point"]["old_page_event"],
                )
                print(
                    f"  {f:<10} revised page {v['auc_within_clip']:.3f}  event hits "
                    f"{new['hit_rate']:.2f} fa {new['false_alarm_rate']:.2f}  (before: hits "
                    f"{old['hit_rate']:.2f} fa {old['false_alarm_rate']:.2f})"
                    f"   <- {v['chosen_on_training_signers']}"
                )
        for f, d in r["folds"].items():
            b, t, k = d["browser_detector"], d["tuned_detector"], d["supervised_compact"]
            kk = "n/a" if k is None else f"{k['auc_within_clip']:.3f}"
            print(
                f"  {f:<10} page {b['auc_within_clip']:.3f}  tuned {t['auc_within_clip']:.3f}  "
                f"compact {kk}   <- {t['chosen_on_training_signers']}"
            )
        print("  within-clip AUC, with the AUC over all frames in brackets")
        print(
            f"  {'fold':<10}{'clips':>6}{'pos%':>7}  {'heuristic':>10}{'evidence':>10}"
            f"{'supervised':>16}"
        )
        for f, d in r["folds"].items():
            h, e, s = d["heuristic_raw_signal"], d["heuristic_clip_evidence"], d["supervised"]
            sup = "n/a" if s is None else f"{s['auc_within_clip']:.3f} [{s['auc_all_frames']:.3f}]"
            print(
                f"  {f:<10}{d['n_clips']:>6}{100 * h['positive_fraction']:>6.1f}%  "
                f"{h['auc_within_clip']:>10.3f}{e['auc_within_clip']:>10.3f}{sup:>16}"
            )
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
