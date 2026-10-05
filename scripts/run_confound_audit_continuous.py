#!/usr/bin/env python
"""C1 on continuous signing: do non-signer FER models read grammar as negative affect?

M1 tested this on isolated WLASL signs with heuristic markers and found nothing, and said
why that was not the test: the confound is a discourse phenomenon, and a dictionary video
has no discourse. The test it asked for needs continuous signing with markers that are
there *because of the syntax*. This is that test: the 200 EmoSign utterances, the same
three RAF-DB-trained FER models, and the **human** SignStream annotations as the marker
source, frame-aligned through `signstream.marker_frame_mask` at each clip's own frame
rate.

Everything below was fixed before the first run, and the first run's output is the
result. Nothing here was chosen by looking at a shift.

Design
------
* **Unit.** Windows of 6 frames at 12 fps (0.5 s), stride 3. The isolated-sign audit
  used 2 s windows; annotated events here have a median length of 0.5 to 0.6 s for brow
  raise, brow furrow, head shake and head nod, so a 2 s window would mostly straddle
  them. Chosen from event durations alone.
* **Contrast.** Within clip. A window *bears* a marker when the annotation covers at
  least half of it and is *free* of it when the annotation does not touch it; windows in
  between are on neither side. Each bearing window is paired with the nearest free
  window of the same clip on (amplitude, speed), within 1.0 pooled sd, as in M1. A
  clip's affect label is constant across its windows, so affect cannot produce a
  within-clip difference (rule 12).
* **Only windows inside the signing span.** A clip starts and ends with the signer at
  rest, and a resting face is not a signing face. Grammatical markers sit inside the
  utterance, so comparing them with *all* marker-free windows would partly compare
  signing with not signing. The primary analysis therefore keeps only windows lying
  wholly between the first and last annotated gloss. The unrestricted version is
  reported beside it as a sensitivity check, and the decision rule reads the primary.
* **Read-out.** Negative probability mass, P(anger)+P(disgust)+P(fear)+P(sadness), as in
  M1. Shift = bearing minus free. Positive means the marker is read as more negative.
* **Inference.** Cluster bootstrap over clips, 4,000 resamples, with the minimum
  detectable effect beside every estimate (rule 17). Fewer than 10 clips: no interval.
* **Markers.** Ten, all from human annotation. Four forms - brow_raise, brow_furrow,
  head_shake, head_nod - and six grammatical functions - negation, topic, conditional,
  question_yn, question_wh, question_rhetorical.

Controls
--------
* **Placebo.** Every marker's track, circularly shifted within its clip by a seeded
  offset between a quarter and three quarters of the clip. Same duty cycle, same run
  lengths, no relation to the face. A real effect must be absent here.
* **Null model.** The same windows scored by a uniform-random classifier.
* **Positive control.** `tests/test_audit.py` plants a +0.05 shift on human-marked frames
  and recovers it through this exact code path.

Decision rule
-------------
C1 is **supported for a marker** when its shift is positive with a Bonferroni-corrected
p below 0.05 (corrected over the markers that have an interval) in **at least two of the
three models**, and its placebo is not significant in those models. Otherwise the marker
is reported as a null with its MDE - or, when a significant shift is negative or appears
in one model only, as that, and not as support. C1 as a whole is supported if any marker
is.

Writes ``artifacts/audit/confound_audit_continuous.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from seam.data import asllrp
from seam.data import emosign as em
from seam.data.signstream import Utterance, marker_frame_mask, parse_directory
from seam.eval import fer
from seam.eval import fer_audit as A
from seam.logging import get, setup
from seam.paths import artifacts_root, default_data_root
from seam.perception import extract as extract_mod
from seam.preprocess import normalize as N
from seam.provenance import KEY as PROVENANCE_KEY
from seam.provenance import stamp

log = get("confound_audit_continuous")

WINDOW = 6
STRIDE = 3
BEARING_MIN = 0.5

FORM_MARKERS = ("brow_raise", "brow_furrow", "head_shake", "head_nod")
FUNCTION_MARKERS = (
    "negation",
    "topic",
    "conditional",
    "question_yn",
    "question_wh",
    "question_rhetorical",
)
MARKERS = FORM_MARKERS + FUNCTION_MARKERS
PLACEBO = "placebo:"

ALPHA = 0.05
MODELS_REQUIRED = 2


def load_models(fer_dir: Path) -> list[tuple[str, object]]:
    import torch

    out = []
    for ckpt in sorted(fer_dir.glob("*.pt")):
        blob = torch.load(ckpt, weights_only=False)
        spec = fer.FerModelSpec(**blob["spec"])
        model = fer.build_model(spec, n_classes=blob["n_classes"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        out.append((spec.name, model))
    if not out:
        raise FileNotFoundError(f"no FER checkpoints under {fer_dir}")
    return out


def score(clips: list[dict], models: list[tuple[str, object]], cache: Path, force: bool) -> None:
    """Per-frame FER probabilities for every clip and model, cached like the WLASL pass."""
    for name, model in models:
        dest = cache / f"{name}.npz"
        if dest.is_file() and not force:
            log.info("%s: cached", name)
            continue
        cache.mkdir(parents=True, exist_ok=True)
        keys: list[str] = []
        frames: list[np.ndarray] = []
        for c in clips:
            probs = A.score_clip_faces(
                c["video"],
                c["arrays"]["landmarks"].astype(np.float32),
                c["arrays"]["presence"].astype(bool),
                model,
            )
            keys.append(c["id"])
            frames.append(probs)
        np.savez_compressed(
            dest,
            keys=np.array(keys),
            frames=np.concatenate(frames),
            offsets=np.cumsum([0] + [len(f) for f in frames]),
        )
        log.info("%s: scored %d clips -> %s", name, len(keys), dest)


def load_scores(cache: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for npz in sorted(cache.glob("*.npz")):
        blob = np.load(npz, allow_pickle=True)
        keys = [str(k) for k in blob["keys"]]
        frames, offsets = blob["frames"], blob["offsets"]
        out[npz.stem] = {k: frames[offsets[i] : offsets[i + 1]] for i, k in enumerate(keys)}
    if not out:
        raise FileNotFoundError(f"no cached probabilities under {cache}")
    return out


def placebo_track(track: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The same track, rolled by between a quarter and three quarters of the clip."""
    n = len(track)
    if n < 4:
        return track.copy()
    return np.roll(track, int(rng.integers(n // 4, 3 * n // 4 + 1)))


def signing_span(u: Utterance, n: int, fps: float) -> np.ndarray:
    """Frames between the first and last annotated gloss, on the clip's own frames."""
    mask = np.zeros(n, dtype=bool)
    bounds = [(a, b) for _g, a, b in u.glosses if a is not None and b is not None]
    if not bounds or u.start_frame is None:
        return mask
    lo = asllrp.crop_frame_position(min(a for a, _ in bounds), u.start_frame, fps)
    hi = asllrp.crop_frame_position(max(b for _, b in bounds), u.start_frame, fps)
    lo_i, hi_i = max(int(np.ceil(lo)), 0), min(int(np.floor(hi)), n - 1)
    if hi_i >= lo_i:
        mask[lo_i : hi_i + 1] = True
    return mask


def tracks_for(u: Utterance, n: int, fps: float, seed: int) -> dict[str, np.ndarray]:
    """Human tracks for every marker, plus a placebo for each, on the clip's own frames."""
    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for name in MARKERS:
        out[name] = marker_frame_mask(u, name, n, fps)
    for name in MARKERS:
        out[PLACEBO + name] = placebo_track(out[name], rng)
    return out


def load_clips(xml_dir: Path) -> tuple[list[dict], dict[str, int]]:
    root = default_data_root()
    utterances, _ = parse_directory(xml_dir)
    by_id = {u.utterance_id: u for u in utterances}
    video_root, shard_root = root / "emosign" / "video", root / "emosign" / "landmarks"
    clips: list[dict] = []
    skipped = {"no_annotation": 0, "no_video_or_shard": 0, "unreadable_shard": 0}
    for i, rec in enumerate(em.load(root)):
        u = by_id.get(rec.utterance_id)
        if u is None:
            skipped["no_annotation"] += 1
            continue
        video = video_root / f"{rec.utterance_id}.mp4"
        shard = shard_root / f"{rec.utterance_id}.npz"
        if not video.is_file() or not shard.is_file():
            skipped["no_video_or_shard"] += 1
            continue
        try:
            arrays, meta = extract_mod.load_shard(shard)
        except extract_mod.ExtractionError:
            skipped["unreadable_shard"] += 1
            continue
        n = int(arrays["landmarks"].shape[0])
        clips.append(
            {
                "id": rec.utterance_id,
                "signer": rec.signer,
                "video": video,
                "arrays": arrays,
                "meta": meta,
                "tracks": tracks_for(u, n, float(meta.native_fps), seed=i),
                "signing": signing_span(u, n, float(meta.native_fps)),
            }
        )
    return clips, skipped


def windows_for(
    clips: list[dict], table: dict[str, np.ndarray]
) -> tuple[list[A.Window], list[A.Window], int]:
    """``(inside the signing span, all windows, clips dropped)`` for one model."""
    windows: list[A.Window] = []
    inside: list[A.Window] = []
    short = 0
    for c in clips:
        probs = table.get(c["id"])
        n = int(c["arrays"]["landmarks"].shape[0])
        if probs is None or len(probs) != n:
            # A video that decodes to fewer frames than its shard cannot be indexed by
            # the shard's frame numbers. Counted, never padded.
            short += 1
            continue
        built = A.build_windows(
            c["id"],
            c["arrays"],
            c["meta"],
            probs,
            window=WINDOW,
            stride=STRIDE,
            frame_markers=c["tracks"],
            bearing_min=BEARING_MIN,
        )
        windows += built
        # The same resampling the builder applied, so a window's frames can be looked up
        # in the signing span.
        idx = N.resample_indices(n, float(c["meta"].native_fps), A.TARGET_FPS, max_frames=None)
        signing = c["signing"][idx]
        inside += [w for w in built if signing[w.start : w.start + w.n_frames].all()]
    return inside, windows, short


def null_windows(windows: list[A.Window]) -> list[A.Window]:
    rng = np.random.default_rng(0)
    out = []
    for w in windows:
        if not w.scored:
            continue
        out.append(
            A.Window(
                clip=w.clip,
                index=w.index,
                start=w.start,
                n_frames=w.n_frames,
                markers=w.markers,
                prosody=w.prosody,
                probs=[float(v) for v in rng.dirichlet(np.ones(len(fer.EMOSIGN_LABELS)))],
                scored=True,
                names=w.names,
                partial=w.partial,
            )
        )
    return out


def audit(windows: list[A.Window]) -> dict[str, dict]:
    """Matched shift for every marker and its placebo, with the Bonferroni correction."""
    out: dict[str, dict] = {}
    for name in MARKERS + tuple(PLACEBO + m for m in MARKERS):
        pairs = A.match_windows(windows, name)
        bearing = sum(1 for w in windows if w.scored and w.marker(name))
        free = sum(1 for w in windows if w.scored and w.free_of(name))
        s = A.bootstrap_shift(pairs, name).as_dict()
        s["n_bearing_windows"] = bearing
        s["n_free_windows"] = free
        out[name] = s
    for family in (MARKERS, tuple(PLACEBO + m for m in MARKERS)):
        tested = [m for m in family if out[m]["p_value"] == out[m]["p_value"] and out[m]["n_pairs"]]
        for m in family:
            p = out[m]["p_value"]
            out[m]["n_tests"] = len(tested)
            out[m]["p_bonferroni"] = (
                round(min(1.0, p * len(tested)), 5) if m in tested else float("nan")
            )
    return out


def verdicts(per_model: dict[str, dict[str, dict]]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for m in MARKERS:
        rows = {model: stats[m] for model, stats in per_model.items()}
        placebo = {model: stats[PLACEBO + m] for model, stats in per_model.items()}
        with_interval = [r for r in rows.values() if r["p_value"] == r["p_value"] and r["n_pairs"]]
        pos = [
            model
            for model, r in rows.items()
            if r["p_bonferroni"] == r["p_bonferroni"]
            and r["p_bonferroni"] < ALPHA
            and r["mean_shift"] > 0
        ]
        neg = [
            model
            for model, r in rows.items()
            if r["p_bonferroni"] == r["p_bonferroni"]
            and r["p_bonferroni"] < ALPHA
            and r["mean_shift"] < 0
        ]
        placebo_sig = [
            model
            for model, r in placebo.items()
            if r["p_bonferroni"] == r["p_bonferroni"] and r["p_bonferroni"] < ALPHA
        ]
        if not with_interval:
            verdict = "no estimate: fewer than 10 clips contribute a matched pair"
        elif len(pos) >= MODELS_REQUIRED and not set(pos) & set(placebo_sig):
            verdict = "SUPPORTS C1: read as more negative, replicated across models"
        elif len(pos) >= MODELS_REQUIRED:
            verdict = "not interpretable: the placebo is significant in the same models"
        elif pos or neg:
            verdict = (
                "not support: significant in "
                f"{len(pos)} model(s) positive, {len(neg)} negative; the rule needs "
                f"{MODELS_REQUIRED} positive"
            )
        else:
            verdict = "null at the stated MDE"
        out[m] = {
            "verdict": verdict,
            "models_positive_significant": pos,
            "models_negative_significant": neg,
            "placebo_significant_in": placebo_sig,
            "shift_by_model": {k: round(r["mean_shift"], 5) for k, r in rows.items()},
            "mde_by_model": {
                k: (None if r["mde"] != r["mde"] else round(r["mde"], 5)) for k, r in rows.items()
            },
            "n_clips_by_model": {k: r["n_clips"] for k, r in rows.items()},
        }
    return out


def render(per_model: dict[str, dict[str, dict]], names: tuple[str, ...]) -> str:
    lines = []
    for model, stats in per_model.items():
        lines += [
            f"\n=== {model} ===",
            f"{'marker':30s} {'pairs':>6} {'clips':>6} {'shift':>9} {'95% CI':>22} "
            f"{'MDE':>8} {'p':>8} {'p_bonf':>8}",
        ]
        for m in names:
            r = stats[m]
            if r["p_value"] != r["p_value"] or not r["n_pairs"]:
                lines.append(
                    f"{m:30s} {r['n_pairs']:>6} {r['n_clips']:>6} {r['mean_shift']:>+9.4f} "
                    f"{'too few clips':>22}"
                )
                continue
            ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            lines.append(
                f"{m:30s} {r['n_pairs']:>6} {r['n_clips']:>6} {r['mean_shift']:>+9.4f} "
                f"{ci:>22} {r['mde']:>8.4f} {r['p_value']:>8.4f} {r['p_bonferroni']:>8.4f}"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--xml", default=str(default_data_root() / "asllrp_signstream_xml" / "raw"))
    ap.add_argument("--fer-dir", default=None)
    ap.add_argument("--force", action="store_true", help="re-score even if cached")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup("INFO")

    fer_dir = Path(args.fer_dir) if args.fer_dir else artifacts_root() / "fer"
    cache = fer_dir / "clip_probs_emosign"
    out_path = (
        Path(args.out)
        if args.out
        else artifacts_root() / "audit" / "confound_audit_continuous.json"
    )

    clips, skipped = load_clips(Path(args.xml))
    log.info(
        "%d clips with video, landmarks and human annotation (skipped %s)", len(clips), skipped
    )
    score(clips, load_models(fer_dir), cache, args.force)
    tables = load_scores(cache)

    per_model: dict[str, dict[str, dict]] = {}
    unrestricted: dict[str, dict[str, dict]] = {}
    counts: dict[str, dict] = {}
    biggest: list[A.Window] = []
    for model, table in sorted(tables.items()):
        windows, every, short = windows_for(clips, table)
        per_model[model] = audit(windows)
        unrestricted[model] = audit(every)
        scored = [w for w in windows if w.scored]
        counts[model] = {
            "n_windows_all": len(every),
            "n_windows": len(windows),
            "n_scored": len(scored),
            "clips_dropped_frame_count_mismatch": short,
            "negative_mass_baseline": round(
                float(np.mean([fer.negative_mass(w.prob_vector()) for w in scored])), 4
            )
            if scored
            else None,
        }
        if len(windows) > len(biggest):
            biggest = windows
    null = audit(null_windows(biggest))

    signers: dict[str, int] = {}
    for c in clips:
        signers[c["signer"]] = signers.get(c["signer"], 0) + 1
    report = {
        "question": "on continuous signing, are windows that carry a human-annotated "
        "grammatical marker read as more negative by non-signer FER models than matched "
        "marker-free windows of the same clip?",
        "design": {
            "window_frames": WINDOW,
            "stride_frames": STRIDE,
            "analysis_fps": A.TARGET_FPS,
            "bearing_min_coverage": BEARING_MIN,
            "free_max_coverage": 0.0,
            "matching": "within clip, nearest on (amplitude, speed), 1.0 pooled sd",
            "readout": "negative mass = P(anger)+P(disgust)+P(fear)+P(sadness)",
            "bootstrap": "cluster over clips, 4000 resamples",
            "markers": {"form": list(FORM_MARKERS), "function": list(FUNCTION_MARKERS)},
            "marker_source": "human SignStream annotation, frame-aligned at the clip's fps",
            "placebo": "each track rolled within its clip by 1/4 to 3/4 of the clip length",
            "primary_analysis": "windows wholly inside the signing span (first to last "
            "annotated gloss); the unrestricted analysis is a sensitivity check",
            "decision_rule": (
                f"supported for a marker when the shift is positive with Bonferroni p < {ALPHA} "
                f"in at least {MODELS_REQUIRED} of the models and the placebo is not "
                "significant in those models"
            ),
            "fixed_before_first_run": True,
        },
        "data": {"n_clips": len(clips), "clips_by_signer": signers, "skipped": skipped},
        "fer": fer.describe(),
        "counts": counts,
        "models": per_model,
        "sensitivity_all_windows": {
            "models": unrestricted,
            "verdicts": verdicts(unrestricted),
        },
        "null_model": null,
        "verdicts": verdicts(per_model),
    }
    report["c1_supported_on_continuous_signing"] = any(
        v["verdict"].startswith("SUPPORTS") for v in report["verdicts"].values()
    )
    report[PROVENANCE_KEY] = stamp(__file__)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(render(per_model, MARKERS))
    print("\n--- placebo (tracks rolled within clip) ---")
    print(render(per_model, tuple(PLACEBO + m for m in MARKERS)))
    print("\n--- null model (uniform, cannot be biased) ---")
    print(render({"null": null}, MARKERS))
    print("\nverdicts (primary: windows inside the signing span):")
    for m, v in report["verdicts"].items():
        print(f"  {m:22s} {v['verdict']}")
    print("\nsensitivity (all windows, including rest frames at the clip edges):")
    print(render(unrestricted, MARKERS))
    for m, v in report["sensitivity_all_windows"]["verdicts"].items():  # type: ignore[index]
        print(f"  {m:22s} {v['verdict']}")
    print(f"\nC1 supported on continuous signing: {report['c1_supported_on_continuous_signing']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
