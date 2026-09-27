#!/usr/bin/env python
"""Run the M1 confound audit end to end.

Three stages, each cached so a re-run is cheap:

1. **score** - decode every WLASL clip's video, crop the face from the stored
   landmarks, run each FER baseline, and cache the **per-frame** probabilities.
   The shards hold geometry, not pixels, and a FER model has to see pixels to be
   a fair stand-in for a hearing non-signer. Per-frame rather than per-window on
   purpose: the resampling and windowing then happen in exactly one place, in
   stage 3, and the two stages cannot disagree about what a window is.
2. **windows** - resample to 12 fps, compute markers and prosody, attach
   probabilities, keep the windows with hands and a visible face.
3. **audit** - matched-pair statistics with a cluster bootstrap over clips, the
   per-marker probability table, and a uniform-random null model.

The null model is not decoration. If a uniform classifier shows a shift, the
shift is coming from the matching procedure and none of the other numbers mean
anything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seam.data import wlasl as W
from seam.eval import fer
from seam.eval import fer_audit as A
from seam.features import markers as M
from seam.logging import get, setup
from seam.paths import artifacts_root
from seam.perception import extract as extract_mod

#: Module level, because the stage functions below use it and a logger
#: created inside main() is invisible to them.
log = get("audit")

WLASL_ROOT = Path("/home/bhuwan/Videos/wlasl/videos")
SHARD_ROOT = Path("/mnt/DevProd/seam_data/wlasl/landmarks")


@dataclass(slots=True)
class Clip:
    """A manifest clip plus the shard key, which is all the audit needs."""

    key: str
    path: str
    gloss: str
    signer_id: int
    split: str


def usable_clips(manifest: Path) -> list[Clip]:
    clips = [
        Clip(
            key=f"{c.gloss}__{c.instance_id}",
            path=c.path or "",
            gloss=c.gloss,
            signer_id=c.signer_id,
            split=c.split,
        )
        for c in W.load(manifest)
        if c.status == "ok" and c.path
    ]
    clips.sort(key=lambda c: c.key)
    return clips


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


def stage_score(clips: list[Clip], models, cache: Path, limit: int | None, force: bool) -> None:
    for name, model in models:
        dest = cache / f"{name}.npz"
        if dest.is_file() and not force:
            log.info("%s: cached", name)
            continue
        cache.mkdir(parents=True, exist_ok=True)
        keys: list[str] = []
        frames: list[np.ndarray] = []
        started = time.perf_counter()

        for n, clip in enumerate(clips, start=1):
            if limit and n > limit:
                break
            shard = extract_mod.shard_path(SHARD_ROOT, clip.key)
            video = WLASL_ROOT / clip.path
            if not shard.is_file() or not video.is_file():
                continue
            try:
                arrays, _meta = extract_mod.load_shard(shard)
            except extract_mod.ExtractionError:
                continue
            probs = A.score_clip_faces(
                video,
                arrays["landmarks"].astype(np.float32),
                arrays["presence"].astype(bool),
                model,
            )
            if not len(probs):
                continue
            keys.append(clip.key)
            frames.append(probs)
            if n % 50 == 0:
                rate = n / max(time.perf_counter() - started, 1e-6)
                print(
                    f"  {name}: {n}/{len(clips)}  {rate:.2f} clips/s  "
                    f"eta {(len(clips) - n) / max(rate, 1e-9) / 60:.1f}m",
                    file=sys.stderr,
                    flush=True,
                )

        np.savez_compressed(
            dest,
            keys=np.array(keys),
            frames=np.concatenate(frames) if frames else np.zeros((0, len(fer.EMOSIGN_LABELS))),
            offsets=np.cumsum([0] + [len(f) for f in frames]),
        )
        log.info("%s: scored %d clips -> %s", name, len(keys), dest)


def load_cached(cache: Path) -> dict[str, dict[str, np.ndarray]]:
    """Per-model, per-clip frame-aligned probability tables."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for npz in sorted(cache.glob("*.npz")):
        blob = np.load(npz, allow_pickle=True)
        keys = [str(k) for k in blob["keys"]]
        frames = blob["frames"]
        offsets = blob["offsets"]
        out[npz.stem] = {k: frames[offsets[i] : offsets[i + 1]] for i, k in enumerate(keys)}
        log.info("cached probabilities: %s (%d clips)", npz.stem, len(keys))
    if not out:
        raise FileNotFoundError(f"no cached probabilities under {cache}; run --stages score")
    return out


def build_windows(
    clips: list[Clip], tables: dict[str, dict[str, np.ndarray]]
) -> dict[str, list[A.Window]]:
    """One window list per FER model, built through the same code path."""
    out: dict[str, list[A.Window]] = {}
    for model_name, table in tables.items():
        windows: list[A.Window] = []
        for clip in clips:
            probs = table.get(clip.key)
            if probs is None:
                continue
            shard = extract_mod.shard_path(SHARD_ROOT, clip.key)
            if not shard.is_file():
                continue
            try:
                arrays, meta = extract_mod.load_shard(shard)
            except extract_mod.ExtractionError:
                continue
            windows.extend(A.build_windows(clip.key, arrays, meta, probs))
        out[model_name] = windows
        log.info(
            "%s: %d windows, %d scorable",
            model_name,
            len(windows),
            sum(w.scored for w in windows),
        )
    return out


def null_windows(windows: list[A.Window]) -> list[A.Window]:
    """The same windows scored by a classifier that cannot be biased.

    Resampling probs uniformly on the simplex per window. If the audit's
    machinery is sound this produces a shift indistinguishable from zero; if it
    does not, every other number in the report is uninterpretable.
    """
    rng = np.random.default_rng(0)
    out = []
    for w in windows:
        if not w.scored:
            continue
        clone = A.Window(
            clip=w.clip,
            index=w.index,
            start=w.start,
            n_frames=w.n_frames,
            markers=w.markers,
            prosody=w.prosody,
            probs=None,
            scored=True,
        )
        clone.probs = [float(v) for v in rng.dirichlet(np.ones(len(fer.EMOSIGN_LABELS)))]
        out.append(clone)
    return out


def audit_one(windows: list[A.Window], label: str) -> tuple[list[A.AuditStats], dict]:
    stats = []
    for marker in M.MARKERS:
        pairs = A.match_windows(windows, marker)
        if not pairs:
            log.warning("%s: no matched pairs for %s", label, marker)
            continue
        stats.append(A.bootstrap_shift(pairs, marker))
    return stats, A.marker_emotion_table(windows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="score,audit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--fer-dir",
        default=None,
        help="directory holding the FER checkpoints (default artifacts/fer)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    setup("INFO")
    fer_dir = Path(args.fer_dir) if args.fer_dir else artifacts_root() / "fer"
    cache = fer_dir / "clip_probs"
    out_dir = Path(args.out) if args.out else artifacts_root() / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = {s.strip() for s in args.stages.split(",")}

    manifest = artifacts_root() / "data" / "wlasl_manifest.json"
    clips = usable_clips(manifest)
    log.info("%d usable WLASL clips", len(clips))
    if args.limit:
        clips = clips[: args.limit]

    if "score" in stages:
        stage_score(clips, load_models(fer_dir), cache, args.limit, args.force)

    if "audit" not in stages:
        return 0

    tables = load_cached(cache)
    per_model = build_windows(clips, tables)

    report: dict = {
        "design": {
            "window": A.WINDOW,
            "stride": A.STRIDE,
            "target_fps": A.TARGET_FPS,
            "matching": "within-clip, nearest neighbour on (amplitude, speed), "
            "0.5 sd tolerance, unmatched windows dropped",
            "bootstrap": "cluster over clips, 4000 resamples",
            "readout": "negative mass = P(anger)+P(disgust)+P(fear)+P(sadness)",
        },
        "fer": fer.describe(),
        "marker_provenance": M.describe(),
        "models": {},
    }

    for model_name, windows in sorted(per_model.items()):
        stats, table = audit_one(windows, model_name)
        print(f"\n=== {model_name} ===")
        print(A.render_table(stats))
        report["models"][model_name] = {
            "n_windows": len(windows),
            "n_scored": sum(w.scored for w in windows),
            "stats": [s.as_dict() for s in stats],
            "marker_emotion_table": table,
        }

    # Null model, once, on the best-populated model.
    biggest = max(per_model, key=lambda m: len(per_model[m]))
    nw = null_windows(per_model[biggest])
    nstats, ntable = audit_one(nw, "null")
    print("\n=== null model (uniform, cannot be biased) ===")
    print(A.render_table(nstats))
    report["null_model"] = {
        "on": biggest,
        "stats": [s.as_dict() for s in nstats],
        "marker_emotion_table": ntable,
    }

    dest = out_dir / "confound_audit.json"
    dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
