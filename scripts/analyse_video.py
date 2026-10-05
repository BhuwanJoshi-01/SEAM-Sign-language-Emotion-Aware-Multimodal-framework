#!/usr/bin/env python
"""Run SEAM on a video of your own, offline, and print what it reads.

    python scripts/analyse_video.py my_clip.mp4
    python scripts/analyse_video.py my_clip.mp4 --emotion --out my_clip.json

What it does, with nothing trained or downloaded at run time:

1. Runs the three MediaPipe graphs over every frame: 478 face points and 52 blendshapes,
   21 points for each hand, 33 body points.
2. Reads the four grammatical markers the project validated against human annotation -
   brow raise, brow furrow, head shake, head nod - and lists when each is active.
3. With ``--emotion``, also runs the project's three trained facial-expression models on
   the face and reports their average output. That part is shown for inspection only:
   in our own tests the emotion models did not beat chance on signing, and the report
   says so beside the numbers.

Nothing is uploaded. The video is read from disk and the result is printed and, with
``--out``, written as JSON.
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

#: What each marker means in ASL, and how far the project trusts its read-out.
MEANING = {
    "brow_raise": "yes/no question or topic",
    "brow_furrow": "wh-question",
    "head_shake": "negation",
    "head_nod": "affirmation or emphasis",
}
MIN_EVENT_SECONDS = 0.2


def events(mask: np.ndarray, fps: float) -> list[tuple[float, float]]:
    """(start, end) in seconds of every run of active frames at least 0.2 s long."""
    out: list[tuple[float, float]] = []
    start = None
    for i, on in enumerate([*mask.tolist(), False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if (i - start) / fps >= MIN_EVENT_SECONDS:
                out.append((round(start / fps, 2), round(i / fps, 2)))
            start = None
    return out


def validation_status() -> dict[str, str]:
    """Each marker's verdict from the validation artifact, if this copy has it."""
    path = _ROOT / "artifacts" / "m3" / "marker_validation.json"
    if not path.is_file():
        return {}
    markers = json.loads(path.read_text(encoding="utf-8"))["markers"]
    out = {}
    for name, row in markers.items():
        by_fold = row.get("revised_page_within_clip_auc_by_fold") or row.get(
            "heuristic_within_clip_auc_by_fold", {}
        )
        vals = [by_fold[f] for f in ("Cory", "Jonathan", "Rachel") if f in by_fold]
        verdict = row.get("verdict_revised_page") or row.get("verdict_heuristic", "")
        if vals:
            out[name] = f"{verdict}, AUC {min(vals):.2f}-{max(vals):.2f} on unseen signers"
    return out


def emotion_summary(video: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Mean output of the three trained expression models over the clip's face frames."""
    import torch

    from seam.eval import fer
    from seam.eval import fer_audit as A

    fer_dir = _ROOT / "artifacts" / "fer"
    ckpts = sorted(fer_dir.glob("*.pt"))
    if not ckpts:
        return {"error": f"no trained models under {fer_dir}"}
    per_model = {}
    for ckpt in ckpts:
        blob = torch.load(ckpt, weights_only=False)
        spec = fer.FerModelSpec(**blob["spec"])
        model = fer.build_model(spec, n_classes=blob["n_classes"])
        model.load_state_dict(blob["state_dict"])  # type: ignore[attr-defined]
        model.eval()  # type: ignore[attr-defined]
        probs = A.score_clip_faces(video, arrays["landmarks"], arrays["presence"], model)
        good = np.isfinite(probs).all(axis=1) if len(probs) else np.zeros(0, dtype=bool)
        mean = probs[good].mean(axis=0) if good.any() else np.zeros(len(fer.EMOSIGN_LABELS))
        per_model[spec.name] = {
            label: round(float(p), 3) for label, p in zip(fer.EMOSIGN_LABELS, mean, strict=False)
        }
    return {
        "models": per_model,
        "caveat": "Shown for inspection. On signing video these models did not beat chance "
        "in the project's tests (balanced accuracy 0.49-0.51), so this is not a verdict "
        "on what the signer feels.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("video", type=Path, help="a video file (mp4, mov, avi, webm ...)")
    ap.add_argument("--out", type=Path, default=None, help="also write the result as JSON here")
    ap.add_argument("--max-frames", type=int, default=None, help="stop after this many frames")
    ap.add_argument("--emotion", action="store_true", help="also run the trained emotion models")
    args = ap.parse_args()

    try:
        from seam.features import head_motion as HM
        from seam.features import markers as VM
        from seam.perception import extract as X
    except ImportError as exc:
        print(f"A package is missing ({exc}). Run the setup script first (full install).")
        return 2

    print(f"reading {args.video} ...")
    try:
        arrays, meta = X.extract_clip(args.video, max_frames=args.max_frames)
    except X.ExtractionError as exc:
        print(f"could not read the video: {exc}")
        return 1
    fps = float(meta.native_fps) or 25.0
    n = int(meta.frame_count)
    if n == 0 or "head_rotation" not in arrays:
        print("no face was found in this video, so there is nothing to read.")
        return 1

    bs = arrays["blendshapes"].astype(np.float64)
    rot = arrays["head_rotation"].astype(np.float64)
    sig = VM.signals(bs, rot, fps=fps)
    fired = VM.fire(sig)
    angles = HM.head_angles(rot)
    status = validation_status()

    result: dict[str, object] = {
        "video": str(args.video),
        "frames": n,
        "fps": round(fps, 2),
        "seconds": round(n / fps, 2),
        "detected_fraction": {k: round(float(v), 3) for k, v in meta.detection_rates.items()},
        "head_angle_range_deg": {
            name: round(float(np.ptp(angles[:, i])), 1) for i, name in enumerate(HM.AXES)
        },
        "markers": {},
    }
    print(f"\n{n} frames at {fps:.1f} fps ({n / fps:.1f} s)")
    found = "  ".join(f"{k} {100 * v:.0f}%" for k, v in meta.detection_rates.items())
    print(f"found in frame: {found}")
    print("\ngrammatical markers")
    for name, meaning in MEANING.items():
        mask = np.asarray(fired[name], dtype=bool)
        ev = events(mask, fps)
        result["markers"][name] = {  # type: ignore[index]
            "asl_meaning": meaning,
            "active_fraction": round(float(mask.mean()), 3),
            "events_seconds": ev,
            "validation": status.get(name, "not available in this copy"),
        }
        shown = ", ".join(f"{a:.1f}-{b:.1f}s" for a, b in ev[:6]) or "none"
        more = f" (+{len(ev) - 6} more)" if len(ev) > 6 else ""
        print(f"  {name:<12} active {100 * mask.mean():5.1f}% of frames   events: {shown}{more}")
        print(f"  {'':<12} in ASL: {meaning}.  Read-out: {status.get(name, 'see README')}")

    if args.emotion:
        try:
            result["emotion_models"] = emotion_summary(args.video, arrays)
        except ImportError as exc:
            result["emotion_models"] = {"error": f"needs the full install ({exc})"}
        emo = result["emotion_models"]
        print("\ntrained emotion models (for inspection, not a verdict)")
        if isinstance(emo, dict) and "models" in emo:
            for model_name, probs in emo["models"].items():  # type: ignore[union-attr]
                top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
                print(f"  {model_name}: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
            print(f"  {emo['caveat']}")
        else:
            print(f"  {emo}")

    if args.out:
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
