#!/usr/bin/env python
"""Build the blinded M7 study videos from generated GLB stimuli.

Ties three things together:

    <stimuli>/*.glb  ->  study spec JSON  ->  make_study_stimuli.py  ->  blinded mp4s

**What this script decides, and what it refuses to decide.**

- It decides the *packaging*: which clips enter the study, how long each video is, and
  the blinding seed. Those are mechanical.
- It refuses to decide the *comparison arm*. `condition B` is a study-design choice
  owned by the author (see `paper/provenance/TASK5_QUESTIONS_FOR_AUTHOR.md` Q1). Per
  Bhuwan's instruction we proceed with condition A only ("for now lets not think of
  baseline first lets make out"). So every stimulus is written under condition `A`, the
  system's own output, and no baseline is invented.

**Why a duration cap.** The EmoSign clips run 3-8 s. A rater comparing two arms has to
hold both in memory, so very long pairs are unratable. ``--max-seconds`` trims each clip
to its first N seconds; it never pads or loops, because a repeated body motion would read
as a different performance.

Writes:
  <work>/study_spec.json      the input spec for make_study_stimuli.py
  <out>/public/*.mp4          the blinded, rater-visible videos
  <out>/public/manifest.json  opaque names only (no condition anywhere)
  <out>/private/blinding_key.json   token -> condition; NEVER shown to a rater

Run from the repository root with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Defaults are the repo's own output locations, so a reader can run the pipeline in order
# without editing paths. ``make_stimuli.py`` writes the meshes; this writes the study.
DEFAULT_STIMULI = Path("artifacts/m7a/stimuli")
DEFAULT_OUT = Path("artifacts/m7a/study")
DEFAULT_FPS = 25


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stimuli", default=str(DEFAULT_STIMULI), help="dir of generated .glb")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="study output dir")
    ap.add_argument(
        "--work",
        default=None,
        help="where the spec JSON goes; default is a 'spec' dir beside --out",
    )
    ap.add_argument("--fps", type=int, default=DEFAULT_FPS)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=6.0,
        help="trim each clip to its first N seconds (0 disables); never pads or loops",
    )
    ap.add_argument(
        "--condition",
        default="A",
        help="the only arm defined so far; baseline/condition B is the author's call",
    )
    ap.add_argument("--limit", type=int, default=0, help="cap number of clips (0 = all)")
    args = ap.parse_args()

    stimuli = Path(args.stimuli)
    glbs = sorted(stimuli.glob("*.glb"))
    if not glbs:
        print(f"no .glb in {stimuli}; run scripts/make_stimuli.py first")
        return 1

    max_frames = round(args.max_seconds * args.fps) if args.max_seconds > 0 else 0

    # Every mesh enters the study under the one defined condition. The frame cap is passed
    # to `make_study_stimuli.py` as `max_frames` in the spec, which trims rather than
    # pads - see that script's `--max-frames` for why that distinction matters.
    used = []
    for glb in glbs:
        used.append({"clip_id": glb.stem, "condition": args.condition, "mesh": str(glb)})
        if args.limit and len(used) >= args.limit:
            break

    spec = {
        "note": (
            "Condition A only. The comparison arm (condition B / baseline) is a "
            "study-design decision owned by the author and is deliberately absent."
        ),
        "fps": args.fps,
        "max_frames": max_frames,
        "stimuli": used,
    }
    work = Path(args.work) if args.work else Path(args.out) / "spec"
    work.mkdir(parents=True, exist_ok=True)
    spec_path = work / "study_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {spec_path}  ({len(used)} stimuli, condition={args.condition})")
    print(
        "next: PYTHONPATH=src python scripts/make_study_stimuli.py "
        f"--input {spec_path} --out {args.out} --fast --max-frames {max_frames}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
