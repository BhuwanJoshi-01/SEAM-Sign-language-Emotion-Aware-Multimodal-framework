#!/usr/bin/env python
"""Extract landmarks and blendshapes for every usable WLASL clip.

Resumable: re-running skips clips that already have a valid shard. Run the full
set detached - it takes on the order of an hour at two workers.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from seam.data import wlasl as W
from seam.perception import batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/bhuwan/Videos/wlasl/videos")
    ap.add_argument(
        "--annotation",
        default="/mnt/Volume2/Sign_Language_Recognition/data/metadata/WLASL_v0.3.json",
    )
    ap.add_argument("--out", default="/mnt/DevProd/seam_data/wlasl/landmarks")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=batch.DEFAULT_MAX_FRAMES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    clips = W.load_annotation(pathlib.Path(args.annotation))
    W.attach(clips, root)
    W.probe(clips, root)
    usable = [c for c in clips if c.status == "ok"]
    if args.limit:
        usable = usable[: args.limit]
    print(f"{len(usable)} usable clips", file=sys.stderr)

    jobs = [batch.Job(video=str(root / c.path), key=f"{c.gloss}__{c.instance_id}") for c in usable]
    return batch.run(
        jobs,
        pathlib.Path(args.out),
        workers=args.workers,
        max_frames=args.max_frames,
        force=args.force,
        label="wlasl",
    )


if __name__ == "__main__":
    raise SystemExit(main())
