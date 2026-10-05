#!/usr/bin/env python
"""Do the two hand slots hold two hands? And is re-extraction repeatable?

Until 2026-10-05 `seam.perception.tasks_api` built two identical hand graphs, each asked
for one hand on the same image, and filed the first result under ``left_hand`` and the
second under ``right_hand``. Two identical graphs find the same hand. Nothing checked it,
because every consumer read the slots by index and each slot, taken alone, looked like a
hand.

This script is that check, kept so it cannot happen silently again. For every EmoSign
landmark shard it counts the frames where both slots are filled and, of those, the frames
where the two wrists are the same point. It reports the same for a second shard directory
when one is given (the pre-fix extraction is kept beside the current one), and, because
the two directories are two extractions of the same videos, how far the *face* read-out
moved between them - which is a measurement of how repeatable MediaPipe's video-mode
tracking is, and has nothing to do with hands.

Writes ``artifacts/m3/hand_slots.json``. Exits 1 if the current shards hold one hand twice.
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

from seam.paths import artifacts_root, default_data_root
from seam.perception import extract as X
from seam.perception.tasks_api import PART_SLICES
from seam.provenance import KEY as PROVENANCE_KEY
from seam.provenance import stamp

#: Two wrists closer than this, in image-normalised units, are the same detection.
SAME_POINT = 1e-4
#: A blendshape that moved by more than this between two extractions counts as changed.
CHANGED = 0.02


def slot_report(directory: Path) -> dict[str, object]:
    left0, right0 = PART_SLICES["left_hand"][0], PART_SLICES["right_hand"][0]
    names = list(X.PART_NAMES)
    li, ri = names.index("left_hand"), names.index("right_hand")
    frames = both = identical = clips = 0
    rates: list[np.ndarray] = []
    for shard in sorted(directory.glob("*.npz")):
        try:
            arrays, _ = X.load_shard(shard)
        except X.ExtractionError:
            continue
        lm, pres = arrays["landmarks"].astype(np.float64), arrays["presence"].astype(bool)
        two = pres[:, li] & pres[:, ri]
        gap = np.linalg.norm(lm[:, left0, :2] - lm[:, right0, :2], axis=1)
        clips += 1
        frames += len(lm)
        both += int(two.sum())
        identical += int((two & (gap < SAME_POINT)).sum())
        rates.append(pres[:, [li, ri]].mean(axis=0))
    mean = np.mean(rates, axis=0) if rates else np.zeros(2)
    return {
        "directory": str(directory),
        "clips": clips,
        "frames": frames,
        "frames_with_both_slots_filled": both,
        "of_those_the_two_wrists_are_one_point": identical,
        "mean_detection_rate_left": round(float(mean[0]), 4),
        "mean_detection_rate_right": round(float(mean[1]), 4),
    }


def face_repeatability(current: Path, earlier: Path) -> dict[str, object]:
    """How far the blendshapes moved between two extractions of the same videos."""
    worst_per_frame: list[np.ndarray] = []
    clips_changed = clips = 0
    for shard in sorted(current.glob("*.npz")):
        other = earlier / shard.name
        if not other.is_file():
            continue
        try:
            a, _ = X.load_shard(shard)
            b, _ = X.load_shard(other)
        except X.ExtractionError:
            continue
        if len(a["blendshapes"]) != len(b["blendshapes"]):
            continue
        d = np.abs(a["blendshapes"] - b["blendshapes"]).max(axis=1)
        worst_per_frame.append(d)
        clips += 1
        clips_changed += int((d > CHANGED).any())
    if not worst_per_frame:
        return {"clips_compared": 0}
    d = np.concatenate(worst_per_frame)
    return {
        "what": "largest change in any of the 52 blendshapes, per frame, between two "
        "extractions of the same videos by the same face graph",
        "clips_compared": clips,
        "frames_compared": len(d),
        "fraction_of_frames_identical": round(float((d < 1e-6).mean()), 4),
        "fraction_of_frames_changed_by_more_than_0.02": round(float((d > CHANGED).mean()), 4),
        "clips_with_any_such_frame": clips_changed,
    }


def main() -> int:
    emosign = default_data_root() / "emosign"
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--landmarks", type=Path, default=emosign / "landmarks")
    ap.add_argument(
        "--earlier",
        type=Path,
        default=emosign / "landmarks_one_hand_2026-10-05",
        help="an earlier extraction of the same clips, compared if present",
    )
    ap.add_argument("--out", default=str(artifacts_root() / "m3" / "hand_slots.json"))
    args = ap.parse_args()

    current = slot_report(args.landmarks)
    if not current["clips"]:
        print(f"no shards under {args.landmarks}; run `make landmarks` first")
        return 1
    out: dict[str, object] = {
        "question": "do the left_hand and right_hand slots hold two different hands?",
        "current": current,
    }
    if args.earlier.is_dir():
        out["before_the_fix"] = slot_report(args.earlier)
        out["face_repeatability"] = face_repeatability(args.landmarks, args.earlier)
    out[PROVENANCE_KEY] = stamp(__file__)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    for label in ("current", "before_the_fix"):
        row = out.get(label)
        if isinstance(row, dict):
            print(
                f"{label:>15}: {row['clips']} clips, both slots filled on "
                f"{row['frames_with_both_slots_filled']} frames, one hand twice on "
                f"{row['of_those_the_two_wrists_are_one_point']}; detection "
                f"left {row['mean_detection_rate_left']} right {row['mean_detection_rate_right']}"
            )
    rep = out.get("face_repeatability")
    if isinstance(rep, dict) and rep.get("clips_compared"):
        print(
            f"face read-out between the two extractions: identical on "
            f"{rep['fraction_of_frames_identical']} of frames, changed by more than 0.02 on "
            f"{rep['fraction_of_frames_changed_by_more_than_0.02']}, in "
            f"{rep['clips_with_any_such_frame']} of {rep['clips_compared']} clips"
        )
    print(f"wrote {path}")
    both, same = (
        current["frames_with_both_slots_filled"],
        current["of_those_the_two_wrists_are_one_point"],
    )
    if both and same / both > 0.01:  # type: ignore[operator]
        print("FAIL: the two hand slots hold the same hand")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
