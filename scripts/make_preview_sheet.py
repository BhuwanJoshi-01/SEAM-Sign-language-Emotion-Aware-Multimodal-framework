"""Render a contact sheet of M7 stimulus clips, for eyeballing the retargeting.

Why this exists. ``make_stimuli.py`` writes ``.glb`` files, which is what the study needs,
and ``make_study_stimuli.py`` turns those into blinded videos - but neither writes the one
thing a human needs to spot a broken retarget: several clips side by side, at a few times
through each. Reviewing the output otherwise means opening each ``.glb`` in a viewer one at
a time, and a systematic fault - every body leaning, every leg splayed - is exactly the
kind that is obvious in a grid and invisible in a single file.

**It renders the ``.glb`` written to disk**, using the same loader the blinded-video step
uses, rather than re-deriving meshes from landmarks. Re-deriving here would hide any bug
between the landmarks and the file, which is precisely what is worth checking.

Usage::

    PYTHONPATH=src python scripts/make_preview_sheet.py \\
        --stimuli <dir with *.glb> \\
        --out     <dir>/_sheet.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from make_study_stimuli import _load_mesh_sequence

from seam.avatar.fast_render import render_frame_fast
from seam.avatar.render import Camera


def render_clip_tiles(
    seq: dict[str, np.ndarray],
    *,
    width: int,
    height: int,
    camera: Camera,
    n_samples: int,
    ground_y: float | None,
) -> list[np.ndarray]:
    """Render ``n_samples`` evenly spaced frames of one clip.

    ``ground_y``, when given, draws a horizontal line at that world height. It is a
    preview aid only and is never added to a study stimulus: without it a body in a
    featureless frame reads as floating even when it is exactly where the tracking put
    it, because there is nothing for the eye to measure against. The line is computed
    from the projection the renderer itself uses, not guessed, so it lands where the
    feet actually are.
    """
    verts, faces = seq["vertices"], seq["faces"]
    idx = np.unique(np.linspace(0, len(verts) - 1, n_samples).round().astype(int))
    tiles = [
        render_frame_fast(verts[int(t)], faces, width=width, height=height, camera=camera)
        for t in idx
    ]
    if ground_y is None:
        return tiles

    from seam.avatar.render import _project

    # Project a single point on the ground under the body. A point near the body rather
    # than two far-apart ones: `_project` clamps depth for anything behind the camera, so
    # a distant probe returns a meaningless row and averaging it with a near one is
    # nonsense - measured, that put the line at screen row 84645 in a 360-row image.
    probe = np.array([[0.0, ground_y, 0.0]])
    xy, _ = _project(probe, camera, width, height)
    row = round(float(xy[0, 1]))
    if 0 <= row < height:
        thickness = max(1, height // 160)
        for img in tiles:
            img[row : min(row + thickness, height)] = np.array(
                [120, 120, 125], dtype=img.dtype
            )
    return tiles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stimuli", required=True, help="directory holding *.glb")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--clip", action="append", default=None, help="clip id; repeatable")
    ap.add_argument("--samples", type=int, default=5, help="frames per clip")
    ap.add_argument("--width", type=int, default=260)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--limit", type=int, default=4, help="clips shown when --clip is absent")
    ap.add_argument(
        "--ground-y",
        type=float,
        default=None,
        help=(
            "world height of the ground line, drawn for readability. Default: the lowest "
            "mesh vertex seen across the clips, which is where the feet actually are."
        ),
    )
    ap.add_argument(
        "--no-ground",
        action="store_true",
        help="omit the ground line entirely",
    )
    args = ap.parse_args()

    stimuli = Path(args.stimuli)
    paths = sorted(stimuli.glob("*.glb"))
    if args.clip:
        wanted = set(args.clip)
        paths = [p for p in paths if p.stem in wanted]
    else:
        paths = paths[: args.limit]
    if not paths:
        print(f"no *.glb in {stimuli}", file=sys.stderr)
        return 1

    camera = Camera(distance_m=2.6, fov_deg=42.0, yaw_deg=30.0, pitch_deg=8.0)

    sequences = {p: _load_mesh_sequence(p) for p in paths}

    ground_y = args.ground_y
    if ground_y is None and not args.no_ground:
        ground_y = min(
            float(seq["vertices"][:, :, 1].min()) for seq in sequences.values()
        )
        print(f"ground line at y = {ground_y:+.3f} m (lowest vertex seen)")

    rows: list[np.ndarray] = []
    for path, seq in sequences.items():
        tiles = render_clip_tiles(
            seq,
            width=args.width,
            height=args.height,
            camera=camera,
            n_samples=args.samples,
            ground_y=ground_y,
        )
        rows.append(np.concatenate(tiles, axis=1))
        print(f"  {path.stem}: {len(seq['vertices'])} frames -> {len(tiles)} tiles")

    w = max(r.shape[1] for r in rows)
    rows = [
        np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255)
        for r in rows
    ]
    sheet = np.concatenate(rows, axis=0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image  # type: ignore

        Image.fromarray(sheet.astype(np.uint8)).save(out)
    except ImportError:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        plt.imsave(out, sheet)
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
