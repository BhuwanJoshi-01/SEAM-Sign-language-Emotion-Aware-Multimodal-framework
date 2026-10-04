"""Render the rest pose from four yaws, so the facing is obvious by eye.

`check_facing.py` measures the geometry, but a number like "dot = +0.858" is hard to
trust without seeing it. The rest pose is the honest test: a neutral standing body with
no pose applied. If yaw 0 shows a face and yaw 180 shows the back of the head, then the
convention is settled and the study renderer's default can be judged against it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

#: Licence-gated SMPL-X model. `SEAM_SMPLX_MODEL` first, then the author's local
#: location. The original was a single hardcoded Windows desktop path, which made this
#: script unrunnable on any other machine and on CI; the fallback is kept so it still
#: works where it was written.
MODEL = Path(
    os.environ["SEAM_SMPLX_MODEL"]
    if os.environ.get("SEAM_SMPLX_MODEL")
    else r"C:/Users/nancy/Desktop/.models/smplx"
)
OUT = Path(os.environ.get("SEAM_TMP", "/tmp/posturecheck"))


def main() -> int:
    from PIL import Image

    from seam.avatar.mesh import smplx_mesh
    from seam.avatar.render import Camera, render_frame
    from seam.avatar.synthesis import SmplxFrame, load_smplx

    model = load_smplx(MODEL)
    seq = smplx_mesh([SmplxFrame()], model)
    v = np.asarray(seq.vertices[0], dtype=np.float64)
    f = np.asarray(seq.faces, dtype=np.int64)

    size = 340
    tiles = []
    for yaw in (0, 90, 180, 270):
        cam = Camera(yaw_deg=float(yaw), pitch_deg=0.0, distance_m=2.6, fov_deg=42)
        img = render_frame(v, f, width=size, height=size, camera=cam)
        tiles.append((yaw, img))

    canvas = Image.new("RGB", (size * 4 + 12, size), (16, 16, 16))
    for i, (_yaw, img) in enumerate(tiles):
        canvas.paste(Image.fromarray(np.asarray(img, dtype=np.uint8)), (i * (size + 4), 0))
    canvas.save(OUT / "rest_yaws.png")
    print(f"wrote {OUT / 'rest_yaws.png'}  (order: yaw 0, 90, 180, 270)")

    # Also render the study's own default, for direct comparison.
    cam = Camera()  # yaw 30, pitch 8
    img = render_frame(v, f, width=size * 2, height=size * 2, camera=cam)
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(OUT / "rest_study_default.png")
    print(f"wrote {OUT / 'rest_study_default.png'}  (study default: yaw 30, pitch 8)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
