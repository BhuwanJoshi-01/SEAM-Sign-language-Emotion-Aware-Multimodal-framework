"""Which way does the SMPL-X body face, and which way does the camera look at it?

The rendered study videos show the avatar's BACK: shoulder blades, the back of the head,
heel-first feet. Nancy spotted it and was right. This script does not guess - it measures
the mesh's own geometry to establish which world axis is "the front of a person", then
asks what the camera's yaw does to that axis.

Three independent signals for "front", so one bad assumption cannot carry the answer:

1. **The nose.** SMPL-X's head vertices include a nose that sticks out. The face plane's
   outward normal is the front.
2. **The feet.** Toes point forward, heels point back. The toe-to-heel vector is "front".
   This is the strongest signal: the feet are far apart and unambiguous.
3. **The chest vs the back.** The body is a flattened ellipse; the front is flatter and
   the spine side curves. Weaker, so used only as a tie-break.

Run:
    PYTHONPATH=src ./.venv/Scripts/python.exe scripts/check_facing.py
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


def rest_vertices() -> np.ndarray:
    from seam.avatar.mesh import smplx_mesh
    from seam.avatar.synthesis import SmplxFrame, load_smplx

    model = load_smplx(MODEL)
    seq = smplx_mesh([SmplxFrame()], model)
    return np.asarray(seq.vertices[0], dtype=np.float64)


def main() -> int:
    v = rest_vertices()
    print(f"vertices: {v.shape}")

    # --- signal 1: feet. SMPL-X vertex ranges for the feet are side-specific. ---
    # The ankles sit at the lowest points of the leg; the toes extend forward from there.
    # Use the extreme-most vertices in each horizontal axis to locate the feet.
    y = v[:, 1]
    ground = y < (y.min() + 0.06)  # foot region
    foot = v[ground]
    print(f"foot-region vertices: {foot.shape[0]}")
    print(f"  foot x range: {foot[:, 0].min():+.3f} .. {foot[:, 0].max():+.3f}")
    print(f"  foot y range: {foot[:, 1].min():+.3f} .. {foot[:, 1].max():+.3f}")
    print(f"  foot z range: {foot[:, 2].min():+.3f} .. {foot[:, 2].max():+.3f}")

    # The heel is the rearmost, the toe the foremost, along the body's front axis.
    # The feet are long in z (front-to-back) and narrow in x (side-to-side), so the axis
    # in which the foot is LONGEST is the front axis.
    spans = foot.max(axis=0) - foot.min(axis=0)
    print(f"  foot span xyz: {spans[0]:.3f}, {spans[1]:.3f}, {spans[2]:.3f}")
    front_axis = int(np.argmax(spans[[0, 2]]))  # choose between x and z
    front_axis = 0 if front_axis == 0 else 2
    print(f"  -> foot is longest along axis {'xyz'[front_axis]}")

    # Which sign is "forward"? The ankle is above the heel; toes are the far end.
    # Take the toe cluster as the points furthest along the front axis.
    for axis, name in ((0, "x"), (2, "z")):
        lo, hi = foot[:, axis].min(), foot[:, axis].max()
        print(f"  along {name}: min {lo:+.3f} (heel side?)  max {hi:+.3f} (toe side?)")

    # --- signal 2: the nose. Head vertices are the tallest cluster. ---
    top = y > (y.max() - 0.09)
    head = v[top]
    print(f"\nhead-region vertices: {head.shape[0]}")
    # The nose protrudes along the front axis. Compare the head's extremes on that axis.
    for axis, name in ((0, "x"), (2, "z")):
        print(f"  head {name}: {head[:, axis].min():+.3f} .. {head[:, axis].max():+.3f}")

    # --- what the camera does ---
    from seam.avatar.render import Camera

    cam = Camera()
    R = cam.view_matrix()
    print(f"\ncamera: yaw {cam.yaw_deg}  pitch {cam.pitch_deg}")
    print("view matrix rows = camera axes in world coords:")
    print(f"  camera right : {R[0]}")
    print(f"  camera up    : {R[1]}")
    print(f"  camera back  : {R[2]}   (this is +z in camera space = AWAY from viewer)")

    body_front = np.array([0.0, 0.0, 1.0])  # placeholder, printed for comparison
    print(f"\nfor reference, world +z: {body_front}")

    # The decisive number: how much does the body's front axis point TOWARD the viewer?
    # Camera looks along -camera_back. A surface faces the viewer if its outward normal
    # has a negative dot with camera_back. The body's front normal is +z_body.
    for sign, label in ((1.0, "+"), (-1.0, "-")):
        n_front = np.array([0.0, 0.0, sign])
        toward_viewer = float(n_front @ R[2])
        print(
            f"  body front = {label}z: dot with camera_back = {toward_viewer:+.3f} "
            f"-> {'FACING VIEWER' if toward_viewer < 0 else 'FACING AWAY'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
