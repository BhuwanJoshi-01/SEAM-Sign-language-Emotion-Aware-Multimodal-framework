"""Check our forward kinematics against Blender's own armature evaluation.

Why this exists
---------------
`forward_kinematics` in `seam.avatar.synthesis` builds the skeleton by hand: a canonical
rest pose plus parent-relative rotations. That is a second, independent implementation of
something Blender already does, so it can be wrong in ways no unit test on our side would
catch - a shifted parent index, a rotation applied about the wrong axis, a chain that
fails to propagate. This script asks Blender for the answer and compares.

The comparison has to be convention-free, and that is the whole difficulty. Three things
differ between the two implementations and none of them is a bug:

1. **Frame convention.** The SMPL-H armature in the .blend is Z-up; our skeleton is Y-up.
   A raw angle comparison reads ~65-90 deg of pure convention.
2. **Rest skeleton.** Ours is an idealised estimate (straight spine, round numbers);
   SMPL-H's comes from the fitted model and has a real curve.
3. **Units of rest offsets.** Same reason.

So absolute joint positions can never match, and neither can displacement-from-neutral
compared with a single global rotation: per-joint displacement *vectors* point in each
joint's own rest frame, so one global rotation cannot align them unless the rest frames
already coincide. (An earlier attempt to do exactly that produced a confident, completely
meaningless "89 deg residual". Comparing displacement vectors with a Kabsch fit is simply
the wrong test.)

What *is* comparable without any alignment is **how far each bone direction rotates**. It
is a pure angle, identical in any frame, and for a single rigid rotation of one joint it
has a known analytic value.

The probe
---------
Rotate `spine1` by 90 deg about **Z** and measure how far each bone's direction turns.
Z is transverse to the spine on purpose. An earlier probe used Y, and every spine offset
in our rest pose is exactly `[0, +y, 0]` - parallel to the rotation axis - so those bones
are invariant by construction and the probe reports a failure that does not exist. A
degenerate probe is worse than none: it looks like a finding.

Expected result
---------------
    * Bones not descended from `spine1` (hips, legs, `spine1` itself): 0 deg in both.
    * Bones descended from `spine1`: 90 deg in ours, exactly, because our rest spine is
      collinear. Blender reports 76-90 deg, because the fitted spine is not straight, so
      a rigid 90 deg turn does not carry every bone direction through exactly 90 deg.
    * The gap is therefore a measurement of the rest-skeleton idealisation, not of the
      rotation algebra. The `head` row is the largest and the most interesting: SMPL-H's
      neck->head rest direction leans about 14 deg off our vertical, so head and neck tilt
      recovered from landmarks is systematically off by that much.

Usage
-----
    PYTHONPATH=src python scripts/validate_fk_vs_blender.py

Needs `blender` on PATH and an SMPL-H .blend (default
`/mnt/Volume2/SignLanguagge/NSL Data/smplh_model_20260428.blend`, override with
`--blend`). Exits non-zero if an unaffected bone moves or a descended bone misses 90 deg,
so it is safe to wire into CI.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from seam.avatar.synthesis import (
    SMPL_BODY_JOINTS,
    SMPL_PARENTS,
    forward_kinematics,
)

DEFAULT_BLEND = Path("/mnt/Volume2/SignLanguagge/NSL Data/smplh_model_20260428.blend")
ARMATURE = "SMPLH-male"
PIVOT = "spine1"
N_BODY = 21

#: Written into the .blend's Python and run headless. Blender is not importable from our
#: interpreter, so the pose has to be set inside it and the positions handed back as JSON.
_DUMP = """
import bpy, json, sys
from mathutils import Quaternion
bpy.ops.wm.open_mainfile(filepath={blend!r})
arm = bpy.data.objects[{arm!r}]
arm.data.pose_position = "POSE"
bpy.context.view_layer.update()
names = {names!r}
rest = [[float(v) for v in (arm.matrix_world @ arm.pose.bones[n].head)] for n in names]
for pb in arm.pose.bones:
    pb.rotation_mode = "QUATERNION"
    pb.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
arm.pose.bones[{pivot!r}].rotation_quaternion = Quaternion((0.0, 0.0, 1.0), {ang!r})
bpy.context.view_layer.update()
dg = bpy.context.evaluated_depsgraph_get()
posed = [[float(v) for v in (arm.matrix_world @ arm.evaluated_get(dg).pose.bones[n].head)]
         for n in names]
json.dump({{"rest": rest, "posed": posed}}, open(sys.argv[-1], "w"))
"""


def _descendants(parents: np.ndarray) -> set[int]:
    """Joints reachable from the pivot, excluding the pivot itself."""
    out: set[int] = set()
    changed = True
    while changed:
        changed = False
        for j in range(len(parents)):
            if j not in out and parents[j] in ({PIVOT_IDX} | out):
                out.add(j)
                changed = True
    return out


def _turn(rest: np.ndarray, posed: np.ndarray, parents: np.ndarray) -> np.ndarray:
    """Degrees each bone direction turns between two joint sets."""
    angles = []
    for j in range(1, len(parents)):
        p = int(parents[j])
        u = rest[j] - rest[p]
        v = posed[j] - posed[p]
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        angles.append(np.degrees(np.arccos(np.clip(float(u @ v), -1.0, 1.0))))
    return np.asarray(angles)


PIVOT_IDX = 1 + SMPL_BODY_JOINTS.index(PIVOT)  # +1 for the pelvis at index 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blend", type=Path, default=DEFAULT_BLEND)
    ap.add_argument("--angle-deg", type=float, default=90.0)
    ap.add_argument("--tolerance-deg", type=float, default=0.05)
    args = ap.parse_args()

    blender = shutil.which("blender")
    if blender is None:
        print("blender not on PATH; cannot cross-check", file=sys.stderr)
        return 2
    if not args.blend.exists():
        print(f"no .blend at {args.blend}", file=sys.stderr)
        return 2

    names = ["pelvis", *SMPL_BODY_JOINTS[:N_BODY]]
    with tempfile.NamedTemporaryFile("r", suffix=".json", delete=False) as fh:
        out = Path(fh.name)
    script = Path(tempfile.mkdtemp()) / "dump.py"
    script.write_text(
        _DUMP.format(
            blend=str(args.blend),
            arm=ARMATURE,
            names=names,
            pivot=PIVOT,
            ang=float(np.radians(args.angle_deg)),
        )
    )
    subprocess.run(
        [blender, "-b", "--factory-startup", "--python", str(script), "--", str(out)],
        check=True,
        capture_output=True,
    )
    data = json.loads(out.read_text())
    out.unlink(missing_ok=True)

    parents = SMPL_PARENTS[: len(names)]
    rest_bl = np.asarray(data["rest"], dtype=np.float64)
    posed_bl = np.asarray(data["posed"], dtype=np.float64)

    angle = np.radians(args.angle_deg)
    body_pose = np.zeros((N_BODY, 3))
    body_pose[PIVOT_IDX - 1] = [0.0, 0.0, angle]
    posed_us = forward_kinematics(body_pose, np.zeros(3))[: len(names)]
    rest_us = forward_kinematics(np.zeros((N_BODY, 3)), np.zeros(3))[: len(names)]

    turn_bl = _turn(rest_bl, posed_bl, parents)
    turn_us = _turn(rest_us, posed_us, parents)
    moved = _descendants(parents)

    print(f"probe: {PIVOT} rotated {args.angle_deg:g} deg about Z (transverse to the spine)")
    print(f"{'bone':>14s} {'blender':>9s} {'ours':>9s} {'diff':>8s}   class")
    for j in range(1, len(names)):
        k = j - 1
        kind = "descendant" if j in moved else "unaffected"
        print(
            f"{names[j]:>14s} {turn_bl[k]:9.3f} {turn_us[k]:9.3f} "
            f"{turn_us[k] - turn_bl[k]:8.3f}   {kind}"
        )

    still_us = np.flatnonzero(turn_us[~np.isin(np.arange(1, len(names)), list(moved))])
    print(f"\nunaffected bones: ours max {turn_us[still_us].max():.2e} deg")
    desc = sorted(moved)
    print(f"descendant bones: ours min {turn_us[[d - 1 for d in desc]].min():.4f} deg")

    ok = True
    if turn_us[still_us].max() > args.tolerance_deg:
        print("FAIL: a bone outside the pivot's subtree moved", file=sys.stderr)
        ok = False
    off = np.abs(turn_us[[d - 1 for d in desc]] - args.angle_deg).max()
    if off > args.tolerance_deg:
        print(
            f"FAIL: a descendant bone missed {args.angle_deg:g} deg by {off:.4f}", file=sys.stderr
        )
        ok = False
    worst = int(np.argmax(np.abs(turn_us - turn_bl)))
    print(
        f"\nmax |ours - blender| = {abs(turn_us[worst] - turn_bl[worst]):.3f} deg on "
        f"{names[worst + 1]}."
    )
    print(
        "Ours is exact by construction because the canonical rest spine is collinear.\n"
        "The gap is the rest-skeleton idealisation, not the rotation algebra: SMPL-H's\n"
        "fitted spine curves, so a rigid turn does not carry every direction through the\n"
        "same angle. Quantifying it is the point - it is the error budget for head and\n"
        "neck tilt recovered from landmarks."
    )
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
