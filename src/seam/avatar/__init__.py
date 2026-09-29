"""Expressive avatar driven by tracked keypoints.

The target is **SMPL-X**, chosen because it is a composition of parts that match our
inputs one-to-one: SMPL body from pose keypoints, MANO hands from the 21-point hand
skeletons, FLAME face from the blendshape coefficients. No single-source model supplies
all three, and bridging them would mean guessing.

The model files are licence-gated by the Max Planck Institute and are deliberately not
vendored; :func:`seam.avatar.synthesis.load_smplx` takes a path the operator supplies.

The three documented retargeting traps - gimbal flips, joint limits, and finger
distortion - are handled in :mod:`seam.avatar.synthesis` and each has a test, because
all three fail silently: the animation still renders, it is just wrong.
"""

from seam.avatar.synthesis import (
    SmplxFrame,
    export_glb,
    load_smplx,
    retarget_body,
    retarget_expression,
    retarget_hand,
    synthesise,
    synthesise_sequence,
)

__all__ = [
    "SmplxFrame",
    "export_glb",
    "load_smplx",
    "retarget_body",
    "retarget_expression",
    "retarget_hand",
    "synthesise",
    "synthesise_sequence",
]
