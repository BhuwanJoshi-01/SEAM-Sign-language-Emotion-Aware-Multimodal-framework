"""Measure the signer's posture from tracked landmarks: seated or standing?

**Conclusion after checking real frames: every EmoSign signer is SEATED.**

Video frames of the lowest-, middle- and highest-ranked clips by hip-to-knee drop
(29423, 31657453, 1372, 1677, 28816, 841720, 29052074, 7985731) all show an adult
sitting in an office chair, a sheet of paper on the lap, legs crossed or stretched
forward. The single highest-ranked clip (7985731, ratio 0.94, "most standing") is
visibly a seated man with one ankle across the opposite knee.

So the hip-to-knee ratio below does NOT separate two postures. It measures how well
MediaPipe tracked the lower body, which is occluded by the chair seat and the paper.
Clips that *look* standing (high ratio, hip->ankle ~1.4-1.5) are the ones where the
tracker hallucinated a full-length leg through the chair; clips with a low ratio are
the ones where it partially followed the real cross-legged shape.

This matters for FRAMING, not for correctness. The retargeting reproduces whatever the
tracking reports, so an avatar whose legs hang straight down is faithfully playing back
a hallucinated standing leg. The fix is at the *input* end - either mask and exclude the
occluded lower body, or accept that the legs are untrustworthy and frame the avatar from
the hips up to match the source videos, which are themselves mostly waist-up.

Run:
    PYTHONPATH=src ./.venv/Scripts/python.exe scripts/posture_histogram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

DATA = Path("C:/Users/nancy/Desktop/.seam_data/emosign/landmarks")

# MediaPipe pose indices
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_SH, R_SH = 11, 12
L_WR, R_WR = 15, 16


def main() -> int:
    paths = sorted(DATA.glob("*.npz"))
    if not paths:
        print(f"no landmarks in {DATA}", file=sys.stderr)
        return 1

    print(f"{'clip':>10}  {'hip->knee':>9}  {'hip->ankle':>10}  {'knee/ankle':>10}  leg-tracking")
    print(f"{'':>10}  {'(ratio)':>9}  {'(ratio)':>10}  {'ankle drop':>10}")
    ratios = []

    for path in paths:
        with np.load(path) as fh:
            lm = np.asarray(fh["landmarks"], dtype=np.float64)
        pose = lm[:, 0:33, :2]  # (T, 33, 2), pose-first layout

        hip = (pose[:, L_HIP] + pose[:, R_HIP]) / 2.0
        knee = (pose[:, L_KNEE] + pose[:, R_KNEE]) / 2.0
        ankle = (pose[:, L_ANKLE] + pose[:, R_ANKLE]) / 2.0
        sh = (pose[:, L_SH] + pose[:, R_SH]) / 2.0

        # Image y grows downward, so "below" is a positive difference.
        torso = np.linalg.norm(sh - hip, axis=1)
        good = torso > 1e-9
        if not np.any(good):
            continue
        knee_drop = float(np.median((knee[good, 1] - hip[good, 1]) / torso[good]))
        ankle_drop = float(np.median((ankle[good, 1] - hip[good, 1]) / torso[good]))
        shin_drop = float(np.median((ankle[good, 1] - knee[good, 1]) / torso[good]))
        ratios.append(knee_drop)

        # The label reports how much leg the tracker invented, not the signer's posture.
        # A torso-length of leg below the hip is more than a seated lap allows.
        label = "full-leg (invented)" if knee_drop > 0.60 else "partial leg"
        if len(ratios) <= 10 or knee_drop > 0.60:
            print(
                f"{path.stem:>10}  {knee_drop:9.2f}  {ankle_drop:10.2f}  {shin_drop:10.2f}  {label}"
            )

    r = np.array(ratios)
    print()
    print(f"clips measured        : {len(r)}")
    print(f"median hip->knee      : {np.median(r):.2f} torso-lengths")
    print(f"ratio > 0.60 (full leg): {(r > 0.60).sum()}")
    print(f"range                 : {r.min():.2f} .. {r.max():.2f}")
    print()
    print("All signers are seated (verified against real frames). This table measures")
    print("leg-tracking quality, not posture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
