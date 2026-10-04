"""Print the full hip-to-knee ratio distribution across every EmoSign clip.

`measure_posture.py` prints only a sample plus totals; to decide whether the signers are
seated or standing we need the *shape* of the distribution. Two postures would give two
humps; one posture with noisy tracking gives one hump.

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

L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_SH, R_SH = 11, 12
L_WR, R_WR = 15, 16


def ratios(path: Path) -> tuple[float, float, float]:
    with np.load(path) as fh:
        lm = np.asarray(fh["landmarks"], dtype=np.float64)
    pose = lm[:, 0:33, :2]
    hip = (pose[:, L_HIP] + pose[:, R_HIP]) / 2.0
    knee = (pose[:, L_KNEE] + pose[:, R_KNEE]) / 2.0
    ankle = (pose[:, L_ANKLE] + pose[:, R_ANKLE]) / 2.0
    sh = (pose[:, L_SH] + pose[:, R_SH]) / 2.0
    torso = np.linalg.norm(sh - hip, axis=1)
    good = torso > 1e-9
    if not np.any(good):
        return np.nan, np.nan, np.nan
    kd = float(np.median((knee[good, 1] - hip[good, 1]) / torso[good]))
    ad = float(np.median((ankle[good, 1] - hip[good, 1]) / torso[good]))
    # Hand height relative to the hip, a third opinion: seated signers keep hands low.
    wr = (pose[:, L_WR] + pose[:, R_WR]) / 2.0
    hd = float(np.median((hip[good, 1] - wr[good, 1]) / torso[good]))
    return kd, ad, hd


def main() -> int:
    paths = sorted(DATA.glob("*.npz"))
    rows = []
    for path in paths:
        kd, ad, hd = ratios(path)
        if np.isfinite(kd):
            rows.append((path.stem, kd, ad, hd))
    kds = np.array([r[1] for r in rows])
    ads = np.array([r[2] for r in rows])

    print(f"clips: {len(rows)}")
    print()
    print("hip->knee drop, in torso-lengths (0 = knee level with hip, ~1.0 = knee a torso below)")
    lo, hi = 0.25, 1.00
    bins = np.arange(lo, hi + 0.05, 0.05)
    counts, edges = np.histogram(kds, bins=bins)
    peak = max(counts.max(), 1)
    for c, a, b in zip(counts, edges[:-1], edges[1:], strict=False):
        bar = "#" * round(40 * c / peak)
        print(f"  {a:.2f}-{b:.2f}  {c:3d}  {bar}")

    print()
    print("hip->ankle drop (standing ~1.4, seated ~0.7)")
    bins2 = np.arange(0.5, 1.75, 0.125)
    counts2, edges2 = np.histogram(ads, bins=bins2)
    peak2 = max(counts2.max(), 1)
    for c, a, b in zip(counts2, edges2[:-1], edges2[1:], strict=False):
        bar = "#" * round(40 * c / peak2)
        print(f"  {a:.3f}-{b:.3f}  {c:3d}  {bar}")

    print()
    print("percentiles of hip->knee drop:")
    for p in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{p:<3d} {np.percentile(kds, p):.3f}")

    # The three lowest and three highest, to sanity-check the extremes by eye.
    order = np.argsort(kds)
    print()
    print("lowest 5 (most seated-looking):")
    for i in order[:5]:
        clip, kd, ad, hd = rows[i]
        print(f"  {clip:>9}  knee {kd:.3f}  ankle {ad:.3f}  hand {hd:.3f}")
    print("highest 5 (most standing-looking):")
    for i in order[-5:]:
        clip, kd, ad, hd = rows[i]
        print(f"  {clip:>9}  knee {kd:.3f}  ankle {ad:.3f}  hand {hd:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
