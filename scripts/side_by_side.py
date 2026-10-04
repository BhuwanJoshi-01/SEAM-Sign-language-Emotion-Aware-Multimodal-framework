"""Build a side-by-side sheet: real EmoSign frame next to the retargeted avatar.

Nancy asked to see the avatar video beside the real clip before committing to the full
148-clip render. This puts one real frame and one rendered avatar frame next to each
other for several clips, so the comparison is a single image rather than two windows.

The avatar tiles are rendered at the same size and stacked vertically, so the eye can
compare limb positions directly instead of guessing across differently-scaled pictures.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

TMP = Path(os.environ.get("SEAM_TMP", "/tmp/posturecheck"))
STIM = Path(os.environ.get("SEAM_STIMULI", "artifacts/m7a/study/public"))
VIDEO = Path(os.environ.get("SEAM_VIDEO", ""))
FFMPEG = Path(
    "C:/Users/nancy/AppData/Local/Microsoft/WinGet/Packages/"
    "Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe/ffmpeg-9.0.1-full_build/bin/ffmpeg.exe"
)

CLIPS = ["7985731", "29423", "1372"]
SIDE = 480


def real_frame(clip: str, out: Path) -> None:
    """Grab one frame from the real source video, letterboxed to a square."""
    src = VIDEO / f"{clip}.mp4"
    vf = (
        f"select=eq(n\\,10),"
        f"scale={SIDE}:{SIDE}:force_original_aspect_ratio=decrease,"
        f"pad={SIDE}:{SIDE}:(ow-iw)/2:(oh-ih)/2:color=0x1a1a1a"
    )
    subprocess.run(
        [
            str(FFMPEG),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            vf,
            "-frames:v",
            "1",
            str(out),
        ],
        check=True,
    )


def main() -> int:
    from make_study_stimuli import _load_mesh_sequence

    from seam.avatar.render import Camera, render_frame

    TMP.mkdir(parents=True, exist_ok=True)

    rows = []
    for clip in CLIPS:
        real_png = TMP / f"sbs_{clip}_real.png"
        real_frame(clip, real_png)

        seq = _load_mesh_sequence(STIM / f"{clip}.glb")
        t = seq["vertices"].shape[0] // 2
        # yaw 0 so the avatar faces the camera the same way the real signer does.
        cam = Camera(yaw_deg=0.0, pitch_deg=0.0, distance_m=2.6, fov_deg=42)
        img = render_frame(seq["vertices"][t], seq["faces"], width=SIDE, height=SIDE, camera=cam)
        avatar_png = TMP / f"sbs_{clip}_avatar.png"
        _write_png(img, avatar_png)

        rows.append((real_png, avatar_png))
        print(f"{clip}: real + avatar rendered")

    _tile(rows, TMP / "side_by_side.png")
    print(f"wrote {TMP / 'side_by_side.png'}")
    return 0


def _write_png(img: np.ndarray, path: Path) -> None:
    """Save an (H, W, 3) uint8 array without pulling in an image library."""
    from PIL import Image

    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(path)


def _tile(rows: list[tuple[Path, Path]], out: Path) -> None:
    from PIL import Image

    gap = 8
    w = SIDE * 2 + gap
    h = SIDE * len(rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (w, h), (16, 16, 16))
    for i, (real_png, avatar_png) in enumerate(rows):
        y = i * (SIDE + gap)
        canvas.paste(Image.open(real_png).convert("RGB"), (0, y))
        canvas.paste(Image.open(avatar_png).convert("RGB"), (SIDE + gap, y))
    canvas = canvas.resize((w // 2, h // 2), Image.LANCZOS)
    canvas.save(out)


if __name__ == "__main__":
    raise SystemExit(main())
