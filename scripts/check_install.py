#!/usr/bin/env python
"""Check that this copy of SEAM can run on this machine, and say what to do next.

Run by the setup scripts at the end, and safe to run by hand at any time:

    python scripts/check_install.py

It never downloads anything and changes nothing. It imports what each part of the project
needs, looks for the model files, starts the web app in-process and asks it for its front
page, and prints a line per check.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

#: What the live website and its server need.
SITE = ("numpy", "fastapi", "uvicorn")
#: What analysing a video, the tests and the experiments need on top.
FULL = ("mediapipe", "cv2", "torch", "onnxruntime", "sklearn", "scipy", "pandas", "PIL")


def models_dir() -> Path:
    env = os.environ.get("SEAM_MEDIAPIPE_MODELS")
    return Path(env) if env else ROOT / "models" / "mediapipe"


def have(module: str) -> bool:
    try:
        importlib.import_module(module)
    except Exception:
        return False
    return True


def main() -> int:
    ok = True
    v = sys.version_info
    good_python = (3, 11) <= (v.major, v.minor) <= (3, 13)
    print(f"[{'ok' if good_python else '!!'}] Python {v.major}.{v.minor}.{v.micro}", end="")
    print("" if good_python else "   (3.11 to 3.13 is what the pinned packages support)")

    site = [m for m in SITE if not have(m)]
    print(
        f"[{'ok' if not site else '!!'}] website packages", "" if not site else f"missing: {site}"
    )
    ok &= not site

    missing = [m for m in FULL if not have(m)]
    full = not missing
    print(
        f"[{'ok' if full else '--'}] full install (analyse a video, tests, experiments)",
        "" if full else f"not installed: {missing}",
    )

    tasks = [models_dir() / f for f in ("face_landmarker.task", "hand_landmarker.task")]
    tasks.append(models_dir() / "pose_landmarker_full.task")
    lost = [p.name for p in tasks if not p.is_file()]
    print(f"[{'ok' if not lost else '--'}] MediaPipe model files in {models_dir()}", end="")
    print("" if not lost else f"   missing {lost}; Python-side tracking would download them")

    fer = sorted((ROOT / "artifacts" / "fer").glob("*.pt"))
    onnx = sorted((ROOT / "artifacts" / "export").glob("*.onnx"))
    print(f"[{'ok' if fer else '--'}] trained emotion models: {len(fer)} PyTorch, {len(onnx)} ONNX")

    if not site:
        try:
            from seam.serve.app import build_app

            app = build_app()
            routes = {getattr(r, "path", "") for r in app.routes}
            page = ROOT / "docs" / "index.html"
            served = "/" in routes and page.is_file()
            print(f"[{'ok' if served else '!!'}] web app builds and has its front page")
            ok &= served
        except Exception as exc:  # the point of this script is to report, not to crash
            print(f"[!!] web app failed to build: {exc}")
            ok = False

    clips = sorted((ROOT / "artifacts" / "m7a" / "demo").glob("*.glb"))
    print(f"[{'ok' if clips else '--'}] 3D avatar clips: {len(clips)}", end="")
    print("" if clips else "   (not in this copy; the page shows rendered frames instead)")

    print()
    if not ok:
        print("Something needed for the website is missing. Re-run the setup script.")
        return 1
    win = os.name == "nt"
    print("Ready. Start the website with:", "run_demo.bat" if win else "./run_demo.sh")
    if full:
        tool = "scripts\\analyse_video.py" if win else "scripts/analyse_video.py"
        print(f"Analyse your own video with: python {tool} your_video.mp4")
    else:
        print("To analyse your own videos, run the setup script again without --lite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
