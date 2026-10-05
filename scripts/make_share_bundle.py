#!/usr/bin/env python
"""Pack everything a teammate needs to run SEAM on another computer into one zip.

    python scripts/make_share_bundle.py                # dist/SEAM_share.zip
    python scripts/make_share_bundle.py --with-avatar  # also the animated avatar clips

The recipient extracts it, runs `setup_windows.bat` or `./setup_linux.sh`, then
`run_demo`. Nothing has to be trained: the three trained emotion models, their ONNX
exports, MediaPipe's model files and every result file are inside.

What goes in
------------
* every file git tracks or would track (the code, the website, the docs, the tests);
* MediaPipe's three `.task` bundles (Apache-2.0), so Python-side tracking works offline;
* the trained emotion models and their ONNX exports;
* the result files (`artifacts/{audit,m3,m4,m5a,bench,reports,superseded}`).

What stays out, always
----------------------
* dataset video and the SignStream annotation files: their terms forbid passing them on;
* cached video frames and per-clip model outputs derived from those videos;
* the SMPL-X body model and SMPLer-X's weights: licensed to each person individually.

What stays out unless asked for
-------------------------------
* the animated avatar clips (`--with-avatar`). A skinned glTF contains the SMPL-X mesh,
  so it is the licensed model in another file format. Pass the flag only when the person
  receiving the zip holds the SMPL-X licence too. The zip's manifest records the choice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOP = "SEAM"

#: Result folders whose JSON files are copied. Small, and the tests read them.
RESULT_DIRS = ("audit", "m3", "m4", "m5a", "bench", "reports", "superseded", "export")
MEDIAPIPE = ("face_landmarker.task", "hand_landmarker.task", "pose_landmarker_full.task")
#: What the website alone needs; the pins are read from pyproject.toml, never typed here.
LITE = ("numpy", "fastapi", "uvicorn", "websockets")
#: Never packed, whatever git says: other tools' folders and anything a dataset could hide in.
NEVER = re.compile(r"(^|/)(\.claude|\.git|\.venv|dist|graphify-out|__pycache__)(/|$)|\.pyc$")


def git_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    files = [REPO / f for f in out.split("\0") if f and not NEVER.search(f)]
    return [f for f in files if f.is_file()]


def mediapipe_dir(explicit: Path | None) -> Path | None:
    sys.path.insert(0, str(REPO / "src"))
    from seam.perception import tasks_api

    roots = [explicit] if explicit else []
    roots += [REPO / "models" / "mediapipe", *tasks_api._SEARCH_DIRS]
    for root in roots:
        if root and all((root / name).is_file() for name in MEDIAPIPE):
            return root
    return None


def lite_requirements() -> str:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    pins = dict(re.findall(r'"([A-Za-z0-9_.-]+)==([^"]+)"', text))
    missing = [name for name in LITE if name not in pins]
    if missing:
        raise SystemExit(f"pyproject.toml has no pin for {missing}")
    lines = ["# What the website needs. Generated from pyproject.toml; do not edit."]
    lines += [f"{name}=={pins[name]}" for name in LITE]
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=REPO / "dist" / "SEAM_share.zip")
    ap.add_argument("--mediapipe-dir", type=Path, default=None)
    ap.add_argument(
        "--with-avatar",
        action="store_true",
        help="include the animated avatar clips; they contain the licensed SMPL-X mesh",
    )
    args = ap.parse_args()

    entries: dict[str, Path] = {}
    for f in git_files():
        entries[f.relative_to(REPO).as_posix()] = f

    art = REPO / "artifacts"
    for sub in RESULT_DIRS:
        for f in sorted((art / sub).rglob("*")):
            if f.is_file() and f.suffix in {".json", ".onnx"}:
                entries[f.relative_to(REPO).as_posix()] = f
    for f in [*sorted((art / "fer").glob("*.pt")), art / "fer" / "metrics.json"]:
        if f.is_file():
            entries[f.relative_to(REPO).as_posix()] = f

    tasks = mediapipe_dir(args.mediapipe_dir)
    if tasks is None:
        print("MediaPipe .task files not found; pass --mediapipe-dir. Packing without them.")
    else:
        for name in MEDIAPIPE:
            entries[f"models/mediapipe/{name}"] = tasks / name

    avatar: list[str] = []
    if args.with_avatar:
        demo = art / "m7a" / "demo"
        for f in [*sorted(demo.glob("*.glb")), demo / "manifest.json"]:
            if f.is_file():
                entries[f.relative_to(REPO).as_posix()] = f
                avatar.append(f.name)

    # A last look at what is about to leave this machine.
    banned = [
        k
        for k in entries
        if re.search(r"\.(mp4|mov|avi|webm|mkv|npz|pkl)$", k)
        or "/frames/" in k
        or "signstream_xml" in k
        or re.search(r"SMPLX_|smpler_x|\.xml$", k)
    ]
    if banned:
        print("refusing to pack these; they look like dataset or licensed model files:")
        for k in banned[:20]:
            print("  ", k)
        return 1
    if not args.with_avatar and any(k.endswith(".glb") for k in entries):
        print("refusing: a .glb is in the file list without --with-avatar")
        return 1

    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
        ).stdout.strip()
    )
    manifest = {
        "made": time.strftime("%Y-%m-%d %H:%M"),
        "git_commit": commit + ("+uncommitted changes" if dirty else ""),
        "files": len(entries) + 2,
        "models": {
            k: {"bytes": v.stat().st_size, "sha256": sha256(v)}
            for k, v in sorted(entries.items())
            if k.endswith((".task", ".pt", ".onnx"))
        },
        "avatar_clips_included": avatar,
        "left_out": {
            "dataset video and SignStream annotation files": "their terms forbid passing them on",
            "cached video frames and per-clip model outputs": "derived from those videos",
            "SMPL-X body model, SMPLer-X weights": "licensed to each person individually",
            **(
                {}
                if avatar
                else {
                    "animated avatar clips (.glb)": "they contain the SMPL-X mesh; rebuild the "
                    "zip with --with-avatar only for someone who holds that licence"
                }
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel, src in sorted(entries.items()):
            info = zipfile.ZipInfo.from_file(src, f"{TOP}/{rel}")
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if (src.stat().st_mode & stat.S_IXUSR or rel.endswith(".sh")) else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            z.writestr(info, src.read_bytes())
            total += src.stat().st_size
        z.writestr(f"{TOP}/requirements-lite.txt", lite_requirements())
        z.writestr(f"{TOP}/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1) + "\n")

    size = args.out.stat().st_size
    print(f"wrote {args.out}")
    print(f"  {len(entries) + 2} files, {total / 1e6:.1f} MB unpacked, {size / 1e6:.1f} MB zipped")
    print(f"  models: {len(manifest['models'])}   avatar clips: {len(avatar)}")
    print("  left out:")
    for what, why in manifest["left_out"].items():
        print(f"    - {what}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
