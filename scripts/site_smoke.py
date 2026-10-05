#!/usr/bin/env python
"""Drive the static live demo in headless Chrome with a fake camera, and check it works.

`docs/index.html` runs MediaPipe in the browser, so nothing server-side can tell whether
it works. This script can: it serves `docs/`, starts Chrome with a video file standing in
for the webcam, lets the page run, and reads back what the page measured. It fails if no
frame was processed or no face was found, and can save screenshots.

    python scripts/site_smoke.py --video clip.mp4 --shot docs/figures/demo_live.png

The video is converted to the raw format Chrome's fake camera needs. Nothing is uploaded:
Chrome reads the file locally and the page sends nothing anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import functools
import http.server
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SITE = REPO / "docs"

#: What the page reports about itself: its live readings, delegate, frame rate and events.
READ_STATE = (
    "JSON.stringify({s: window.__seam || null, "
    "d: document.getElementById('hDelegate')?.textContent, "
    "f: document.getElementById('hFps')?.textContent, "
    "msg: document.getElementById('idleMsg')?.textContent, "
    "ev: document.getElementById('events')?.innerText})"
)
SCROLL_TO_LIVE = "document.getElementById('live').scrollIntoView(); window.scrollBy(0,-70); 1"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def serve(directory: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    port = free_port()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, port


def to_fake_camera(video: Path, out: Path, seconds: int) -> None:
    """Re-encode to raw YUV4MPEG, the format Chrome's fake capture device reads and loops.

    MJPEG is documented as supported too, but an ffmpeg-written MJPEG gave a solid green
    frame here, which looks exactly like a page that has stopped drawing.
    """
    still = video.suffix.lower() in {".png", ".jpg", ".jpeg"}
    loop = ["-loop", "1"] if still else ["-stream_loop", "-1"]
    scale = (
        "scale=640:480:force_original_aspect_ratio=increase,crop=640:480"
        if still
        else "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *loop,
            "-i",
            str(video),
            "-t",
            str(seconds),
            "-vf",
            f"{scale},fps=15",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "yuv4mpegpipe",
            str(out),
        ],
        check=True,
        stdin=subprocess.DEVNULL,
        timeout=180,
    )


async def cdp(ws_url: str, calls: list[tuple[str, dict]]) -> list[dict]:
    import websockets

    out = []
    async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        for i, (method, params) in enumerate(calls, start=1):
            await ws.send(json.dumps({"id": i, "method": method, "params": params}))
            while True:
                msg = json.loads(await ws.recv())
                if msg.get("id") == i:
                    out.append(msg.get("result", msg))
                    break
    return out


def page_ws(port: int, wait: float = 20.0) -> str:
    deadline = time.time() + wait
    while time.time() < deadline:
        with contextlib.suppress(Exception):
            tabs = json.loads(
                urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2).read()
            )
            for t in tabs:
                if t.get("type") == "page":
                    return str(t["webSocketDebuggerUrl"])
        time.sleep(0.4)
    raise RuntimeError("Chrome's debugging endpoint did not come up")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--video", required=True, help="video or image file to use as the camera")
    ap.add_argument("--seconds", type=float, default=30.0, help="how long to let the page run")
    ap.add_argument("--shot", default=None, help="save a screenshot of the live panel here")
    ap.add_argument("--full-shot", default=None, help="save a full-page screenshot here")
    ap.add_argument("--skeleton", action="store_true", help="hide the camera image in the page")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args()

    chrome = (
        shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
        or shutil.which("chromium")
    )
    if chrome is None:
        print("no Chrome binary found")
        return 2

    httpd, port = serve(SITE)
    debug = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        cam = Path(tmp) / "camera.y4m"
        to_fake_camera(Path(args.video), cam, seconds=12)
        query = "autostart=camera" + ("&skeleton=1" if args.skeleton else "")
        url = f"http://127.0.0.1:{port}/index.html?{query}"
        proc = subprocess.Popen(
            [
                chrome,
                "--headless=new",
                f"--remote-debugging-port={debug}",
                "--no-first-run",
                f"--user-data-dir={tmp}/profile",
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-video-capture={cam}",
                "--enable-unsafe-swiftshader",
                "--ignore-gpu-blocklist",
                "--hide-scrollbars",
                f"--window-size={args.width},{args.height}",
                url,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            ws = page_ws(debug)
            state: dict = {}
            deadline = time.time() + args.seconds
            while time.time() < deadline:
                time.sleep(2.0)
                res = asyncio.run(
                    cdp(
                        ws,
                        [
                            (
                                "Runtime.evaluate",
                                {
                                    "expression": READ_STATE,
                                    "returnByValue": True,
                                },
                            )
                        ],
                    )
                )
                state = json.loads(res[0]["result"]["value"])
                frames = (state.get("s") or {}).get("frames", 0)
                print(
                    f"  t+{args.seconds - (deadline - time.time()):4.0f}s frames={frames} "
                    f"fps={state.get('f')} {state.get('d')}"
                )
            if args.shot or args.full_shot:
                calls = [
                    (
                        "Emulation.setDeviceMetricsOverride",
                        {
                            "width": args.width,
                            "height": args.height,
                            "deviceScaleFactor": 1,
                            "mobile": False,
                        },
                    )
                ]
                asyncio.run(cdp(ws, calls))
                if args.shot:
                    res = asyncio.run(
                        cdp(
                            ws,
                            [
                                (
                                    "Runtime.evaluate",
                                    {"expression": SCROLL_TO_LIVE},
                                ),
                                ("Page.captureScreenshot", {"format": "png"}),
                            ],
                        )
                    )
                    Path(args.shot).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.shot).write_bytes(base64.b64decode(res[1]["data"]))
                    print(f"wrote {args.shot}")
                if args.full_shot:
                    res = asyncio.run(
                        cdp(
                            ws,
                            [
                                (
                                    "Page.captureScreenshot",
                                    {"format": "png", "captureBeyondViewport": True},
                                )
                            ],
                        )
                    )
                    Path(args.full_shot).write_bytes(base64.b64decode(res[0]["data"]))
                    print(f"wrote {args.full_shot}")
        finally:
            proc.terminate()
            with contextlib.suppress(Exception):
                proc.wait(timeout=10)
            httpd.shutdown()

    s = state.get("s") or {}
    print(
        json.dumps(
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()}, indent=1
        )
    )
    print("events:", (state.get("ev") or "").replace("\n", " | ")[:400])
    failures = []
    if not s.get("frames"):
        failures.append(f"no frame was processed (page says: {state.get('msg')!r})")
    elif not s.get("face"):
        failures.append("frames were processed but no face was found")
    for f in failures:
        print("FAIL:", f)
    if not failures:
        print("site smoke test passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
