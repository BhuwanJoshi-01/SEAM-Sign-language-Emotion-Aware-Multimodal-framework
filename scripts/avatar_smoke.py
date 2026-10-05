#!/usr/bin/env python
"""Open the front page on the real server in headless Chrome and play every avatar clip.

`make serve-check` can tell that the page and the .glb files are served. It cannot tell
whether a clip actually loads and animates: the old viewer was reported to play one of the
four clips it listed, and nothing automated would have noticed either way. This starts the
server, opens `/`, waits for the 3D viewer to come up, then loads each clip in turn through
the viewer's own code and fails if any of them does not produce a posed, animated skeleton.

    python scripts/avatar_smoke.py --shot out.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import site_smoke as SS

ROOT = Path(__file__).resolve().parents[1]
STATE = "JSON.stringify(window.__avatar || null)"


def evaluate(ws: str, expression: str, *, wait: bool = False) -> object:
    res = asyncio.run(
        SS.cdp(
            ws,
            [
                (
                    "Runtime.evaluate",
                    {"expression": expression, "returnByValue": True, "awaitPromise": wait},
                )
            ],
        )
    )
    return res[0].get("result", {}).get("value")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shot", default=None, help="save a screenshot of the viewer here")
    ap.add_argument("--port", type=int, default=8078)
    args = ap.parse_args()

    chrome = (
        shutil.which("google-chrome")
        or shutil.which("google-chrome-stable")
        or shutil.which("chromium")
    )
    if chrome is None:
        print("no Chrome binary found")
        return 2

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "seam.cli",
            "serve",
            "--port",
            str(args.port),
            "--log-level",
            "warning",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
    )
    debug = SS.free_port()
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        browser = None
        try:
            deadline = time.time() + 40
            import urllib.request

            while time.time() < deadline:
                with contextlib.suppress(Exception):
                    urllib.request.urlopen(f"http://127.0.0.1:{args.port}/api/health", timeout=2)
                    break
                time.sleep(0.4)
            else:
                print("FAIL: server did not come up")
                return 1
            browser = subprocess.Popen(
                [
                    chrome,
                    "--headless=new",
                    f"--remote-debugging-port={debug}",
                    "--no-first-run",
                    f"--user-data-dir={tmp}/profile",
                    "--enable-unsafe-swiftshader",
                    "--ignore-gpu-blocklist",
                    "--hide-scrollbars",
                    "--window-size=1440,1000",
                    f"http://127.0.0.1:{args.port}/",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            ws = SS.page_ws(debug)
            state: dict = {}
            for _ in range(60):
                time.sleep(1.0)
                state = json.loads(evaluate(ws, STATE) or "null") or {}
                if state.get("ready") or state.get("error"):
                    break
            if not state.get("ready"):
                failures.append(f"the viewer never came up: {state.get('error') or state}")
            clips = state.get("clips") or []
            print(f"viewer ready={state.get('ready')} clips={clips}")
            for uid in clips:
                ok = evaluate(ws, f"window.__avatar.load({uid!r})", wait=True)
                time.sleep(0.6)
                now = json.loads(evaluate(ws, STATE) or "null") or {}
                moved = evaluate(
                    ws,
                    "(() => { const c = document.getElementById('av3d'); "
                    "return c && !c.hidden && c.width > 0 && c.height > 0; })()",
                )
                print(
                    f"  clip {uid}: loaded={ok} duration={now.get('duration', 0):.2f}s "
                    f"bones={now.get('bones')} canvas={moved} error={now.get('error')}"
                )
                if not ok or now.get("error"):
                    failures.append(f"clip {uid} did not load: {now.get('error')}")
                elif now.get("bones", 0) < 50 or now.get("duration", 0) <= 0.5:
                    failures.append(f"clip {uid} loaded without a full animated skeleton: {now}")
            if args.shot and clips:
                evaluate(ws, f"window.__avatar.load({clips[0]!r})", wait=True)
                evaluate(ws, "document.getElementById('avatar').scrollIntoView(); 1")
                time.sleep(2.5)
                res = asyncio.run(SS.cdp(ws, [("Page.captureScreenshot", {"format": "png"})]))
                Path(args.shot).write_bytes(base64.b64decode(res[0]["data"]))
                print(f"wrote {args.shot}")
        finally:
            if browser is not None:
                browser.terminate()
            server.terminate()
            with contextlib.suppress(Exception):
                server.wait(timeout=10)
    for f in failures:
        print("FAIL:", f)
    if not failures:
        print("avatar smoke test passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
