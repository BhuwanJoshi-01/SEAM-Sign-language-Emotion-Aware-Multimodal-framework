"""Boot the demo server and check it over HTTP, so `make serve-check` is a gate.

A demo that starts and then serves a 44-byte 404 page is worse than one that
refuses to start - it looks like it works. The first version of this server had
exactly that bug: the web asset path resolved to `src/` instead of the package, the
app built without complaint, and the page was empty.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PORT = 8079
NM_DIM = 52


def wait_up(port: int, timeout: float = 40.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=2).read()
            return True
        except Exception:
            time.sleep(0.4)
    return False


def main() -> int:
    proc = subprocess.Popen(
        [sys.executable, "-m", "seam.cli", "serve", "--port", str(PORT), "--log-level", "warning"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(ROOT),
    )
    failures: list[str] = []
    try:
        if not wait_up(PORT):
            print("FAIL: server did not come up")
            return 1

        page = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=15).read()
        print(f"GET /                 {len(page)} bytes")
        if b"<!doctype html>" not in page[:64].lower():
            failures.append("index did not serve an HTML document")
        if b"tasks-vision" not in page:
            failures.append("page does not load client-side MediaPipe")
        if b"api/analyse" not in page:
            failures.append("page does not post to /api/analyse")

        health = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=15).read()
        )
        got = health.get("expects", {}).get("blendshapes")
        if got != [None, NM_DIM]:
            failures.append(
                f"health contract mismatch: NM_DIM={NM_DIM}, got {got!r}; body={health!r}"
            )

        rng = np.random.default_rng(0)
        payload = {
            "blendshapes": (rng.random((30, NM_DIM)) * 0.2).tolist(),
            "head_pose": np.tile(np.eye(4).tolist(), (30, 1, 1)).tolist(),
            "fps": 24,
            "utterance_id": "24363254",
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/analyse",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )
        out = json.loads(urllib.request.urlopen(req, timeout=20).read())
        print(f"POST /api/analyse     {len(out['markers'])} markers, {out['server_ms']} ms")
        if out["affect"]["supported"] or out["recognition"]["supported"]:
            failures.append("demo is advertising predictions this build cannot support")
        if not out["provenance"]["input"]:
            failures.append("response carries no input provenance")

        # A malformed window must be a 400 with an actionable message, not a 500.
        bad = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/api/analyse",
            data=json.dumps({"blendshapes": [[0.1] * (NM_DIM - 1)] * 20, "fps": 24}).encode(),
            headers={"content-type": "application/json"},
        )
        try:
            urllib.request.urlopen(bad, timeout=15)
            failures.append("a 51-coefficient window was accepted")
        except urllib.error.HTTPError as e:
            if e.code != 400:
                failures.append(f"malformed window gave HTTP {e.code}, expected 400")
            else:
                print("malformed window      400 with a message")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print("serve smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
