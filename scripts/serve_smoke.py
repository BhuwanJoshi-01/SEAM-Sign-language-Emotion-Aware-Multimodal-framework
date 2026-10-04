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
        # The landing page must point at the avatar page. `/` used to be the only thing
        # reachable and advertised, which is how a working feature stayed invisible.
        if b'href="/avatar"' not in page:
            failures.append("index does not link to /avatar")

        health = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/health", timeout=15).read()
        )
        got = health.get("expects", {}).get("blendshapes")
        if got != [None, NM_DIM]:
            failures.append(
                f"health contract mismatch: NM_DIM={NM_DIM}, got {got!r}; body={health!r}"
            )

        # ── /avatar ──────────────────────────────────────────────────────────
        # This page was unlisted and unchecked for its whole life, and it was broken the
        # whole time. A route that is not in the smoke test is a route nobody looks at.
        av = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/avatar", timeout=15).read()
        print(f"GET /avatar           {len(av)} bytes")
        if b"<!doctype html>" not in av[:64].lower():
            failures.append("/avatar did not serve an HTML document")
        for needle, why in (
            (b"/api/demo/manifest", "/avatar never fetches the manifest"),
            (b"importmap", "/avatar has no import map, so bare `three` specifiers cannot resolve"),
        ):
            if needle not in av:
                failures.append(why)
        if b"cdn.jsdelivr" in av:
            failures.append(
                "/avatar loads three.js from a CDN; a product page that needs the network "
                "to render is not a product page"
            )
        if b"cdn.jsdelivr" in av or b"unpkg.com" in av:
            failures.append("/avatar must not reference a CDN")

        # Vendored three.js must actually be reachable, or the page is a red screen.
        for asset in (
            "/static/vendor/three/three.module.js",
            "/static/vendor/three/examples/jsm/controls/OrbitControls.js",
            "/static/vendor/three/examples/jsm/loaders/GLTFLoader.js",
        ):
            try:
                body = urllib.request.urlopen(f"http://127.0.0.1:{PORT}{asset}", timeout=30).read()
            except urllib.error.HTTPError as e:
                failures.append(f"{asset} -> HTTP {e.code}; /avatar cannot load three.js")
                continue
            if len(body) < 1000:
                failures.append(f"{asset} served {len(body)} bytes; that is not a library")
            else:
                print(f"GET {asset:52.52} {len(body)} bytes")

        # Path containment on both file routes. `{path:path}` patterns match `../`, so the
        # guard has to be on the resolved path rather than trusted from the route shape.
        for route in ("/static/vendor/", "/api/demo/"):
            url = f"http://127.0.0.1:{PORT}{route}..%2f..%2f..%2fetc%2fpasswd"
            try:
                body = urllib.request.urlopen(url, timeout=10).read()
                if b"root:" in body:
                    failures.append(f"{route} served /etc/passwd")
            except urllib.error.HTTPError as e:
                if e.code not in (400, 404):
                    failures.append(f"{route} traversal gave HTTP {e.code}, expected 400/404")

        # The manifest endpoint must degrade to a 200 with a reason, never a 500: the page
        # renders that reason, and a traceback would replace it.
        mf = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/demo/manifest", timeout=15).read()
        )
        if not isinstance(mf.get("clips"), list):
            failures.append(f"/api/demo/manifest has no clips list: {list(mf)}")
        elif mf["clips"] and not any(c.get("arm_a_smplerx", {}).get("glb") for c in mf["clips"]):
            failures.append("manifest lists clips but none carry a .glb")
        else:
            print(f"GET /api/demo/manifest      {len(mf['clips'])} clip(s)")

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
